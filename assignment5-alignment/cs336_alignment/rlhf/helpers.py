import json
import random
import torch
import regex as re
from tqdm import tqdm
from xopen import xopen
from pathlib import Path
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SPLIT_PATTERN = r"\n\n(?=Human:|Assistant:)"
CLEAN_PATTERN = r"^(Human:|Assistant:)\s*"


def split_data(data_path: str | Path, ratio: float = 0.95, seed: int = 42):
    data_path = Path(data_path)
    samples = []
    
    files = list(data_path.glob("*.gz"))
    print(f"Found {len(files)} files.")

    for file in tqdm(files, desc="Processing"):
        with xopen(file, "rt") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                except:
                    continue

                raw_chosen = sample.get("chosen", "").strip()
                raw_rejected = sample.get("rejected", "").strip()
                
                parts_chosen = re.split(SPLIT_PATTERN, raw_chosen)
                parts_rejected = re.split(SPLIT_PATTERN, raw_rejected)

                parts_chosen = [p.strip() for p in parts_chosen if p.strip()]
                parts_rejected = [p.strip() for p in parts_rejected if p.strip()]

                if len(parts_chosen) != 2 or len(parts_rejected) != 2:
                    continue

                p1 = re.sub(CLEAN_PATTERN, "", parts_chosen[0]).strip()
                r1 = re.sub(CLEAN_PATTERN, "", parts_chosen[1]).strip()
                r2 = re.sub(CLEAN_PATTERN, "", parts_rejected[1]).strip()

                if not p1 or not r1 or not r2:
                    continue

                if p1 != re.sub(CLEAN_PATTERN, "", parts_rejected[0]).strip():
                    continue

                samples.append({
                    "instruction": p1,
                    "chosen": r1,
                    "rejected": r2,
                })

    print(f"Total valid samples: {len(samples)}")

    random.seed(seed)
    random.shuffle(samples)

    train_size = int(len(samples) * ratio)
    train_data = samples[:train_size]
    test_data = samples[train_size:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    def save_jsonl(data, filename):
        with open(data_path / filename, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(test_data, "test.jsonl")
    print("Done.")


### >>> for test
def _get_log_probs(
    lm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids = input_ids.to(lm.device)
    eos = torch.tensor([[tokenizer.eos_token_id]])
    input_ids = torch.cat([input_ids, eos], dim=-1)
    labels = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    with torch.no_grad():
        logits = lm(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs.sum()


def compute_per_instance_dpo_loss(
    lm: AutoModelForCausalLM,
    lm_ref: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    beta: float,
    prompt: torch.Tensor,
    response_chosen: torch.Tensor,
    response_rejected: torch.Tensor,
) -> torch.Tensor:
    with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
        prompt_template = f.read()
    prompt_chosen = prompt_template.format(
        instruction=prompt,
        response=response_chosen
    )
    prompt_rejected = prompt_template.format(
        instruction=prompt,
        response=response_rejected
    )

    log_probs_chosen = _get_log_probs(lm, tokenizer, prompt_chosen)
    log_probs_rejected = _get_log_probs(lm, tokenizer, prompt_rejected)

    ref_log_probs_chosen = _get_log_probs(lm_ref, tokenizer, prompt_chosen)
    ref_log_probs_rejected = _get_log_probs(lm_ref, tokenizer, prompt_rejected)

    diff = (log_probs_chosen - log_probs_rejected)
    ref_diff = (ref_log_probs_chosen - ref_log_probs_rejected).to(lm.device)
    return - F.logsigmoid(beta * (diff - ref_diff))
### <<< for test