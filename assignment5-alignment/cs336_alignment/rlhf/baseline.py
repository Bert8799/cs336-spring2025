import os
import torch
import logging
import regex as re
import pandas as pd
from pathlib import Path
from typing import List, Callable, Any, Tuple, Optional, Dict
from dataclasses import dataclass
from contextlib import contextmanager

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from cs336_alignment.rlhf.config import BaseConfig

# --- Configuration & Logging ---
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


# --- Data Structures ---
@dataclass
class EvalItem:
    """Standardized interface for a single evaluation example."""
    prompt: str                  # The instruction prompt payload
    ground_truth: Any            # The expected answer
    metadata: Dict[str, Any]     # Extra info for debugging (e.g., subject, original question)


# --- Infrastructure / Lifecycle Management ---
@contextmanager
def vllm_context(model_name: str, gpu_memory_utilization: float):
    """
    Robust resource management for vLLM.
    Guarantees cleanup of distributed process groups and VRAM.
    """
    llm = None
    try:
        print(f"Initializing vLLM with model: {model_name}")
        llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
        yield llm
    finally:
        destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
        print("vLLM resources and VRAM released.")


# --- Parsers (Strategy Pattern) ---
def mmlu_parser(response: str) -> Optional[str]:
    text = response.strip()
    # Strategy 1: Strict Format
    match = re.search(r"The correct answer is\s*[:\-\s]*[\*\(]*([A-D])[\*\)\.]*", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # Strategy 2: Heuristics (Start/End)
    if match := re.match(r"^[\*\(]*([A-D])[\*\)\.]", text): return match.group(1).upper()
    if match := re.search(r"[\s\.\*]([A-D])[\.\*]*$", text): return match.group(1).upper()
    
    return None


def gsm8k_parser(response: str) -> Optional[float]:
    text = response.strip()
    text = text.replace(',', '') # Normalization
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


# --- Data Loaders (Adapter Pattern) ---
def load_mmlu_data(cfg: BaseConfig) -> List[EvalItem]:
    eval_dir = Path(cfg.data_dir) / cfg.mmlu_dataset / "val"
    prompt_path = Path(cfg.prompt_dir) / cfg.mmlu_prompt_file
    
    with open(prompt_path, 'r') as f:
        template = f.read()

    items = []
    print(f"Loading MMLU data from {eval_dir}")
    for file in eval_dir.glob("*.csv"):
        # Vectorized read but iterative processing for templating
        df = pd.read_csv(file, names=['question', 'A', 'B', 'C', 'D', 'ground_truth'])
        subject = file.name.rsplit("_", 1)[0]
        
        # Use simple iteration here as template formatting is difficult to vectorize purely
        for _, row in df.iterrows():
            formatted_prompt = template.format(
                subject=subject,
                question=row['question'],
                options=[row['A'], row['B'], row['C'], row['D']]
            )
            items.append(EvalItem(
                prompt=formatted_prompt,
                ground_truth=row['ground_truth'],
                metadata={'subject': subject, 'question': row['question']}
            ))
    return items


def load_gsm8k_data(cfg: BaseConfig) -> List[EvalItem]:
    data_path = Path(cfg.data_dir) / cfg.gsm8k_dataset / "validation.jsonl"
    prompt_path = Path(cfg.prompt_dir) / cfg.question_only_prompt_file
    
    with open(prompt_path, 'r') as f:
        template = f.read()

    print(f"Loading GSM8K data from {data_path}")
    df = pd.read_json(data_path, lines=True)
    
    items = []
    for _, row in df.iterrows():
        items.append(EvalItem(
            prompt=template.format(question=row['prompt']),
            ground_truth=float(row['ground_truth'].replace(',', '')), # Type casting for numerical stability
            metadata={'question': row['prompt']}
        ))
    return items


def load_ssft_data(cfg: BaseConfig) -> List[EvalItem]:
    data_path = Path(cfg.data_dir) / cfg.simple_safety_dataset / "simple_safety_tests.csv"
    print(f"Loading SSFT data from {data_path}")
    df = pd.read_csv(data_path, usecols=['prompts_final'])
    items = []
    for _, row in df.iterrows():
        items.append(EvalItem(
            prompt=row['prompts_final'],
            ground_truth=None,
            metadata={'question': row['prompts_final']}
        ))
    return items

# --- Core Evaluation Engine ---
def run_evaluation(
    cfg: BaseConfig,
    llm: LLM,
    dataset_name: str,
    data_loader: Callable[[BaseConfig], List[EvalItem]],
    parser: Callable[[str], Tuple[Any, bool]],
    comparator: Callable[[Any, Any], bool] = lambda x, y: x == y,
    custom_sampling_params: Optional[SamplingParams] = None,
    only_generate: bool = False,
):
    """
    Generic evaluation pipeline.
    
    Args:
        cfg: Configuration object.
        dataset_name: Identifier for logging/saving.
        data_loader: Function to load and format dataset into EvalItems.
        parser: Function to extract answer from raw LLM output.
        comparator: Function to compare prediction vs ground truth (e.g., exact match vs float tolerance).
    """
    # 1. Load Data
    eval_items = data_loader(cfg)
    print(f"[{dataset_name}] Loaded {len(eval_items)} examples.")
    
    if not eval_items:
        print(f"[{dataset_name}] No data found. Exiting.")
        return

    # 2. Prepare Prompts (Apply System Prompt)
    system_prompt_path = Path(cfg.prompt_dir) / cfg.system_prompt_file
    with open(system_prompt_path, 'r') as f:
        sys_template = f.read()
    
    # Batch prompt construction
    full_prompts = [sys_template.format(instruction=item.prompt) for item in eval_items]
    
    # 3. Inference (Model Lifecycle)
    if custom_sampling_params:
        sampling_params = custom_sampling_params
    else:
        sampling_params = SamplingParams(
            temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
            min_p=cfg.min_p, max_tokens=cfg.max_tokens, stop=cfg.stop, seed=cfg.seed
        )

    outputs = llm.generate(full_prompts, sampling_params)
    
    # 4. Parse & Metric Computation
    results = []
    correct_count = 0

    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text
        parsed_answer = parser(response_text)

        if not only_generate:
            # Check correctness
            is_correct = False
            if parsed_answer is not None:
                is_correct = comparator(parsed_answer, eval_items[i].ground_truth)
            
            if is_correct: correct_count += 1
            
            # Aggregate result
            results.append({
                **eval_items[i].metadata,
                'ground_truth': eval_items[i].ground_truth,
                'response': response_text,
                'answer': parsed_answer,
                'correct': is_correct,
            })
        else:
            results.append({
                **eval_items[i].metadata,
                'response': response_text,
            })

    # 5. Reporting & Serialization
    if not only_generate:
        accuracy = correct_count / len(eval_items)
        print(f"[{dataset_name}] Accuracy: {accuracy*100:.2f}% ({correct_count}/{len(eval_items)})")

    result_dir = Path(cfg.result_dir) / cfg.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    out_file = result_dir / f"{dataset_name}_{cfg.output_type}.jsonl"
    
    pd.DataFrame(results).to_json(out_file, lines=True, orient='records')
    print(f"Results saved to {out_file}")


# --- Main Execution ---
if __name__ == "__main__":
    config = BaseConfig()
    with vllm_context(config.model_name, config.gpu_memory_utilization) as llm_engine: 
        # Run MMLU
        run_evaluation(
            cfg=config,
            llm=llm_engine,
            dataset_name="mmlu",
            data_loader=load_mmlu_data,
            parser=mmlu_parser
            # Default comparator (equality) is sufficient for MMLU
        )

        # Run GSM8K (Requires float tolerance comparator)
        run_evaluation(
            cfg=config,
            llm=llm_engine,
            dataset_name="gsm8k",
            data_loader=load_gsm8k_data,
            parser=gsm8k_parser,
            comparator=lambda pred, gt: abs(pred - gt) < 1e-3
        )

        # Run SSFT (Only generation, no evaluation)
        sst_sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=1024, 
            stop=["# Query:"]
        )

        run_evaluation(
            cfg=config,
            llm=llm_engine,
            dataset_name="ssft",
            data_loader=load_ssft_data,
            parser=lambda x: x.strip(),  # No parsing needed
            custom_sampling_params=sst_sampling_params,
            only_generate=True
        )