import os
import json
import torch
import logging
import pandas as pd
from os import PathLike
from typing import List, Callable
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from cs336_alignment.config import BaseConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


baseline_cfg = BaseConfig()
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


def get_prompts(prompt_template, problems: List[str]) -> List[str]:
    """Generate prompts by filling in the template with each problem."""
    prompts = []
    for problem in problems:
        prompt = prompt_template.replace(baseline_cfg.prompt_placeholder, problem)
        prompts.append(prompt)
    return prompts


def get_sampling_params() -> SamplingParams:
    """Get sampling parameters for vLLM generation."""
    # Based on Dr. GRPO: stop when the model completes its answer
    # https://github.com/sail-sg/understand-r1-zero/blob/
    # c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=baseline_cfg.seed
    )


def destroy_vllm():
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: SamplingParams,
    prompts: List[str],
    ground_truth: List[str],
    output_file: str | PathLike | None = None,
    use_tqdm: bool = False,
    ei_mode: bool = False,
) -> List[dict[str, float]]:
    """Evaluate the model on the given prompts using the reward function."""
    responses = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=use_tqdm)
    all_records = []

    for i, req in enumerate(responses):
        for output in req.outputs:
            gen = output.text
            metrics = reward_fn(gen, ground_truth[i]) 
            
            all_records.append({
                "prompt": prompts[i],
                "generated": gen,
                "ground_truth": ground_truth[i],
                **metrics,
            })

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for r in all_records: 
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not ei_mode:
        return all_records

    ei_dataset = []
    seen_prompts = set()
    
    for r in all_records:
        if r.get('reward', 0) >= 1.0 and r['prompt'] not in seen_prompts:
            ei_dataset.append({
                "prompt": r['prompt'], 
                "response": r['generated'], 
                "ground_truth": r['ground_truth']
            })
            seen_prompts.add(r['prompt'])
    print(f"Collect: {len(ei_dataset)} / {len(prompts)}")
    return ei_dataset


def evaluate_model(use_test_model: bool = False):
    """Evaluate a model on mathematical reasoning tasks.

    Args:
        use_test_model: If True, evaluate the test_model_dir; otherwise evaluate the base model_dir.
    """
    model_path = baseline_cfg.test_model_dir if use_test_model else baseline_cfg.model_dir
    output_path = baseline_cfg.test_output_path if use_test_model else baseline_cfg.baseline_output_path

    print(f"Evaluating {'test' if use_test_model else 'base'} model: {model_path}")
    print(f"Output: {output_path}")

    llm = LLM(model=model_path)
    with open(baseline_cfg.prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    sampling_params = get_sampling_params()
    df = pd.read_json(baseline_cfg.validation_input_path, lines=True)
    questions = df['prompt'].tolist()
    answers = df['ground_truth'].tolist()
    prompts = get_prompts(prompt_template, questions)
    evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        sampling_params,
        prompts,
        answers,
        output_file=output_path,
        use_tqdm=True,
    )
    destroy_vllm()
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate_model(use_test_model=True)