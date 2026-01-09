import os
import logging
import json
import torch
import pandas as pd
from os import PathLike
from typing import List, Callable
from vllm import LLM, SamplingParams
from contextlib import contextmanager
from vllm.distributed.parallel_state import destroy_model_parallel

from cs336_alignment.config import BaseConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


config = BaseConfig()
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


def get_prompts(prompt_template, problems: List[str]) -> List[str]:
    """Generate prompts by filling in the template with each problem."""
    prompts = []
    for problem in problems:
        prompt = prompt_template.replace(config.prompt_placeholder, problem)
        prompts.append(prompt)
    return prompts


@contextmanager
def vllm_resource_scope(model_obj):
    try:
        yield model_obj
    finally:
        destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: SamplingParams,
    prompts: List[str],
    solutions: List[str],
    output_file: str | PathLike | None = None
) -> None:
    """Evaluate the model on the given prompts and solutions using the reward function.

    Args:
        vllm_model: The vLLM model to use for generation.
        eval_sampling_params: Sampling parameters for generation.
        prompts: List of problem prompts.
        solutions: List of ground truth solutions.
        reward_fn: Function to compute rewards given generated and ground truth solutions.
        output_file: Optional file path to save detailed results as CSV.
    Returns:
        A tuple containing:
            - List of reward dictionaries for each example.
            - List of generated solutions.
    """
    responses = vllm_model.generate(prompts, eval_sampling_params)
    generations_list = [resp.outputs[0].text for resp in responses]
    rewards_list = [reward_fn(gen_sol, gt_sol) for gen_sol, gt_sol in zip(generations_list, solutions)]

    if output_file:
        with open(output_file, "w") as f:
            for (prompt, gt_sol, gen_sol, rewards) in zip(
                prompts,solutions, generations_list, rewards_list
            ):
                record = {
                    "reward": rewards,
                    "ground_truth": gt_sol,
                    "generated": gen_sol,
                    "prompt": prompt,
                }
                f.write(json.dumps(record) + "\n")


def run_r1_zero_baseline():
    llm = LLM(model=config.model_dir)
    with open(config.prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    # Based on Dr. GRPO: stop when the model completes its answer
    # https://github.com/sail-sg/understand-r1-zero/blob/
    # c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    df = pd.read_json(config.validation_input_path, lines=True)
    questions = df[config.question_placeholder.strip("{}")].tolist()
    answers = df[config.answer_placeholder.strip("{}")].tolist()
    prompts = get_prompts(prompt_template, questions)
    print("Evaluating R1-Zero baseline")
    with vllm_resource_scope(llm):
        evaluate_vllm(
            llm,
            r1_zero_reward_fn,
            sampling_params,
            prompts,
            answers,
            output_file=config.baseline_output_path,
        )
    print("Evaluation complete.")


if __name__ == "__main__":
    run_r1_zero_baseline()