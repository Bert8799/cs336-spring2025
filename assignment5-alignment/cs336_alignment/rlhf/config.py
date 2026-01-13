from gguf import Literal
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConfig:
    seed: int = 42

    model_name: str = "data/model/Qwen/Qwen3-1.7B"

    data_dir: str = "data"
    result_dir: str = "result/rlhf/bl"

    mmlu_dataset: str = "mmlu"
    gsm8k_dataset: str = "gsm8k"
    alpaca_eval_dataset: str = "alpaca_eval"
    simple_safety_dataset: str = "simple_safety_tests"

    prompt_dir: str = "cs336_alignment/prompts"
    mmlu_prompt_file: str = "mmlu.prompts"
    question_only_prompt_file: str = "question_only.prompt"
    system_prompt_file: str = "zero_shot_system_prompt.prompt"
    
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 1024
    stop: Literal["# Query:"] = "# Query:"

    gpu_memory_utilization: float = 0.8