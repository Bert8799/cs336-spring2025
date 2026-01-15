from gguf import Literal
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseConfig:
    seed: int = 42

    model_name: str = "data/model/Qwen/Qwen3-1.7B"

    data_dir: str = "data"
    result_dir: str = "result/rlhf"
    output_dir: str = "bl"

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


@dataclass(frozen=True)
class RLHFConfig(BaseConfig):
    sft_output_dir: str = "sft"

    sft_dataset_path: str = "rlhf"
    sft_train_data: str = "train.jsonl.xz"
    sft_dev_data: str = "test.jsonl.xz"

    sft_epochs: int = 1
    # Reduce training time, 
    # originally (len(train_data) // micro_batch_size) * sft_epochs
    sft_train_steps: int = 512
    sft_seq_length: int = 512
    sft_batch_size: int = 32
    sft_gradient_accumulation_steps: int = 8
    # sft_eval_interval: int = 2000
    sft_eval_interval: int = 16
    sft_eval_steps: int = 16

    sft_learning_rate: float = 2e-5
    sft_warmp_ratio: float = 0.03
    sft_adam_beta1: float = 0.9
    sft_adam_beta2: float = 0.98
    sft_adam_eps: float = 1e-9
    sft_grad_clip: float = 1.0

    sft_wandb_project: str = "cs336-rlhf-sft"
    sft_wandb_entity: str = "jiangningning"
    sft_wandb_run_name: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "sft_wandb_run_name",
            f"sft-{self.model_name.split('/')[-1]}-{self.sft_train_steps}steps"
        )