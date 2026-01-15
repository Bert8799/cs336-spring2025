from gguf import Literal
from typing import Optional
from dataclasses import dataclass, field


# Simplified config: only need to change `dataset` field
_DATASET_CONFIGS = {
    "MATH": {
        "raw_question_placeholder": "{problem}",
        "raw_solution_placeholder": "{solution}",
        "extract_function": "boxed",
    },
    "gsm8k": {
        "raw_question_placeholder": "{question}",
        "raw_solution_placeholder": "{answer}",
        "extract_function": "raw",
    },
}


@dataclass
class BaseConfig:
    seed: int = 42
    data_dir: str = "data"
    model_dir: str = "data/model/Qwen/Qwen2.5-Math-1.5B"

    test_model_dir: str = "result/alignment/grpo/GRPO_MATH_3*256*64"
    
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    prompt_placeholder: str = "{question}"

    question_placeholder: str = "{question}"
    answer_placeholder: str = "{answer}"

    filter_long_ratio: float = 0.8

    # Only need to change this field!
    dataset: str = "MATH"  # Options: "MATH", "gsm8k"

    # Auto-computed fields based on dataset
    test_output_path: str = field(init=False)
    validation_input_path: str = field(init=False)
    baseline_output_path: str = field(init=False)
    raw_question_placeholder: str = field(init=False)
    raw_solution_placeholder: str = field(init=False)
    extract_function: str = field(init=False)

    def __post_init__(self):
        # Compute paths based on dataset
        self.test_output_path = f"result/alignment/grpo/grpo_gsm8k_{self.dataset}.jsonl"
        self.validation_input_path = f"{self.data_dir}/{self.dataset}/validation.jsonl"
        self.baseline_output_path = f"result/alignment/baseline/baseline_r1_zero_{self.dataset}.jsonl"

        # Get dataset-specific configs
        config = _DATASET_CONFIGS.get(self.dataset, _DATASET_CONFIGS[self.dataset])
        self.raw_question_placeholder = config["raw_question_placeholder"]
        self.raw_solution_placeholder = config["raw_solution_placeholder"]
        self.extract_function = config["extract_function"]


@dataclass(frozen=True)
class SFTConfig:
    seed: int = 42
    data_dir: str = "data"
    model_dir: str = "data/model/Qwen/Qwen2.5-Math-1.5B"
    dataset: str = "MATH"  # Options: "MATH", "gsm8k"
    train_data: str = "stf.jsonl"
    output_dir: str = "result/alignment/sft"

    gpu_memory_utilization: float = 0.75

    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    prompt: str = "prompt"
    response: str = "response"
    ground_truth: str = "ground_truth"
    question: str = "question"
    answer: str = "answer"

    dtype: str = "bfloat16"
    device_train: str = "cuda:0"
    device_eval: str = "cuda:1"

    # Use a subset of training data instead of full data
    train_data_size: int = 1024
    train_steps: int = 512
    train_batch_size: int = 32
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8 # train_batch_size // micro_batch_size

    eval_batch_size: int = 256
    eval_interval: int = 16

    lr: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float | None = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9

    wandb_project: str = "cs336_alignment_sft"
    wandb_entity: str = "jiangningning"
    wandb_run_name: str = field(init=False)

    def __post_init__(self):
        # Generate wandb run name
        object.__setattr__(self, "wandb_run_name", f"SFT_{self.dataset}_{self.train_data_size}*{self.train_steps}")


@dataclass(frozen=True)
class EIConfig(SFTConfig):
    # Original model's accuracy is too low
    # so we use the SFT model as the starting point
    model_dir: str = "result/alignment/sft/SFT_MATH_1024*512"
    output_dir: str = "result/alignment/ei"

    train_data_size: Optional[int] = None
    train_steps: int = 128
    # Avoid cuda out of memory
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    ei_batch_size: int = 512
    ei_steps: int = 5
    G: int = 4

    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4

    wandb_run_name: str = field(init=False)

    def __post_init__(self):
        # Generate wandb run name for EI
        object.__setattr__(self, "wandb_run_name", f"EI_{self.dataset}_{self.G}*{self.ei_batch_size}")


@dataclass(frozen=True)
class GRPOConfig(SFTConfig):
    # Original model's accuracy is too low
    # so we use the SFT model as the starting point
    model_dir: str = "result/alignment/sft/SFT_MATH_1024*512"
    dataset: str = "MATH"
    train_data: str = "train.jsonl"
    output_dir: str = "result/alignment/grpo"

    train_data_size: Optional[int] = None 
    train_steps: Optional[int] = None # on policy
    train_batch_size: int = 256 # 1 rollout, 1 step
    # Avoid cuda out of memory
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 64

    eval_batch_size: int = 1024 # for SFT MATH
    # eval_batch_size: int = 512 # for base gsm8k
    eval_interval: int = 8

    n_grpo_steps: int = 64 # for SFT MATH
    # n_grpo_steps: int = 256 # for base gsm8k
    rollout_batch_size: int = 256
    group_size: int = 8
    epochs_per_rollout_batch: int = 3 # if 1, on policy ==> reinforce
    advantage_eps: float = 1e-6
    cliprange: float = 0.2
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "grpo_clip"
    masked_type: Literal[0, 1] = 1
    use_std_normalization: bool = False

    lr: float = 3e-5
    weight_decay: float = 0.0
    adam_beta2: float = 0.95

    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4

    wandb_project: str = "cs336_alignment_grpo"
    wandb_run_name: str = field(init=False)

    def __post_init__(self):
        # Generate wandb run name for GRPO
        object.__setattr__(self, "wandb_run_name", f"GRPO-{self.dataset}-{self.epochs_per_rollout_batch}epochs-{self.n_grpo_steps}steps")