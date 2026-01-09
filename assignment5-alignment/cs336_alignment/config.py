from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConfig:
    data_dir: str = "data"
    model_dir: str = "data/model/Qwen/Qwen2.5-Math-1.5B"

    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    prompt_placeholder: str = "{question}"

    validation_input_path: str = "data/MATH/validation.jsonl"
    baseline_output_path: str = "result/baseline/baseline_r1_zero_MATH.jsonl"
    dataset: str = "MATH"  # Options: "MATH", "gsm8k"
    raw_question_placeholder: str = "{problem}" # Options: "{problem}", "{question}"
    raw_solution_placeholder: str = "{solution}" # Options: "{solution}", "{answer}"
    
    extract_function: str = "boxed"  # Options: "boxed", "raw"
    
    question_placeholder: str = "{question}"
    answer_placeholder: str = "{answer}"