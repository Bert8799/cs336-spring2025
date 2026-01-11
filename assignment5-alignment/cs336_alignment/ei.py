import pandas as pd
from vllm import SamplingParams

from cs336_alignment.sft_utils import *
from cs336_alignment.config import EIConfig
from cs336_alignment.baseline import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import (
    sft_train, run_training, load_policy_into_vllm_instance
)


def ei_train(cfg, model, tokenizer, vllm, optimizer, train_data, eval_data):
    """
    Expert Iteration training loop.

    Args:
        cfg: EIConfig
        model: The model to train
        tokenizer: Tokenizer
        vllm: vLLM instance for generation
        optimizer: Optimizer
        train_data: Training dataset (used for sampling questions)
        eval_data: Evaluation dataset
    """
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=cfg.seed,
        max_tokens=cfg.sampling_max_tokens,
        min_tokens=cfg.sampling_min_tokens,
        n=cfg.G,
    )

    for step in range(cfg.ei_steps):
        load_policy_into_vllm_instance(model, vllm)
        ei_batch = train_data.sample(cfg.ei_batch_size)
        ei_data = evaluate_vllm(
            vllm,
            r1_zero_reward_fn,
            sampling_params,
            ei_batch[cfg.prompt].tolist(),
            ei_batch[cfg.ground_truth].tolist(),
            use_tqdm=True,
            ei_mode=True,
        )
        assert len(ei_data) >= cfg.micro_batch_size, "Not enough data with reward >= 1.0."
        loss, acc = sft_train(
            cfg, model, tokenizer, vllm, optimizer,
            pd.DataFrame(ei_data), eval_data,
            start_step=step*cfg.train_steps
        )
        print(f"EI Step {step+1} - SFT Loss: {loss:.3f}, Eval Accuracy: {acc:.3f}")


def run_ei():
    """Run Expert Iteration training."""
    ei_cfg = EIConfig()
    run_training(ei_cfg, ei_train)

if __name__ == "__main__":
    run_ei()