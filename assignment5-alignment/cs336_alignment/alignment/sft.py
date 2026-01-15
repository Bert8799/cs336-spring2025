import torch
import wandb
import random
import numpy as np
import pandas as pd
from vllm import LLM
from torch.amp import autocast
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import (
    PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
)

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.alignment.sft_utils import *
from cs336_alignment.alignment.config import SFTConfig
from cs336_alignment.alignment.baseline import (
    evaluate_vllm, get_prompts, get_sampling_params, destroy_vllm
)


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_optimizer(cfg, model):
    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_decay, "weight_decay": cfg.weight_decay},
        {"params": params_not_decay, "weight_decay": 0.0},
    ]
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        fused=True,
    )
    return optimizer


def sft_train(cfg: SFTConfig, model, tokenizer, vllm, optimizer, train_data, eval_data, start_step=0):
    """
    The main SFT training loop with mixed precision training support.

    Args:
        cfg: SFT configuration
        model: The model to train
        tokenizer: Tokenizer
        vllm: vLLM instance for evaluation
        optimizer: Optimizer
        train_data: Training dataset
        eval_data: Evaluation dataset
    """
    assert cfg.train_steps % cfg.eval_interval == 0, \
        "train_steps must be divisible by eval_interval"
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"

    with open(cfg.prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    sampling_params = get_sampling_params()

    if cfg.train_data_size is not None:
        print(f'Length(train_data): {cfg.train_data_size} / {len(train_data)}')
        train_data = train_data.sample(cfg.train_data_size)

    final_loss, final_acc = None, None
    for step in range(start_step, start_step + cfg.train_steps):
        loss_accum = 0.0
        entropy_accum = 0.0

        # Micro-batch training with mixed precision
        for _ in range(cfg.gradient_accumulation_steps):
            batch = train_data.sample(cfg.micro_batch_size)
            dicts = tokenize_prompt_and_output(
                batch[cfg.prompt].tolist(),
                batch[cfg.response].tolist(),
                tokenizer
            )

            input_ids = dicts['input_ids'].to(cfg.device_train)
            labels = dicts['labels'].to(cfg.device_train)
            response_mask = dicts['response_mask'].to(cfg.device_train)

            # bf16 don't need GradScaler
            with autocast(device_type=cfg.device_train, dtype=torch.bfloat16):
                response_log_probs = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=True
                )
                policy_log_probs = response_log_probs['log_probs']
                token_entropy = response_log_probs['token_entropy']

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    cfg.gradient_accumulation_steps,
                    normalize_constant=response_mask.sum(dim=-1).max(),
                )

            loss_accum += loss.item()
            entropy_accum += token_entropy.mean().item() / cfg.gradient_accumulation_steps

        # Gradient clipping and optimizer step
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        optimizer.zero_grad()

        final_loss = loss_accum
        wandb.log({
            "train/loss": loss_accum,
            "train/entropy": entropy_accum,
            "train_step": step + 1,
        })

        print(f"Step {step+1}/{start_step+cfg.train_steps}: "
              f"Loss: {loss_accum:.3f}, "
              f"Entropy: {entropy_accum:.3f}")

        # Evaluation
        if (step + 1) % cfg.eval_interval == 0:
            load_policy_into_vllm_instance(model, vllm)
            batch = eval_data.sample(cfg.eval_batch_size)
            prompts = get_prompts(prompt_template, batch['prompt'].tolist())
            all_records = evaluate_vllm(
                vllm,
                r1_zero_reward_fn,
                sampling_params,
                prompts,
                batch['ground_truth'].tolist(),
                use_tqdm=True,
            )

            accuracy = sum(r.get("reward", 0.0) for r in all_records) / len(all_records)
            format_accuracy = sum(r.get("format_reward", 0.0) for r in all_records) / len(all_records)
            answer_accuracy = sum(r.get("answer_reward", 0.0) for r in all_records) / len(all_records)

            final_acc = accuracy

            wandb.log({
                "eval/accuracy": accuracy,
                "eval/format_accuracy": format_accuracy,
                "eval/answer_accuracy": answer_accuracy,
                "eval_step": step + 1,
            })

            print(f"Evaluation at step {step+1}/{start_step+cfg.train_steps}: "
                  f"Accuracy: {accuracy:.3f}, "
                  f"Format Accuracy: {format_accuracy:.3f}, "
                  f"Answer Accuracy: {answer_accuracy:.3f}")

            model.save_pretrained(save_directory=f'{cfg.output_dir}/{cfg.wandb_run_name}')
            tokenizer.save_pretrained(save_directory=f'{cfg.output_dir}/{cfg.wandb_run_name}')

    return final_loss, final_acc


def run_training(cfg: SFTConfig, train_fn, **train_kwargs):
    """
    Unified entry point for training workflows (SFT, EI, etc.).

    Handles wandb init, model setup, training, and cleanup.

    Args:
        cfg: Configuration object (SFTConfig, EIConfig, etc.)
        train_fn: Training function with signature:
                 (cfg, model, tokenizer, vllm, optimizer, train_data, eval_data, scaler, **train_kwargs)
        use_amp: If True, enable automatic mixed precision training
        **train_kwargs: Additional keyword arguments passed to train_fn

    Returns:
        The return value from train_fn
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize wandb
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        entity=cfg.wandb_entity,
        config=vars(cfg),
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    model.gradient_checkpointing_enable()
    # Required by HF when using gradient checkpointing.
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"Loaded model from {cfg.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    optimizer = get_optimizer(cfg, model)

    # Initialize vLLM
    vllm = init_vllm(
        model_id=cfg.model_dir,
        device=cfg.device_eval,
        seed=cfg.seed,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    # Load datasets
    train_data_path = f"{cfg.data_dir}/{cfg.dataset}/{cfg.train_data}"
    print(f"Loading training data from {train_data_path}")
    train_data = pd.read_json(train_data_path, lines=True)
    eval_data_path = f"{cfg.data_dir}/{cfg.dataset}/validation.jsonl"
    print(f"Loading evaluation data from {eval_data_path}")
    eval_data = pd.read_json(eval_data_path, lines=True)

    # Run training
    result = train_fn(cfg, model, tokenizer, vllm, optimizer, train_data, eval_data, **train_kwargs)

    destroy_vllm()

    # Connection Failed, need ctrl+c to stop
    # wandb.finish()

    return result


def run_sft():
    sft_cfg = SFTConfig()
    return run_training(sft_cfg, sft_train)


if __name__ == "__main__":
    run_sft()