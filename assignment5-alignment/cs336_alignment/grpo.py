import wandb
import pandas as pd
from typing import List
from torch.amp import autocast
from vllm import LLM, SamplingParams

from cs336_alignment.grpo_utils import *
from cs336_alignment.config import GRPOConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.baseline import (
    evaluate_vllm, get_prompts, get_sampling_params
)
from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output, get_response_log_probs
)
from cs336_alignment.sft import (
    run_training, load_policy_into_vllm_instance
)


def rollout(
    policy: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    groud_truth: List[str],
    use_tqdm: bool = False,
) -> pd.DataFrame:
    """Generate rollouts from the policy given the prompts."""
    responses = policy.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    generated = [[output.text for output in resp.outputs] for resp in responses] # List[List[str]]

    df = pd.DataFrame({
        "prompt": prompts,
        "response": generated,
        "ground_truth": groud_truth
    })

    return df.explode(column="response")


# TODO: the effect is not good, why?
def grpo_train(cfg: GRPOConfig, policy, tokenizer, old_policy, optimizer, train_data: pd.DataFrame, eval_data):
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    assert cfg.rollout_batch_size % cfg.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size
    assert cfg.train_batch_size >= cfg.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    # note: micro_batch_size is = train_batch_size // gradient_accumulation_steps
    #       1 rollout, 1 step ==> train_batch_size == rollout_batch_size
    #       ==> n_microbatches_per_rollout_batch == gradient_accumulation_steps
    n_microbatches_per_rollout_batch = cfg.rollout_batch_size // cfg.micro_batch_size
    with open(cfg.prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=cfg.seed,
        max_tokens=cfg.sampling_max_tokens,
        min_tokens=cfg.sampling_min_tokens,
        n=cfg.group_size,
    )
    eval_params = get_sampling_params()

    final_acc = None
    for step in range(cfg.n_grpo_steps):
        # rollout
        load_policy_into_vllm_instance(policy, old_policy)
        batch = train_data.sample(n_prompts_per_rollout_batch)
        prompts = get_prompts(prompt_template, batch['prompt'].to_list())
        rollout_batch = rollout(
            old_policy,
            sampling_params,
            prompts,
            batch['ground_truth'].to_list(),
            use_tqdm=False,
        )
        assert len(rollout_batch) == cfg.rollout_batch_size, "Rollout batch size mismatch."

        # compute advantages and rewards
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_batch['response'].to_list(),
            rollout_batch['ground_truth'].to_list(),
            cfg.group_size,
            cfg.advantage_eps,
            normalize_by_std=cfg.use_std_normalization,
            return_metadata=True
        )

        print(f"Step {step+1}/{cfg.n_grpo_steps} Rollout Rewards: "
              f"Mean: {metadata['mean_reward']:.3f}, "
              f"Std: {metadata['std_reward']:.3f}, "
              f"Max: {metadata['max_reward']:.3f}, "
              f"Min: {metadata['min_reward']:.3f}")

        dicts = tokenize_prompt_and_output(
            rollout_batch['prompt'].tolist(),
            rollout_batch['response'].tolist(),
            tokenizer
        )
        input_ids = dicts["input_ids"]
        labels = dicts["labels"]
        response_masks = dicts["response_mask"]

        # NOTE: if epochs_per_rollout_batch == 1,  we don't need compute old logprobs here
        # Compute old policy log probs for clipping
        old_policy_log_probs_list = []
        for mb_idx in range(n_microbatches_per_rollout_batch):
            start_idx = mb_idx * cfg.micro_batch_size
            end_idx = start_idx + cfg.micro_batch_size

            mb_input_ids = input_ids[start_idx:end_idx].to(cfg.device_train)
            mb_labels = labels[start_idx:end_idx].to(cfg.device_train)

            with autocast(device_type=cfg.device_train, dtype=torch.bfloat16):
                old_policy_response = get_response_log_probs(
                    policy, mb_input_ids, mb_labels, 
                    return_token_entropy=False,
                    inference_mode=True,
                )
                # we only need old log probs for clipping
                # store them in cpu to save gpu memory
                old_policy_log_probs_list.append(old_policy_response['log_probs'].cpu())
        old_policy_log_probs = torch.cat(old_policy_log_probs_list, dim=0)

        # GRPO training step
        for epoch in range(cfg.epochs_per_rollout_batch):
            loss_accum = 0.0
            entropy_accum = 0.0
            for mb_idx in range(n_microbatches_per_rollout_batch):
                start_idx = mb_idx * cfg.micro_batch_size
                end_idx = start_idx + cfg.micro_batch_size

                mb_input_ids = input_ids[start_idx:end_idx].to(cfg.device_train)
                mb_labels = labels[start_idx:end_idx].to(cfg.device_train)
                mb_response_masks = response_masks[start_idx:end_idx].to(cfg.device_train)
                mb_advantages = advantages[start_idx:end_idx].unsqueeze(-1).to(cfg.device_train)
                mb_raw_rewards = raw_rewards[start_idx:end_idx].unsqueeze(-1).to(cfg.device_train)

                mb_old_policy_log_probs = old_policy_log_probs[start_idx:end_idx].to(cfg.device_train)

                assert len(mb_input_ids.shape) == len(mb_advantages.shape), "Microbatch input shape mismatch."

                with autocast(device_type=cfg.device_train, dtype=torch.bfloat16):
                    policy_response = get_response_log_probs(
                        policy, mb_input_ids, mb_labels, 
                        return_token_entropy=True
                    )
                    policy_log_probs = policy_response['log_probs']
                    policy_entropy = policy_response['token_entropy']

                    loss, _ = grpo_microbatch_train_step(
                        policy_log_probs,
                        mb_response_masks,
                        cfg.gradient_accumulation_steps,
                        cfg.loss_type,
                        mb_raw_rewards,
                        mb_advantages,
                        mb_old_policy_log_probs,
                        cfg.cliprange,
                        masked_type=cfg.masked_type,
                    )
                    loss_accum += loss.item()
                    entropy_accum += policy_entropy.mean().item() / cfg.gradient_accumulation_steps

            # Since n_microbatches_per_rollout_batch == gradient_accumulation_steps
            # we can step the optimizer here
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
                "train/loss": loss_accum,
                "train/entropy": entropy_accum,
                "train/mean_reward": metadata['mean_reward'],
                "train_step": step*cfg.epochs_per_rollout_batch + epoch + 1,
            })

            print(f"Epoch {epoch+1}/{cfg.epochs_per_rollout_batch}: "
                  f"Loss: {loss_accum:.3f}, "
                  f"Entropy: {entropy_accum:.3f}")
            

        if (step+1) % cfg.eval_interval == 0:
            load_policy_into_vllm_instance(policy, old_policy)
            batch = eval_data.sample(cfg.eval_batch_size)
            prompts = get_prompts(prompt_template, batch['prompt'].tolist())
            all_records = evaluate_vllm(
                old_policy,
                r1_zero_reward_fn,
                eval_params,
                prompts,
                batch['ground_truth'].tolist(),
                use_tqdm=False,
            )
            accuracy = sum(r.get("reward", 0.0) for r in all_records) / len(all_records)
            format_accuracy = sum(r.get("format_reward", 0.0) for r in all_records) / len(all_records)
            answer_accuracy = sum(r.get("answer_reward", 0.0) for r in all_records) / len(all_records)

            final_acc = accuracy
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/format_accuracy": format_accuracy,
                "eval/answer_accuracy": answer_accuracy,
                "eval_step": (step+1)*cfg.epochs_per_rollout_batch,
            })

            print(f"Evaluation at step {step+1}/{cfg.n_grpo_steps}: "
                  f"Accuracy: {accuracy:.3f}, "
                  f"Format Accuracy: {format_accuracy:.3f}, "
                  f"Answer Accuracy: {answer_accuracy:.3f}")

            policy.save_pretrained(save_directory=f'{cfg.output_dir}/{cfg.wandb_run_name}')
            tokenizer.save_pretrained(save_directory=f'{cfg.output_dir}/{cfg.wandb_run_name}')
    
    return metadata['mean_reward'], final_acc


def run_grpo():
    """Run GRPO training."""
    grpo_cfg = GRPOConfig()
    run_training(grpo_cfg, grpo_train)


if __name__ == "__main__":
    run_grpo()