import torch
from einops import rearrange
from typing import Callable, List, Literal

from cs336_alignment.alignment.sft_utils import masked_normalize


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
    return_metadata: bool = False
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards_list = []
    for rollout, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(rollout, gt)
        
        final_score = scores["format_reward"] * 0.1 + scores["answer_reward"] * 0.9
        rewards_list.append(final_score)
    raw_rewards = torch.tensor(rewards_list)
    
    # Reshape to (n_prompts_per_rollout_batch, group_size)
    rewards_grouped = rearrange(raw_rewards, '(n g) -> n g', g=group_size)
    advantages = rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        advantages /= (rewards_grouped.std(dim=-1, keepdim=True) + advantage_eps)
    # Reshape back to (rollout_batch_size,)
    advantages = rearrange(advantages, 'n g -> (n g)')
    if return_metadata:
        metadata = {
            "mean_reward": raw_rewards.mean().item(),
            "std_reward": raw_rewards.std().item(),
        }
    else:
        metadata = {}
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_loss = advantages * prob_ratio
    clipped_ratio = torch.clamp(prob_ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = advantages * clipped_ratio
    loss = - torch.min(unclipped_loss, clipped_loss)
    # Metadata: whether each token was clipped
    was_clipped = (loss != - unclipped_loss).float().detach()  # shape (batch_size, sequence_length)
    metadata = {
        "was_clipped": was_clipped,
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs: (batch_size, sequence_length), 
            per-token log-probabilities from the policy being trained.
        loss_type: 
            One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: (batch_size, 1),
            Required if loss_type == "no_baseline".
        advantages: (batch_size, 1),
            Required for "reinforce_with_baseline" and "grpo_clip".
        old_log_probs: shape (batch_size, sequence_length),
            Required for "grpo_clip".
        cliprange: 
            Required for "grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        - loss (batch_size, sequence_length), per-token loss.
        - metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip)
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    masked_type: int = 0, # 0 for masked_mean, 1 for masked_normalize
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if masked_type == 0:
        loss = masked_mean(
            loss,
            response_mask,
            dim=1,
        ).mean() / gradient_accumulation_steps
    elif masked_type == 1:
        normalize_constant = response_mask.sum(dim=-1).max()
        loss = masked_normalize(
            loss,
            response_mask,
            normalize_constant,
            dim=-1,
        ).mean() / gradient_accumulation_steps
    else:
        ValueError(f"Unknown masked_func: {masked_type}; 0 or 1")
    loss.backward()
    return loss, metadata