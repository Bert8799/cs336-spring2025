import torch
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

def get_response_log_probs(lm, input_ids, attention_mask, loss_mask):
    """
    Computes log probabilities for the response tokens only using functional CE.
    Args:
        lm: The language model.
        input_ids: (B, L)
        attention_mask: (B, L)
        loss_mask: (B, L) 1 for response tokens, 0 otherwise.
    Returns:
        (B,) tensor containing the sum of log-probs for the response.
    """
    # Forward pass
    logits = lm(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Slice for next-token prediction
    # (B, L-1, V)
    shift_logits = logits[..., :-1, :].contiguous()
    # (B, L-1)
    shift_labels = input_ids[..., 1:].contiguous()
    # (B, L-1) - mask aligns with the label being predicted
    shift_mask = loss_mask[..., 1:].contiguous()

    # Use functional CE to save overhead (no object creation)
    # reduction='none' computes loss per token
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1), 
        reduction='none'
    )
    
    # Reshape back to (B, L-1)
    token_losses = token_losses.view(shift_labels.shape)
    
    # CE is -log(p), so -CE is log(p)
    token_log_probs = -token_losses * shift_mask
    
    response_log_probs = token_log_probs.sum(dim=-1)

    # Valid check (prevent NaN/Inf on empty responses)
    is_valid = shift_mask.sum(dim=-1) > 0
    return torch.where(
        is_valid,
        response_log_probs,
        torch.tensor(-100.0, device=response_log_probs.device)
    )


def dpo_loss(pi_logps, ref_logps, beta=0.1):
    """
    Pure Math: Calculates DPO loss and metrics.
    Loss = -log sigmoid( beta * ( (pi_w - pi_l) - (ref_w - ref_l) ) )
    """
    pi_chosen, pi_rejected = pi_logps
    ref_chosen, ref_rejected = ref_logps

    chosen_diff = pi_chosen - ref_chosen
    rejected_diff = pi_rejected - ref_rejected
    
    reward_margin = chosen_diff - rejected_diff
    loss = -F.logsigmoid(beta * reward_margin)
    accuracy = (reward_margin > 0).float()
    
    return loss.mean(), reward_margin.mean(), accuracy.mean()


def to_device(batch, device, prefix=""):
    """Recursive helper to move batch to device non-blocking."""
    p = f"{prefix}_" if prefix else ""
    # if batch is raw_data, with prefix 'chosen' or 'rejected'
    # elif batch is already prefixed
    return {
        f"input_ids": batch[f"{p}input_ids"].to(device, non_blocking=True),
        f"attention_mask": batch[f"{p}attention_mask"].to(device, non_blocking=True),
        f"loss_mask": batch[f"{p}loss_mask"].to(device, non_blocking=True),
    }


def create_dpo_model_pair(model_name, policy_device, ref_device):
    """Factory function to load Policy and Reference models."""
    logger.info(f"Creating DPO model pair from {model_name}")
    logger.info(f"Loading Policy model on {policy_device}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=policy_device
    )
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    
    logger.info(f"Loading Reference model on {ref_device}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=ref_device
    )
    ref_model.eval()
    
    return policy, ref_model


def iterate_batches(dataset, batch_size, collate_fn, shuffle=True, drop_last=True, num_workers=4):
    """Infinite iterator with multi-process pre-fetching."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True 
    )
    while True:
        for batch in dataloader:
            yield batch