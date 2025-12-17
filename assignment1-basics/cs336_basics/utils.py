import os
import typing
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Stable softmax implementation.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def log_softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """Computes the log-softmax of the input tensor along the specified dimension."""
    max_x = torch.max(x, dim=dim, keepdim=True).values
    stable_x = x - max_x
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_x), dim=dim, keepdim=True))
    return stable_x - log_sum_exp


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """Computes the cross-entropy loss between logits and targets."""
    vocab_size = logits.size(-1)
    logits = rearrange(logits, "... v -> (...) v", v=vocab_size)
    targets = rearrange(targets, "... -> (...)")
    log_probs = log_softmax(logits, dim=-1)
    loss = -log_probs[torch.arange(logits.size(0)), targets]
    return loss.mean()


def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    t: float = 1.0
) -> torch.Tensor:
    """Performs top-p (nucleus) sampling on the logits."""
    if t <= 0:
        raise ValueError("Temperature t must be greater than 0.")
    logits = logits / t
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    
    # Create a mask for tokens to keep
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the mask to the right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter the mask back to the original indices
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    
    # Set logits of removed tokens to -inf
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Sample from the filtered distribution
    probs = softmax(logits, dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1)
    
    return sampled_indices.squeeze(-1)


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input-target pairs from the dataset.
    Remenber to use mmap to load large datasets efficiently.
    """
    num_tokens = x.shape[0]
    start_indices = np.random.randint(0, num_tokens - context_length, size=batch_size)
    
    input_batch = np.stack([x[i:i + context_length] for i in start_indices])
    target_batch = np.stack([x[i + 1:i + context_length + 1] for i in start_indices])
    
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)
    
    return input_tensor, target_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """Saves the model and optimizer state dictionaries to a checkpoint file."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    device: torch.device = torch.device("cpu")
) -> int:
    """Loads the model and optimizer state dictionaries from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration