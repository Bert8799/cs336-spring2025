import torch
import torch.nn as nn
import numpy as np
import os, typing
from einops import rearrange


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