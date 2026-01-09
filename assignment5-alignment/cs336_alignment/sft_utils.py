import torch
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and output strings for model input.

    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings corresponding to prompts.
        tokenizer: PreTrainedTokenizerBase instance for tokenization.
    Returns:
        A dictionary with tokenized 'input_ids', 'labels' and 'response_mask' tensors.
    """
    token_list = []
    mask_list = []
    max_seq_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_enc = tokenizer.encode(prompt, add_special_tokens=False)
        output_enc = tokenizer.encode(output, add_special_tokens=False)
        token_list.append(prompt_enc + output_enc)
        mask_list.append([0] * len(prompt_enc) + [1] * len(output_enc))
        max_seq_len = max(max_seq_len, len(prompt_enc) + len(output_enc))
    
    # Train: right padding to get better efficiency
    # Infer: left padding to align the *generation from the last token*
    tokens_padded = torch.stack([
        F.pad(torch.tensor(t), (0, max_seq_len - len(t)), value=tokenizer.pad_token_id)
        for t in token_list
    ])
    mask_padded = torch.stack([
        F.pad(torch.tensor(m), (0, max_seq_len - len(m)), value=0)
        for m in mask_list
    ])

    return {
        "input_ids": tokens_padded[:, :-1],
        "labels": tokens_padded[:, 1:],
        "response_mask": mask_padded[:, 1:]
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the probability distribution defined by logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor, # shape (batch_size, seq_len)
    return_token_entropy: bool = False,
    inference_mode: bool = False,
) -> dict[str, torch.Tensor]:
    context = torch.inference_mode() if inference_mode else nullcontext()
    with context:
        logits = model(input_ids).logits # shape (batch_size, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        result = {"log_probs": log_probs_labels} # for loss computation
        if return_token_entropy:
            entropy = compute_entropy(logits)
            result["token_entropy"] = entropy # for perplexity and exploration bonus
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float=1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=-1,
    ).mean() / gradient_accumulation_steps
    loss.backward()
    metadata = {}
    return loss, metadata
