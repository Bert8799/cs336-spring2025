import torch
import torch.nn as nn
from .layers import softmax, Linear
from einops import einsum, rearrange


class SwiGLU(nn.Module):
    """
    SwiGLU class that inherits from torch.nn.Module.
    Applies the SwiGLU activation function: SwiGLU(x) = W2 @ (SiLU(W1 @ x) * W3 @ x),
    where x is split into two halves along the last dimension.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # round up to the nearest multiple of 64
            d_ff = (d_ff + 63) // 64 * 64
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        return self.w2(self.silu(x1) * x2)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding class that inherits from torch.nn.Module.
    Applies RoPE to the input tensor.
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for Rotary Position Embedding.")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device) / d_k)) # (d_k//2)
        position = torch.arange(0, max_seq_len, device=device) # (max_seq_len)
        sinusoidal_inp = einsum("i,j->ij", position, freqs) # (max_seq_len, d_k//2)

        self.register_buffer("sin_cache", torch.sin(sinusoidal_inp), persistent=False)
        self.register_buffer("cos_cache", torch.cos(sinusoidal_inp), persistent=False)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        sin = self.sin_cache[token_position]
        cos = self.cos_cache[token_position]

        rotate_matrix = torch.stack(
            (torch.stack((cos, -sin), dim=-1),
             torch.stack((sin,  cos), dim=-1)),
            dim=-2
        )

        x_pair = rearrange(x, "... (d_pair 2) -> ... d_pair 2", d_pair=self.d_k // 2)
        x_rotated = einsum("... i j, ... j -> ... i", rotate_matrix, x_pair)
        out = rearrange(x_rotated, "... d_pair 2 -> ... (d_pair 2)", d_pair=self.d_k // 2)

        return out.to(dtype)
    

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention mechanism.
    """
    d_k = query.size(-1)
    scores = einsum("...seq_q d_k, ...seq_k d_k -> ... seq_q seq_k", query, key) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)
    output = einsum("... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v", attn_weights, value)

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multihead Self-Attention class that inherits from torch.nn.Module.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.query_linear = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.key_linear = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.value_linear = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        query = rearrange(query, "b seq (h d_k) -> b h seq d_k", h=self.num_heads)
        key = rearrange(key, "b seq (h d_k) -> b h seq d_k", h=self.num_heads)
        value = rearrange(value, "b seq (h d_k) -> b h seq d_k", h=self.num_heads)

        if token_position is None:
            token_position = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        if self.rope is not None:
            token_position = token_position.unsqueeze(1).expand(-1, self.num_heads, -1)
            query = self.rope(query, token_position)
            key = self.rope(key, token_position)

        casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        attn_output = scaled_dot_product_attention(query, key, value, casual_mask)

        attn_output = rearrange(attn_output, "b h seq d_k -> b seq (h d_k)")

        output = self.out_linear(attn_output)

        return output