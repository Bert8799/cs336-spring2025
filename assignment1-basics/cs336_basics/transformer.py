import torch
import torch.nn as nn
from .layers import RMSNorm, Embedding, Linear
from .transformer_layers import SwiGLU, RotaryPositionEmbedding, MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head self-attention and feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionEmbedding | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
        )
        self.feed_forward = SwiGLU(
            d_model=d_model,
            d_ff=d_ff
        )
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), token_position)
        x = x + self.feed_forward(self.norm2(x))

        return x
    

class TransformerLM(nn.Module):
    """
    A Transformer-based Language Model.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        theta: float = 100000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.position_embedding = RotaryPositionEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=self.position_embedding,
                device=device
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = x.size()
        if token_position is None:
            token_position = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_position)

        x = self.norm(x)
        output = self.output_linear(x)
        return output