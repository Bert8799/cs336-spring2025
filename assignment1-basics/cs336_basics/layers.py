import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """
    Linear class that inherits from torch.nn.Module 
    and performs a linear transformation: y = Wx.
    Linear weights: N(mu=0, sigma^2=2/(in+out)), truncated to [-3*sigma, 3*sigma].
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Xavier initialization
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum("... i, j i -> ... j", x, self.weights)
    

class Embedding(nn.Module):
    """
    Embedding class that inherits from torch.nn.Module 
    and performs embedding lookup.
    Embedding weights: N(mu=0, sigma^2=1), truncated to [-3*sigma, 3*sigma].
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_weights = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Normal initialization
        std = 1.0
        nn.init.trunc_normal_(self.embedding_weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_weights[x]
    

class RMSNorm(nn.Module):
    """
    RMSNorm class that inherits from torch.nn.Module 
    and performs Root Mean Square Layer Normalization.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.scale = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """use torch.float32 to compute the rms for better numerical stability"""
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = (x / rms) * self.scale
        return x_norm.to(dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Stable softmax implementation.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum