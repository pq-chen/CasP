from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class CrossContextCluster(Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj0 = nn.Linear(dim, dim, bias=bias)
        self.proj1 = nn.Linear(dim, dim * 2, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        n, h, w, _ = x0.shape
        m = n * self.num_heads

        point0 = (
            self.proj0(x0).permute(0, 3, 1, 2).reshape(m, self.head_dim, h * w)
        )
        point1, value1 = (
            self.proj1(x1)
            .permute(0, 3, 1, 2)
            .reshape(m, self.head_dim * 2, -1)
            .chunk(2, dim=1)
        )
        point0, point1 = F.normalize(point0, dim=1), F.normalize(point1, dim=1)
        similarity = point0.transpose(-2, -1) @ point1
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = (
                mask.reshape(n, 1, -1)
                .expand(-1, self.num_heads, -1)
                .reshape(m, 1, -1)
            )
            similarity.masked_fill_(~mask, float("-inf"))
        similarity = similarity.sigmoid()

        m_range = torch.arange(m, device=x0.device)[:, None]
        max_sim, indices = similarity.max(dim=-1, keepdim=True)
        message = max_sim * value1[m_range, :, indices[:, :, 0]]
        message = (
            message.reshape(n, self.num_heads, h, w, self.head_dim)
            .permute(0, 2, 3, 1, 4)
            .flatten(start_dim=-2)
        )
        message = self.out_proj(message)
        return message


class CrossCoCBlock(Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.context_cluster = CrossContextCluster(dim, num_heads, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * 2, dim * 2, bias=bias)
        self.gelu = nn.GELU()
        self.conv = nn.Conv2d(dim * 2, dim, 3, padding=1, bias=bias)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        message = self.norm1(self.context_cluster(x0, x1, mask=mask))
        message = torch.cat([x0, message], dim=-1)
        message = self.gelu(self.linear(message)).permute(0, 3, 1, 2)
        message = self.conv(message).permute(0, 2, 3, 1)
        x0 = x0 + self.norm2(message)
        return x0
