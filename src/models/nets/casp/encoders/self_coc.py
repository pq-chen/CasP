from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class SelfContextCluster(Module):
    def __init__(
        self, dim: int, num_heads: int, num_anchors: int, bias: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj = nn.Linear(dim, dim * 2, bias=bias)
        self.pool = nn.AdaptiveMaxPool2d(num_anchors)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        n, h, w, _ = x.shape
        m = n * self.num_heads

        x = self.proj(x).permute(0, 3, 1, 2).reshape(m, self.head_dim * 2, h, w)
        point0, value0 = x.flatten(start_dim=-2).chunk(2, dim=1)
        point1, value1 = self.pool(x).flatten(start_dim=-2).chunk(2, dim=1)
        point0, point1 = F.normalize(point0, dim=1), F.normalize(point1, dim=1)
        similarity = point0.transpose(-2, -1) @ point1
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = (
                mask.reshape(n, 1, -1)
                .expand(-1, self.num_heads, -1)
                .reshape(m, -1, 1)
            )
            similarity.masked_fill_(~mask, float("-inf"))
        similarity = similarity.sigmoid()

        indices = similarity.argmax(dim=-1, keepdim=True)
        mask = torch.zeros_like(similarity).scatter_(-1, indices, 1.0)
        similarity = (mask * similarity)[:, None, :, :]
        value0, value1 = value0[:, :, :, None], value1[:, :, None, :]
        message = (value0 * similarity).sum(dim=2, keepdim=True) + value1
        message = message / (similarity.sum(dim=2, keepdim=True) + 1)
        message = (similarity * message).sum(dim=3)
        message = message.reshape(n, -1, h, w).permute(0, 2, 3, 1)
        message = self.out_proj(message)
        return message


class SelfCoCBlock(Module):
    def __init__(
        self, dim: int, num_heads: int, num_anchors: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.context_cluster = SelfContextCluster(
            dim, num_heads, num_anchors, bias=bias
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        message = self.norm1(self.context_cluster(x, mask=mask))
        message = torch.cat([x, message], dim=-1)
        x = x + self.norm2(self.mlp(message))
        return x


class SelfCoC(Module):
    def __init__(
        self,
        dim_list: List[int],
        num_blocks_list: List[int],
        num_heads_list: List[int],
        num_anchors_list: List[int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_embeds, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dim_list) - 1):
            dim0, dim1 = dim_list[i : i + 2]
            self.patch_embeds.append(
                nn.Conv2d(dim0, dim1, 3, stride=2, padding=1)
            )
            block = SelfCoCBlock(
                dim1, num_heads_list[i], num_anchors_list[i], bias=bias
            )
            self.layers.append(
                nn.ModuleList(
                    [deepcopy(block) for _ in range(num_blocks_list[i])]
                )
            )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> List[Tensor]:
        out = []
        for i in range(len(self.layers)):
            if mask is not None:
                mask = F.max_pool2d(mask.float(), 2).bool()

            if i != 0:
                x = x.permute(0, 3, 1, 2)
            x = self.patch_embeds[i](x).permute(0, 2, 3, 1)
            for block in self.layers[i]:
                x = block(x, mask)
            out.append(x)
        return out
