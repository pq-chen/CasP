import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .cross_coc import CrossCoCBlock
from .transformer import TransformerBlock


class MergeBlock(Module):
    def __init__(self, dim: int, stride: int, bias: bool = True) -> None:
        super().__init__()
        self.stride = stride

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        message = x1.permute(0, 3, 1, 2).contiguous()
        message = F.interpolate(
            message,
            scale_factor=self.stride,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        message = torch.cat([x0, message], dim=-1)
        message = self.norm(self.mlp(message))
        x0 = x0 + message
        message = message.permute(0, 3, 1, 2)
        x1 = x1 + F.max_pool2d(message, self.stride).permute(0, 2, 3, 1)
        return x0, x1


class HybridInteraction(Module):
    def __init__(
        self, dim: int, num_heads: int, num_layers: int, bias: bool = True
    ) -> None:
        super().__init__()
        merge_block = MergeBlock(dim, 2, bias=bias)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in range(num_layers)]
        )
        cross_coc_block = CrossCoCBlock(dim, num_heads, bias=bias)
        self.cross_coc_blocks = nn.ModuleList(
            [copy.deepcopy(cross_coc_block) for _ in range(num_layers)]
        )
        attention_block = TransformerBlock(dim, num_heads, stride=2)
        self.self_attention_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(num_layers)]
        )
        self.cross_attention_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(num_layers)]
        )

    def forward(
        self,
        x0_list: List[Tensor],
        x1_list: List[Tensor],
        encoding0: Tensor,
        encoding1: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert len(x0_list) == len(x1_list) == 2
        x0_16x, x0_32x = x0_list
        x1_16x, x1_32x = x1_list

        mask00 = mask11 = mask01 = mask10 = None
        if mask0 is not None and mask1 is not None:
            n = len(mask0)
            mask0 = F.max_pool2d(mask0.float(), 4).bool()
            mask1 = F.max_pool2d(mask1.float(), 4).bool()
            mask00 = mask0.reshape(n, -1, 1) & mask0.reshape(n, 1, -1)
            mask11 = mask1.reshape(n, -1, 1) & mask1.reshape(n, 1, -1)
            mask01 = mask0.reshape(n, -1, 1) & mask1.reshape(n, 1, -1)
            mask10 = mask01.transpose(-2, -1)

        for i in range(len(self.merge_blocks)):
            x0_16x, x0_32x = self.merge_blocks[i](x0_16x, x0_32x)
            x1_16x, x1_32x = self.merge_blocks[i](x1_16x, x1_32x)
            x0_16x = self.cross_coc_blocks[i](x0_16x, x1_32x, mask=mask1)
            x1_16x = self.cross_coc_blocks[i](x1_16x, x0_32x, mask=mask0)
            x0_16x = self.self_attention_blocks[i](
                x0_16x, x0_16x, encoding=encoding0, mask=mask00
            )
            x1_16x = self.self_attention_blocks[i](
                x1_16x, x1_16x, encoding=encoding1, mask=mask11
            )
            x0_16x = self.cross_attention_blocks[i](x0_16x, x1_16x, mask=mask01)
            x1_16x = self.cross_attention_blocks[i](x1_16x, x0_16x, mask=mask10)
        x0_16x, x1_16x = x0_16x.permute(0, 3, 1, 2), x1_16x.permute(0, 3, 1, 2)
        return x0_16x, x1_16x
