from copy import deepcopy
from typing import List, Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..utils import gather_attended, patchify, unpatchify

try:
    # This will be deprecated after torch 2.3.0, see https://github.com/pytorch/pytorch/releases/tag/v2.3.0
    from torch.backends.cuda import sdp_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False


def apply_rotary_emb(x: Tensor, encoding: Tensor) -> Tensor:
    sin, cos = encoding.unflatten(-1, (-1, 2)).chunk(2, dim=-1)
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack([-x2, x1], dim=-1)
    x = (x * cos + rotated_x * sin).flatten(start_dim=-2)
    return x


class Attention(Module):
    def __init__(
        self, enable_sdpa: bool = False, enable_flash: bool = False
    ) -> None:
        super().__init__()
        if enable_sdpa and not SDPA_AVAILABLE:
            warn(
                "`scaled_dot_product_attention` (SDPA) is not available. "
                "Consider installing PyTorch >= 2.0.",
                stacklevel=2,
            )
        self.enable_sdpa = enable_sdpa and SDPA_AVAILABLE
        if enable_flash and not self.enable_sdpa:
            warn("Flash attention requires SDPA to be enabled.", stacklevel=2)
        self.enable_flash = enable_flash and self.enable_sdpa

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if self.enable_sdpa:
            if self.enable_flash:
                assert mask is None
                q, k, v = [t.half().contiguous() for t in [q, k, v]]
                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    message = sdpa(q, k, v).to(q.dtype)
            else:
                q, k, v = [t.contiguous() for t in [q, k, v]]
                message = sdpa(q, k, v, attn_mask=mask)
        else:
            q = q * q.shape[-1] ** -0.5
            similarity = q @ k.transpose(-2, -1)
            if mask is not None:
                similarity.masked_fill_(~mask, -float("inf"))
            attention = similarity.softmax(dim=-1)
            message = attention @ v
        if mask is not None:
            message.nan_to_num_()
        return message


class TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        if stride > 1:
            self.patch_embed0 = nn.Conv2d(
                dim, dim, stride, stride=stride, groups=dim, bias=False
            )
            self.patch_embed1 = nn.MaxPool2d(stride, stride=stride)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        encoding: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n, h, w, c = x0.shape
        mask = mask[:, None] if mask is not None else None

        x0_, x1_ = x0, x1
        if self.stride > 1:
            x0_, x1_ = x0_.permute(0, 3, 1, 2), x1_.permute(0, 3, 1, 2)
            x0_ = self.patch_embed0(x0_).permute(0, 2, 3, 1)
            x1_ = self.patch_embed1(x1_).permute(0, 2, 3, 1)
            h, w = h // self.stride, w // self.stride

        q, k, v = self.q_proj(x0_), self.k_proj(x1_), self.v_proj(x1_)
        if encoding is not None:
            q = apply_rotary_emb(q, encoding[:h, :w, :c])
            k = apply_rotary_emb(k, encoding[:h, :w, :c])
        q, k, v = [
            t.reshape(n, -1, self.num_heads, self.head_dim).transpose(-3, -2)
            for t in [q, k, v]
        ]
        message = (
            self.attention(q, k, v, mask=mask)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        message = self.norm1(self.out_proj(message)).unflatten(-2, (h, w))
        if self.stride > 1:
            message = message.permute(0, 3, 1, 2).contiguous()
            message = F.interpolate(
                message,
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        message = torch.cat([x0, message], dim=-1)
        x0 = x0 + self.norm2(self.mlp(message))
        return x0


class RegionSelectiveTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x0: Tensor, x1: Tensor, grid_size: Tuple[int, int]
    ) -> Tensor:
        q, k, v = self.q_proj(x0), self.k_proj(x1), self.v_proj(x1)
        q, k, v = [
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
            for t in [q, k, v]
        ]
        message = (
            self.attention(q, k, v).transpose(-3, -2).flatten(start_dim=-2)
        )
        message = self.norm1(self.out_proj(message))
        message = torch.cat([x0, message], dim=-1)
        message = self.mlp(unpatchify(message, grid_size, self.stride))
        x0 = x0 + self.norm2(patchify(message, self.stride))
        return x0


class RegionSelectiveTransformer(Module):
    def __init__(
        self, dim_list: List[int], num_heads: int, stride: int, num_layers: int
    ) -> None:
        super().__init__()
        assert len(dim_list) == 2
        self.stride = stride

        dim0, dim1 = dim_list
        self.proj0 = nn.Conv2d(dim0, dim1, 1, bias=False)
        self.proj1 = nn.Conv2d(dim1, dim1, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim1, dim1, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim1, dim0, 3, padding=1, bias=False),
        )
        layer = RegionSelectiveTransformerBlock(dim0, num_heads, stride)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

    def fuse(self, x_list: List[Tensor]) -> Tensor:
        x0, x1 = self.proj0(x_list[0]), self.proj1(x_list[1])
        x0 = x0 + F.interpolate(
            x1, scale_factor=self.stride, mode="bilinear", align_corners=False
        )
        x0 = self.fusion(x0)
        return x0

    def forward(
        self,
        x0_list: List[Tensor],
        x1_list: List[Tensor],
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert len(x0_list) == len(x1_list) == 2
        grid_size0 = tuple(x0_list[1].shape[-2:])
        grid_size1 = tuple(x1_list[1].shape[-2:])

        if grid_size0 == grid_size1:
            x_list = [torch.cat(t) for t in zip(x0_list, x1_list)]
            x0, x1 = self.fuse(x_list).chunk(2)
        else:
            x0, x1 = self.fuse(x0_list), self.fuse(x1_list)

        x0, x1 = patchify(x0, self.stride), patchify(x1, self.stride)
        for layer in self.layers:
            attended0_to_1 = gather_attended(x1, indices0_to_1)
            x0 = layer(x0, attended0_to_1, grid_size0)
            attended1_to_0 = gather_attended(x0, indices1_to_0)
            x1 = layer(x1, attended1_to_0, grid_size1)
        attended0_to_1 = gather_attended(x1, indices0_to_1)
        return x0, x1, attended0_to_1, attended1_to_0
