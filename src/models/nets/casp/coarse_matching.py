from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .decoders import RegionSelectiveTransformer
from .utils import remove_border_one_side, unpatchify


class CoarseMatching(Module):
    def __init__(
        self,
        dim_list: List[int],
        num_heads: int,
        num_layers: int,
        topk: int = 8,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(dim_list) == 2
        self.stride = 2
        self.topk = topk
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        grid = create_meshgrid(
            self.stride,
            self.stride,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("grid", grid, persistent=False)

        self.region_selective_module = RegionSelectiveTransformer(
            dim_list, num_heads, self.stride, num_layers
        )

    def map_indices(
        self, x0: Tensor, size0: Tuple[int, int], size1: Tuple[int, int]
    ) -> Tensor:
        row = (x0[..., None] // size1[1]) * self.stride + self.grid[:, 1]
        col = (x0[..., None] % size1[1]) * self.stride + self.grid[:, 0]
        x0 = (
            (row * size1[1] * self.stride + col)
            .reshape(len(x0), size0[0], 1, size0[1], 1, -1)
            .expand(-1, -1, 2, -1, 2, -1)
            .flatten(start_dim=1, end_dim=4)
        )
        return x0

    @torch.no_grad()
    def create_coarse_matching(
        self,
        heatmap: Tuple[Tensor, Tensor],
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
    ) -> Dict[str, Any]:
        heatmap0_to_1, heatmap1_to_0 = heatmap
        score0_to_1, sub_indices0_to_1 = heatmap0_to_1.max(dim=-1, keepdim=True)
        score0_to_1 = score0_to_1[:, :, 0]
        sub_indices1_to_0 = heatmap1_to_0.argmax(dim=-2, keepdim=True)
        indices0_to_1 = indices0_to_1.gather(-1, sub_indices0_to_1)[:, :, 0]
        indices1_to_0 = indices1_to_0.gather(-2, sub_indices1_to_0)[:, 0, :]
        indices0_to_1 = remove_border_one_side(
            indices0_to_1, self.border_removal, size0, mask=mask0
        )
        indices1_to_0 = remove_border_one_side(
            indices1_to_0, self.border_removal, size1, mask=mask1
        )
        biprojection = indices1_to_0.gather(-1, indices0_to_1)
        mask0_to_1 = biprojection == torch.arange(
            heatmap0_to_1.shape[1], device=heatmap0_to_1.device
        )
        if self.border_removal > 0:
            mask0_to_1[:, 0] = False
        mask0_to_1 = mask0_to_1 & (score0_to_1 > self.threshold)
        b_indices, i_indices = mask0_to_1.nonzero(as_tuple=True)
        j_indices = indices0_to_1[b_indices, i_indices]
        indices = torch.stack([b_indices, i_indices, j_indices])
        scores = score0_to_1[mask0_to_1]
        result = {"coarse_cls_indices": indices, "scores": scores}
        return result

    def forward(
        self,
        x0_list: List[Tensor],
        x1_list: List[Tensor],
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        assert len(x0_list) == len(x1_list) == 2
        n, c, h0, w0 = x0_list[0].shape
        _, _, h1, w1 = x1_list[0].shape
        grid_size0 = tuple(x0_list[1].shape[-2:])
        grid_size1 = tuple(x1_list[1].shape[-2:])
        results = {}

        x0_prior = x0_list[1].flatten(start_dim=-2)
        x1_prior = x1_list[1].flatten(start_dim=-2)
        similarity = x0_prior.transpose(-2, -1) @ x1_prior / c
        if mask0 is not None and mask1 is not None:
            mask0_prior = F.max_pool2d(mask0.float(), self.stride).bool()
            mask1_prior = F.max_pool2d(mask1.float(), self.stride).bool()
            mask = mask0_prior.reshape(n, -1, 1) & mask1_prior.reshape(n, 1, -1)
            similarity.masked_fill_(~mask, -float("inf"))

        _, indices0_to_1 = similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = similarity.transpose(-2, -1).topk(self.topk, dim=-1)
        x0, x1, attended0_to_1, attended1_to_0 = self.region_selective_module(
            x0_list, x1_list, indices0_to_1, indices1_to_0
        )
        indices0_to_1 = self.map_indices(indices0_to_1, grid_size0, grid_size1)
        indices1_to_0 = self.map_indices(
            indices1_to_0, grid_size1, grid_size0
        ).transpose(-2, -1)

        similarity0_to_1 = x0 @ attended0_to_1.transpose(-2, -1) / c
        similarity1_to_0 = x1 @ attended1_to_0.transpose(-2, -1) / c
        heatmap0_to_1 = (
            (similarity0_to_1 / self.temperature)
            .softmax(dim=-1)
            .reshape(n, *grid_size0, self.stride, self.stride, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(start_dim=1, end_dim=-2)
        )
        heatmap1_to_0 = (
            (similarity1_to_0 / self.temperature)
            .softmax(dim=-1)
            .reshape(n, *grid_size1, self.stride, self.stride, -1)
            .permute(0, 5, 1, 3, 2, 4)
            .flatten(start_dim=2, end_dim=-1)
        )
        heatmap0_to_1, heatmap1_to_0 = (
            heatmap0_to_1
            * (
                x1.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(-2, indices1_to_0, heatmap1_to_0)
                .gather(-1, indices0_to_1)
            ),
            heatmap1_to_0
            * (
                x0.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(-1, indices0_to_1, heatmap0_to_1)
                .gather(-2, indices1_to_0)
            ),
        )
        heatmap = (heatmap0_to_1, heatmap1_to_0)

        results.update(
            self.create_coarse_matching(
                heatmap,
                indices0_to_1,
                indices1_to_0,
                (h0, w0),
                (h1, w1),
                mask0=mask0,
                mask1=mask1,
            )
        )
        x0 = unpatchify(x0, grid_size0, self.stride)
        x1 = unpatchify(x1, grid_size1, self.stride)
        results["x_8x"] = (x0, x1)
        return results
