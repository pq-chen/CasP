from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class FinePreprocess(Module):
    def __init__(
        self,
        dim_list: List[int],
        window_size: int,
        stride: int,
        padding: int,
        right_extra: int = 0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.window_size0 = window_size
        self.window_size1 = window_size + 2 * right_extra
        self.padding0, self.padding1 = padding, padding + right_extra

        self.projs, self.fusions = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dim_list) - 1):
            dim0, dim1 = dim_list[i : i + 2]
            self.projs.append(nn.Conv2d(dim0, dim1, 1, bias=False))
            self.fusions.append(
                nn.Sequential(
                    nn.Conv2d(dim1, dim1, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(dim1, dim0, 3, padding=1, bias=False),
                )
            )
        self.projs.append(nn.Conv2d(dim1, dim1, 1, bias=False))

    def fuse(self, x_list: List[Tensor]) -> Tensor:
        x0 = self.projs[-1](x_list[-1])
        for i in reversed(range(len(x_list) - 1)):
            x0, x1 = self.projs[i](x_list[i]), x0
            x0 = x0 + F.interpolate(
                x1, scale_factor=2.0, mode="bilinear", align_corners=True
            )
            x0 = self.fusions[i](x0)
        return x0

    def crop_by_indices(
        self, x0: Tensor, x1: Tensor, indices: Tensor
    ) -> Tuple[Tensor, Tensor]:
        b_indices, i_indices, j_indices = indices
        x0_cropped = F.unfold(
            x0, self.window_size0, stride=self.stride, padding=self.padding0
        )[b_indices, :, i_indices]
        x1_cropped = F.unfold(
            x1, self.window_size1, stride=self.stride, padding=self.padding1
        )[b_indices, :, j_indices]
        x0_cropped = x0_cropped.unflatten(
            -1, (-1, self.window_size0, self.window_size0)
        )
        x1_cropped = x1_cropped.unflatten(
            -1, (-1, self.window_size1, self.window_size1)
        )
        return x0_cropped, x1_cropped

    def forward(
        self, x0_list: List[Tensor], x1_list: List[Tensor], indices: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x0, x1 = x0_list[0], x1_list[0]
        if len(indices[0]) == 0:
            x0_cropped = x0.new_empty(0, self.window_size0**2, x0.shape[1])
            x1_cropped = x1.new_empty(0, self.window_size1**2, x1.shape[1])
            return x0_cropped, x1_cropped

        if x0.shape == x1.shape:
            x_list = [torch.cat(x) for x in zip(x0_list, x1_list)]
            x0, x1 = self.fuse(x_list).chunk(2)
        else:
            x0, x1 = self.fuse(x0_list), self.fuse(x1_list)
        x0_cropped, x1_cropped = self.crop_by_indices(x0, x1, indices)
        return x0_cropped, x1_cropped
