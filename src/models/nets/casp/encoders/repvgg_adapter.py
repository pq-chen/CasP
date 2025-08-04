from typing import List

import torch.nn as nn
from torch import Tensor

from .repvgg import RepVGG


class RepVGGAdapter(RepVGG):
    def __init__(self, dim_list: List[int], num_blocks_list: List[int]) -> None:
        assert len(num_blocks_list) == len(dim_list) == 3
        num_blocks_list = [*num_blocks_list, 1]
        width_multiplier = [
            dim / base for dim, base in zip(dim_list, [64, 128, 256])
        ]
        width_multiplier = [*width_multiplier, 1.0]
        super().__init__(num_blocks_list, width_multiplier=width_multiplier)
        del self.stage0, self.stage1, self.stage4, self.gap, self.linear

        in_planes, cur_layer_idx = self.in_planes, self.cur_layer_idx
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.conv = nn.Conv2d(
            1, self.in_planes, 7, stride=2, padding=3, bias=False
        )
        self.norm = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks_list[0], stride=1
        )
        self.in_planes, self.cur_layer_idx = in_planes, cur_layer_idx

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.relu(self.norm(self.conv(x)))
        for block in self.stage1:
            x = block(x)
        x_2x = x
        for block in self.stage2:
            x = block(x)
        x_4x = x
        for block in self.stage3:
            x = block(x)
        x_8x = x
        out = [x_2x, x_4x, x_8x]
        return out
