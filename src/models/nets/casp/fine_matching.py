from typing import Any, Dict

import torch
import torch.nn.functional as F
from kornia.utils import create_meshgrid
from torch.nn import Module


class FineMatching(Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        grid = create_meshgrid(
            window_size, window_size, normalized_coordinates=False
        )
        grid = grid - window_size / 2 + 0.5
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        if len(x0) == 0:
            results = {
                "fine_cls_heatmap": x0.new_empty(
                    0, self.window_size**2, self.window_size**2
                ),
                "fine_cls_indices": x0.new_empty(3, 0, dtype=torch.long),
                "fine_cls_biases0": x0.new_empty(0, 2),
                "fine_cls_biases1": x0.new_empty(0, 2),
            }
            return results

        x = torch.cat([x0, x1])
        grid = (self.grid * 2 / self.window_size).expand(len(x), -1, -1, -1)
        x0, x1 = F.grid_sample(
            x, grid, mode="bilinear", align_corners=True
        ).chunk(2)

        x0, x1 = x0.flatten(start_dim=-2), x1.flatten(start_dim=-2)
        similarity = x0.transpose(-2, -1) @ x1 / x0.shape[1]
        heatmap = similarity.softmax(dim=-2) * similarity.softmax(dim=-1)

        with torch.no_grad():
            m_indices = torch.arange(len(x0), device=x0.device)
            ij_indices = heatmap.flatten(start_dim=-2).argmax(dim=-1)
            i_indices = ij_indices // self.window_size**2
            j_indices = ij_indices % self.window_size**2
            indices = torch.stack([m_indices, i_indices, j_indices])
            biases0 = self.grid.flatten(end_dim=-2)[i_indices]
            biases1 = self.grid.flatten(end_dim=-2)[j_indices]

        results = {
            "fine_cls_heatmap": heatmap,
            "fine_cls_indices": indices,
            "fine_cls_biases0": biases0,
            "fine_cls_biases1": biases1,
        }
        return results
