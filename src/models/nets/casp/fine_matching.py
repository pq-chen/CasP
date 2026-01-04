from typing import Any, Dict

import torch
import torch.nn.functional as F
from kornia.utils import create_meshgrid
from torch.nn import Module


class FineMatching(Module):
    """
    ONNX-friendly FineMatching:
      - Avoids fake batching via cat/chunk around GridSample.
      - Builds per-branch grids using runtime broadcasting tied to x.shape[0]
        (no repeat/tile that can be constant-folded).
    """
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        grid = create_meshgrid(window_size, window_size, normalized_coordinates=False)  # (1, ws, ws, 2)
        grid = grid - window_size / 2 + 0.5
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        # Expected shapes: x0, x1: (M, C, H, W) — M = number of windows/patches
        if x0.numel() == 0 or x0.shape[0] == 0:
            ws2 = self.window_size * self.window_size
            return {
                "fine_cls_heatmap": x0.new_empty(0, ws2, ws2),
                "fine_cls_indices": x0.new_empty(3, 0, dtype=torch.long),
                "fine_cls_biases0": x0.new_empty(0, 2),
                "fine_cls_biases1": x0.new_empty(0, 2),
            }

        if x0.shape != x1.shape:
            raise ValueError(f"FineMatching: x0 and x1 must have same shape; got {x0.shape} vs {x1.shape}")

        M, C, H, W = x0.shape
        ws = self.window_size
        K = ws * ws

        # Base grid normalized to [-1, 1] for grid_sample: (1, ws, ws, 2)
        base_grid = (self.grid.to(dtype=x0.dtype, device=x0.device) * 2.0 / ws)

        # --- Broadcast trick (ONNX-safe, ties batch to runtime M) ---
        # Create zeros of shape (M,1,1,1) from inputs; add to base_grid so it broadcasts to (M, ws, ws, 2)
        zero0 = x0.new_zeros((M, 1, 1, 1))
        zero1 = x1.new_zeros((M, 1, 1, 1))
        grid0 = base_grid + zero0 * 0.0  # (M, ws, ws, 2) by broadcasting at runtime
        grid1 = base_grid + zero1 * 0.0  # (M, ws, ws, 2)

        # Sample features separately to keep N=M consistent on both inputs
        y0 = F.grid_sample(x0, grid0, mode="bilinear", align_corners=True)  # (M, C, ws, ws)
        y1 = F.grid_sample(x1, grid1, mode="bilinear", align_corners=True)  # (M, C, ws, ws)

        # Flatten spatial dims -> (M, C, K)
        y0 = y0.reshape(M, C, K)
        y1 = y1.reshape(M, C, K)

        # Similarity heatmap per window: (M, K, K)
        similarity = torch.matmul(y0.transpose(-2, -1), y1) / float(C)
        heatmap = similarity.softmax(dim=-2) * similarity.softmax(dim=-1)

        # Indices and biases (no grad)
        with torch.no_grad():
            m_indices = torch.arange(M, device=x0.device)
            ij_indices = heatmap.reshape(M, K * K).argmax(dim=-1)  # (M,)
            i_indices = ij_indices // K
            j_indices = ij_indices % K
            indices = torch.stack([m_indices, i_indices, j_indices], dim=0)  # (3, M)

            # Use the *unnormalized* integer grid for biases
            flat_grid = self.grid.to(device=x0.device).reshape(K, 2)  # (K,2)
            biases0 = flat_grid[i_indices]  # (M,2)
            biases1 = flat_grid[j_indices]  # (M,2)

        return {
            "fine_cls_heatmap": heatmap,   # (M, K, K)
            "fine_cls_indices": indices,   # (3, M)
            "fine_cls_biases0": biases0,   # (M, 2)
            "fine_cls_biases1": biases1,   # (M, 2)
        }
