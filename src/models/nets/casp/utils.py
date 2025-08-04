from typing import List, Optional, Tuple

import torch
from torch import Tensor


def crop_to_mask(x: Tensor, mask: Tensor) -> List[Tensor]:
    x_list = []
    for b in range(len(x)):
        b_h = mask[b].sum(dim=0).amax().int().item()
        b_w = mask[b].sum(dim=1).amax().int().item()
        x_list.append(x[[b], :, :b_h, :b_w])
    return x_list


def pad_to_mask(x: Tensor, mask: Tensor) -> Tensor:
    assert len(x) == 1
    _, c, b_h, b_w = x.shape
    _, h, w = mask.shape
    out = x.new_zeros(1, c, h, w)
    out[0, :, :b_h, :b_w] = x
    return out


def patchify(x: Tensor, patch_size: int) -> Tensor:
    n, c, h, w = x.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    x = (
        x.reshape(n, c, grid_h, patch_size, grid_w, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(n, grid_h * grid_w, patch_size * patch_size, c)
    )
    return x


def unpatchify(
    x: Tensor, grid_size: Tuple[int, int], patch_size: int
) -> Tensor:
    n, _, _, c = x.shape
    grid_h, grid_w = grid_size
    x = (
        x.reshape(n, grid_h, grid_w, patch_size, patch_size, c)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(n, c, grid_h * patch_size, grid_w * patch_size)
    )
    return x


def gather_attended(x: Tensor, indices: Tensor) -> Tensor:
    n_range = torch.arange(len(x), device=x.device)[:, None, None]
    x = x[n_range, indices].flatten(start_dim=-3, end_dim=-2)
    return x


def remove_border_one_side(
    x: Tensor,
    border_removal: int,
    size: Tuple[int, int],
    mask: Optional[Tensor] = None,
) -> Tensor:
    assert len(x.shape) == 2
    r = border_removal
    if r == 0:
        return x

    out = x.unflatten(1, size)
    out[:, :r, :] = 0
    out[:, :, :r] = 0
    if mask is not None:
        for b in range(len(x)):
            h = mask[b].sum(dim=0).amax().int().item()
            w = mask[b].sum(dim=1).amax().int().item()
            out[b, h - r :, :] = 0
            out[b, :, w - r :] = 0
    else:
        out[:, -r:, :] = 0
        out[:, :, -r:] = 0
    out = out.flatten(start_dim=1)
    return out
