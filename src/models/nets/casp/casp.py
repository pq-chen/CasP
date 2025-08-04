from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .coarse_matching import CoarseMatching
from .decoders import HybridInteraction
from .encoders import RepVGGAdapter, SelfCoC
from .fine_matching import FineMatching
from .fine_preprocess import FinePreprocess
from .homo.fine_homo import FineHomo
from .positional_encoding import SinusoidalPositionalEncoding
from .utils import crop_to_mask, pad_to_mask


class CasP(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.scales = config["scales"]
        self.data_mode = config["data_mode"]
        self.data_factor = config["data_factor"]

        self.low_level_encoder = RepVGGAdapter(**config["low_level_encoder"])
        self.high_level_encoder = SelfCoC(**config["high_level_encoder"])
        self.positional_encoding = SinusoidalPositionalEncoding(
            **config["positional_encoding"]
        )
        self.coarse_module = HybridInteraction(**config["coarse_module"])
        self.coarse_matching = CoarseMatching(**config["coarse_matching"])
        self.fine_preprocess = FinePreprocess(**config["fine_preprocess"])
        self.fine_cls_matching = FineMatching(**config["fine_cls_matching"])
        self.fine_reg_matching = FineHomo(**config["fine_reg_matching"])

    def extract_and_transform_features(
        self,
        image0: Tensor,
        image1: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
        enable_crop: bool = True,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if image0.shape == image1.shape:
            image = torch.cat([image0, image1])
            x_list = self.low_level_encoder(image)
            x0_low_list, x1_low_list = map(
                list, zip(*[x.chunk(2) for x in x_list])
            )
        else:
            x0_low_list = self.low_level_encoder(image0)
            x1_low_list = self.low_level_encoder(image1)

        x0_8x, x1_8x = x0_low_list[-1], x1_low_list[-1]
        _, c, h0, w0 = x0_8x.shape
        _, _, h1, w1 = x1_8x.shape
        encoding0 = self.positional_encoding(image0)
        encoding1 = self.positional_encoding(image1)
        x0_8x = x0_8x + encoding0.permute(2, 0, 1)[:c, :h0, :w0]
        x1_8x = x1_8x + encoding1.permute(2, 0, 1)[:c, :h1, :w1]

        if enable_crop and mask0 is not None and mask1 is not None:
            x0_8x_list = crop_to_mask(x0_8x, mask0)
            x1_8x_list = crop_to_mask(x1_8x, mask1)
            x0_16x, x1_16x = [], []
            for b in range(len(x0_8x)):
                b_x0_high_list = self.high_level_encoder(x0_8x_list[b])
                b_x1_high_list = self.high_level_encoder(x1_8x_list[b])
                b_x0_16x, b_x1_16x = self.coarse_module(
                    b_x0_high_list, b_x1_high_list, encoding0, encoding1
                )
                b_mask0 = F.max_pool2d(mask0[[b]].float(), 2).bool()
                b_mask1 = F.max_pool2d(mask1[[b]].float(), 2).bool()
                x0_16x.append(pad_to_mask(b_x0_16x, b_mask0))
                x1_16x.append(pad_to_mask(b_x1_16x, b_mask1))
            x0_16x, x1_16x = torch.cat(x0_16x), torch.cat(x1_16x)
        else:
            x0_high_list = self.high_level_encoder(x0_8x)
            x1_high_list = self.high_level_encoder(x1_8x)
            x0_16x, x1_16x = self.coarse_module(
                x0_high_list,
                x1_high_list,
                encoding0,
                encoding1,
                mask0=mask0,
                mask1=mask1,
            )
        x0_list, x1_list = [*x0_low_list, x0_16x], [*x1_low_list, x1_16x]
        return x0_list, x1_list

    @torch.no_grad()
    def update_points(
        self, data: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        scale_coarse = self.scales[0]
        scale_fine = self.scales[1] * (self.fine_reg_matching.window_size // 2)
        w0 = data["image0"].shape[-1] // scale_coarse
        w1 = data["image1"].shape[-1] // scale_coarse
        b_indices, i_indices, j_indices = results["coarse_cls_indices"]

        points0 = torch.stack([i_indices % w0, i_indices // w0], dim=-1).float()
        points1 = torch.stack([j_indices % w1, j_indices // w1], dim=-1).float()
        points0 = points0 * scale_coarse + results["fine_cls_biases0"]
        points1 = (
            points1 * scale_coarse
            + results["fine_cls_biases1"]
            + results["fine_reg_biases"] * scale_fine
        )
        if "scale0" in data and "scale1" in data:
            points0 = points0 * data["scale0"][b_indices]
            points1 = points1 * data["scale1"][b_indices]
        results["points0"], results["points1"] = points0, points1

    def forward(
        self, data: Dict[str, Any], enable_crop: bool = True
    ) -> Dict[str, Any]:
        image0, image1 = data["image0"], data["image1"]
        mask0, mask1 = data.get("mask0"), data.get("mask1")
        results = {}

        x0_list, x1_list = self.extract_and_transform_features(
            image0, image1, mask0=mask0, mask1=mask1, enable_crop=enable_crop
        )

        results.update(
            self.coarse_matching(
                x0_list[-2:], x1_list[-2:], mask0=mask0, mask1=mask1
            )
        )
        x0_8x, x1_8x = results.pop("x_8x")

        x0_list, x1_list = [*x0_list[:-2], x0_8x], [*x1_list[:-2], x1_8x]
        x0_cropped, x1_cropped = self.fine_preprocess(
            x0_list, x1_list, results["coarse_cls_indices"]
        )

        results.update(self.fine_cls_matching(x0_cropped, x1_cropped))
        init = torch.cat(
            [results["fine_cls_biases0"], results["fine_cls_biases1"]], dim=-1
        )
        init = init / self.scales[0] + 0.5
        results.update(
            self.fine_reg_matching(
                x0_cropped, x1_cropped, num_iters=1, init=init
            )
        )

        self.update_points(data, results)
        return results

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k in list(state_dict.keys()):
            if k.startswith("net."):
                new_k = k.replace("net.", "", 1)
                state_dict[new_k] = state_dict.pop(k)
        return super().load_state_dict(state_dict)
