import argparse
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Type

import cv2
import torch
from numpy import ndarray
from omegaconf import OmegaConf
from torch.nn import Module

from src.data.utils import load_image
from src.models.nets import CasP
from src.models.utils import make_matching_figure

matcher_configs = {
    "casp_outdoor": {
        "matcher": CasP,
        "name": "casp",
        "ckpt_path": "weights/casp_outdoor.pth",
    },
    "casp_minima": {
        "matcher": CasP,
        "name": "casp",
        "ckpt_path": "weights/casp_minima.pth",
    },
}


def load_matcher(
    matcher: Type[Module],
    name: str,
    ckpt_path: str,
    threshold: Optional[float] = None,
    device: str = "cpu",
) -> Module:
    config = OmegaConf.load(f"configs/model/net/{name}.yaml").config
    if threshold is not None:
        config.threshold = threshold
    matcher = matcher(config)
    matcher.load_state_dict(torch.load(ckpt_path))
    matcher = matcher.eval().to(device)
    return matcher


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="CasP")

    parser.add_argument("--path0", type=str, required=True)
    parser.add_argument("--path1", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=1152)

    parser.add_argument(
        "--method",
        type=str,
        default="casp_outdoor",
        choices=["casp_outdoor", "casp_minima"],
    )
    parser.add_argument("--matching_threshold", type=float)

    parser.add_argument(
        "--ransac", type=str, choices=["fundamental", "homography"]
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="CV2_USAC_MAGSAC",
        choices=["CV2_RANSAC", "CV2_USAC_MAGSAC"],
    )
    parser.add_argument("--inlier_threshold", type=float, default=3.0)

    args = parser.parse_args()
    return args


def ransac_optimize(
    points0: ndarray,
    points1: ndarray,
    model: str,
    estimator: str,
    threshold: float,
) -> Tuple[ndarray, ndarray]:
    if model == "fundamental":
        func = cv2.findFundamentalMat
    elif model == "homography":
        func = cv2.findHomography
    else:
        raise NotImplementedError()

    if estimator == "CV2_RANSAC":
        method = cv2.RANSAC
    elif estimator == "CV2_USAC_MAGSAC":
        method = cv2.USAC_MAGSAC
    else:
        raise NotImplementedError()

    mat, inlier_mask = func(
        points0,
        points1,
        method=method,
        ransacReprojThreshold=threshold,
        confidence=0.99999,
        maxIters=10000,
    )
    return mat, inlier_mask


def main(
    args: Dict[str, Any],
) -> Tuple[ndarray, ndarray, ndarray, Optional[ndarray]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = load_matcher(
        **matcher_configs[args["method"]],
        threshold=args["matching_threshold"],
        device=device,
    )
    data_configs = {
        "mode": matcher.data_mode,
        "size": args["image_size"],
        "factor": matcher.data_factor,
    }
    image0, mask0, scale0 = load_image(args["path0"], **data_configs)
    image1, mask1, scale1 = load_image(args["path1"], **data_configs)
    if matcher.data_mode == "gray":
        image0, image1 = image0[None] / 255.0, image1[None] / 255.0
    elif matcher.data_mode == "color":
        image0 = image0.transpose(2, 0, 1) / 255.0
        image1 = image1.transpose(2, 0, 1) / 255.0
    else:
        raise ValueError()
    data = {
        "image0": image0[None],
        "image1": image1[None],
        "scale0": scale0[None],
        "scale1": scale1[None],
    }
    if mask0 is not None:
        data["mask0"] = mask0[None]
    if mask1 is not None:
        data["mask1"] = mask0[None]
    for key, value in data.items():
        data[key] = torch.from_numpy(value).float().to(device)
    with torch.no_grad():
        results = matcher(data)

    points0 = results["points0"].cpu().numpy()
    points1 = results["points1"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    inlier_mask = None
    if args["ransac"] is not None:
        _, inlier_mask = ransac_optimize(
            points0,
            points1,
            args["ransac"],
            args["estimator"],
            args["inlier_threshold"],
        )
        inlier_mask = inlier_mask.ravel() == 1
    return points0, points1, scores, inlier_mask


if __name__ == "__main__":
    args = dict(vars(parse_args()))
    points0, points1, scores, inlier_mask = main(args)
    if inlier_mask is not None:
        points0, points1, scores = [
            t[inlier_mask] for t in [points0, points1, scores]
        ]

    errors = 1 - scores
    text = [args["method"], f"#matches: {len(points0)}"]
    make_matching_figure(
        args["path0"],
        args["path1"],
        points0,
        points1,
        errors,
        0.5,
        dpi=300,
        save_path=args["save_path"],
    )
