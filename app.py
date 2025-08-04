import os
from typing import Tuple

import gradio as gr
import numpy as np
import requests
from matplotlib.figure import Figure
from numpy import ndarray

from demo import main, matcher_configs
from src.models.utils import make_matching_figure

HF_TOKEN = os.getenv("HF_TOKEN")

CSS = """
#desc, #desc * {
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
}
"""

DESCRIPTION = """
<div align="center">
<h1><ins>CasP</ins> 🪜</h1>
<h2>
    Improving Semi-Dense Feature Matching Pipeline Leveraging<br>
    Cascaded Correspondence Priors for Guidance
</h2>
<h3>ICCV 2025</h3>
<p>
    <b>Peiqi Chen<sup>1*</sup> · Lei Yu<sup>2*</sup> · Yi Wan<sup>1&dagger;</sup> Yingying Pei<sup>1</sup> · Xinyi Liu<sup>1</sup> · Yongxiang Yao<sup>1</sup></b><br>
    <b>Yingying Zhang<sup>2</sup> · Lixiang Ru<sup>2</sup> · Liheng Zhong<sup>2</sup> · Jingdong Chen<sup>2</sup> · Ming Yang<sup>2</sup> · Yongjun Zhang<sup>1&dagger;</sup></b>
</p>
<p>
    <sup>1</sup>Wuhan University&emsp;&emsp;&emsp;<sup>2</sup>Ant Group<br>
    *Equal contribution&emsp;&emsp;&emsp;&dagger;Corresponding author
</p>
<div style="display: flex; justify-content: center; align-items: flex-start; flex-wrap: wrap;">
    <a href="https://arxiv.org/abs/2507.17312"><img src="https://img.shields.io/badge/arXiv-2507.17312-b31b1b.svg"></a>&emsp;
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</div>
</div>
"""

examples = [
    [
        "assets/example_pairs/pair1-1.png",
        "assets/example_pairs/pair1-2.png",
        "casp_outdoor",
        "fundamental",
    ],
    [
        "assets/example_pairs/pair2-1.png",
        "assets/example_pairs/pair2-2.png",
        "casp_outdoor",
        "fundamental",
    ],
    [
        "assets/example_pairs/pair3-1.png",
        "assets/example_pairs/pair3-2.png",
        "casp_outdoor",
        "fundamental",
    ],
    [
        "assets/example_pairs/pair4-1.jpg",
        "assets/example_pairs/pair4-2.jpg",
        "casp_minima",
        "homography",
    ],
    [
        "assets/example_pairs/pair5-1.jpg",
        "assets/example_pairs/pair5-2.jpg",
        "casp_minima",
        "homography",
    ],
    [
        "assets/example_pairs/pair6-1.jpg",
        "assets/example_pairs/pair6-2.jpg",
        "casp_minima",
        "homography",
    ],
]


def fig_to_ndarray(fig: Figure) -> ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buffer = fig.canvas.buffer_rgba()
    out = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 4)
    return out


def run_matching(
    method: str,
    path0: str,
    path1: str,
    image_size: int,
    matching_threshold: float,
    ransac: str,
    estimator: str,
    inlier_threshold: float,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    ransac = None if ransac == "none" else ransac
    matching_args = {
        "method": method,
        "path0": path0,
        "path1": path1,
        "image_size": image_size,
        "matching_threshold": matching_threshold,
        "ransac": ransac,
        "estimator": estimator,
        "inlier_threshold": inlier_threshold,
    }
    points0, points1, scores, inlier_mask = main(matching_args)

    errors = 1 - scores
    text = [f"{method} (raw)", f"#matches: {len(points0)}"]
    plotting_args = {
        "path0": path0,
        "path1": path1,
        "points0": points0,
        "points1": points1,
        "errors": errors,
        "threshold": 0.5,
        "text": text,
        "dpi": 300,
    }
    raw_keypoint_fig = fig_to_ndarray(
        make_matching_figure(**plotting_args, enable_line=False)
    )
    raw_matching_fig = fig_to_ndarray(make_matching_figure(**plotting_args))

    ransac_keypoint_fig = ransac_matching_fig = None
    if inlier_mask is not None:
        for key in ["points0", "points1", "errors"]:
            plotting_args[key] = plotting_args[key][inlier_mask]
        plotting_args["text"] = [
            f"{method} (RANSAC)",
            f"#matches: {inlier_mask.sum()}",
        ]
        ransac_keypoint_fig = fig_to_ndarray(
            make_matching_figure(**plotting_args, enable_line=False)
        )
        ransac_matching_fig = fig_to_ndarray(
            make_matching_figure(**plotting_args)
        )
    return (
        raw_keypoint_fig,
        raw_matching_fig,
        ransac_keypoint_fig,
        ransac_matching_fig,
    )


with gr.Blocks(css=CSS) as demo:
    with gr.Tab("Image Matching"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(DESCRIPTION, elem_id="desc")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Panels:")
                with gr.Row():
                    method = gr.Dropdown(
                        choices=["casp_outdoor", "casp_minima"],
                        value="casp_outdoor",
                        label="Matching Model",
                    )
                with gr.Row():
                    path0 = gr.Image(
                        height=300,
                        image_mode="RGB",
                        type="filepath",
                        label="Image 0",
                    )
                    path1 = gr.Image(
                        height=300,
                        image_mode="RGB",
                        type="filepath",
                        label="Image 1",
                    )
                with gr.Row():
                    stop = gr.Button(value="Stop", variant="stop")
                    run = gr.Button(value="Run", variant="primary")
                with gr.Accordion("Advanced Setting", open=False):
                    with gr.Accordion("Image Setting"):
                        with gr.Row():
                            force_resize = gr.Checkbox(
                                label="Force Resize", value=True
                            )
                            image_size = gr.Slider(
                                minimum=512,
                                maximum=1408,
                                value=1152,
                                step=32,
                                label="Longer Side (pixels)",
                            )
                    with gr.Accordion("Matching Setting"):
                        with gr.Row():
                            matching_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1,
                                value=0.2,
                                step=0.05,
                                label="Matching Threshold",
                            )
                    with gr.Accordion("RANSAC Setting"):
                        with gr.Row():
                            ransac = gr.Dropdown(
                                choices=["none", "fundamental", "homography"],
                                value="none",
                                label="Model",
                            )
                        with gr.Row():
                            estimator = gr.Dropdown(
                                choices=["CV2_RANSAC", "CV2_USAC_MAGSAC"],
                                value="CV2_USAC_MAGSAC",
                                label="Estimator",
                                visible=False,
                            )
                        with gr.Row():
                            inlier_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.5,
                                label="Inlier Threshold",
                                visible=False,
                            )
                with gr.Row():
                    with gr.Accordion("Example Pairs"):
                        gr.Examples(
                            examples=examples,
                            inputs=[path0, path1, method, ransac],
                            label="Click an example pair below",
                        )
            with gr.Column():
                gr.Markdown(
                    "### Output Panels: 🟢▲ High confidence | 🔴▼ Low confidence"
                )
                with gr.Accordion("Raw Keypoints", open=False):
                    raw_keypoint_fig = gr.Image(
                        format="png", type="numpy", label="Raw Keypoints"
                    )
                with gr.Accordion("Raw Matches"):
                    raw_matching_fig = gr.Image(
                        format="png", type="numpy", label="Raw Matches"
                    )
                with gr.Accordion("RANSAC Keypoints", open=False):
                    ransac_keypoint_fig = gr.Image(
                        format="png", type="numpy", label="RANSAC Keypoints"
                    )
                with gr.Accordion("RANSAC Matches"):
                    ransac_matching_fig = gr.Image(
                        format="png", type="numpy", label="RANSAC Matches"
                    )

            inputs = [
                method,
                path0,
                path1,
                image_size,
                matching_threshold,
                ransac,
                estimator,
                inlier_threshold,
            ]
            outputs = [
                raw_keypoint_fig,
                raw_matching_fig,
                ransac_keypoint_fig,
                ransac_matching_fig,
            ]

            running_event = run.click(
                fn=run_matching, inputs=inputs, outputs=outputs
            )
            stop.click(
                fn=None, inputs=None, outputs=None, cancels=[running_event]
            )
            force_resize.select(
                fn=lambda checked: gr.update(
                    visible=checked, value=1152 if checked else None
                ),
                inputs=force_resize,
                outputs=image_size,
            )
            ransac.change(
                fn=lambda model: (
                    gr.update(visible=model != "none"),
                    gr.update(visible=model != "none"),
                ),
                inputs=ransac,
                outputs=[estimator, inlier_threshold],
            )


if __name__ == "__main__":
    if HF_TOKEN:
        for method, config in matcher_configs.items():
            url = (
                f"https://huggingface.co/pq-chen/CasP/resolve/main/{method}.pth"
            )
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.get(url, headers=headers)
            with open(config["ckpt_path"], "wb") as f:
                f.write(response.content)
    demo.launch()
