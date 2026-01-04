import torch
from omegaconf import OmegaConf
from src.models.nets import CasP

# ------------------------------------------------------------
# 1. Load config and model
# ------------------------------------------------------------
ckpt_path = "weights/casp_minima.pth"    # or "weights/casp_outdoor.pth"
config_path = "configs/model/net/casp.yaml"

config = OmegaConf.load(config_path).config
model = CasP(config)
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

# ------------------------------------------------------------
# 2. Wrapper: convert tensor inputs → dict for CasP
# ------------------------------------------------------------
class CasPWrapper(torch.nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, image0, image1, scale0, scale1):
        data = {
            "image0": image0,
            "image1": image1,
            "scale0": scale0,
            "scale1": scale1,
        }
        results = self.matcher(data)
        return results["points0"], results["points1"], results["scores"]

wrapper = CasPWrapper(model)

# ------------------------------------------------------------
# 3. Dummy inputs (match config: gray=1ch, color=3ch)
# ------------------------------------------------------------
H, W = 512, 512
if config.data_mode == "gray":
    C = 1
elif config.data_mode == "color":
    C = 3
else:
    raise ValueError(f"Unsupported data_mode: {config.data_mode}")

image0 = torch.randn(1, C, H, W, dtype=torch.float32)   # (N,C,H,W)
image1 = torch.randn(1, C, H, W, dtype=torch.float32)
scale0 = torch.ones(1, 2, dtype=torch.float32)          # (N,2)
scale1 = torch.ones(1, 2, dtype=torch.float32)

# ------------------------------------------------------------
# 4. Export to ONNX
# ------------------------------------------------------------
torch.onnx.export(
    wrapper,
    (image0, image1, scale0, scale1),
    "casp.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=False,
    input_names=["image0", "image1", "scale0", "scale1"],
    output_names=["points0", "points1", "scores"],
    dynamic_axes=None,   # fixed shapes
)

print("✅ Successfully exported CasP model to casp.onnx")
