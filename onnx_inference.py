import numpy as np
import onnxruntime as ort
import cv2
from src.data.utils import load_image
from src.models.utils import make_matching_figure

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
onnx_model = "casp.onnx"
path0 = "./assets/example_pairs/pair1-1.png"
path1 = "./assets/example_pairs/pair1-2.png"
save_path = "./matches_fixed.png"

H, W = 512, 512
ransac_model = "fundamental"
ransac_estimator = "CV2_USAC_MAGSAC"
inlier_threshold = 3.0



# ------------------------------------------------------------
# RANSAC
# ------------------------------------------------------------
def ransac_optimize(points0, points1, model, estimator, threshold):
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
        points0, points1,
        method=method,
        ransacReprojThreshold=threshold,
        confidence=0.99999,
        maxIters=10000,
    )
    return mat, inlier_mask

# ------------------------------------------------------------
# Load images and preprocess
# ------------------------------------------------------------
image0, _, scale0 = load_image(path0, mode="gray", size=(H,W), factor=1)
image1, _, scale1 = load_image(path1, mode="gray", size=(H,W), factor=1)

image0 = image0[None, None] / 255.0
image1 = image1[None, None] / 255.0
scale0 = scale0[None]
scale1 = scale1[None]

image0 = image0.astype(np.float32)
image1 = image1.astype(np.float32)
scale0 = scale0.astype(np.float32)
scale1 = scale1.astype(np.float32)

# ------------------------------------------------------------
# Run ONNX model
# ------------------------------------------------------------
sess = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
inputs = {
    "image0": image0,
    "image1": image1,
    "scale0": scale0,
    "scale1": scale1,
}
points0, points1, scores = sess.run(["points0","points1","scores"], inputs)
points0 = np.asarray(points0)
points1 = np.asarray(points1)
scores  = np.asarray(scores)

# Build mask where score > 0.8
mask = scores > 0.8

# Apply mask consistently
points0 = points0[mask]
points1 = points1[mask]
scores  = scores[mask]

print(f"Kept {len(scores)} matches with score > 0.8")

# ------------------------------------------------------------
# RANSAC filtering
# ------------------------------------------------------------
inlier_mask = None
if ransac_model is not None:
    _, inlier_mask = ransac_optimize(
        points0, points1,
        ransac_model, ransac_estimator, inlier_threshold
    )
    if inlier_mask is not None:
        inlier_mask = inlier_mask.ravel() == 1
        points0, points1, scores = [t[inlier_mask] for t in (points0, points1, scores)]

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
errors = 1 - scores
make_matching_figure(
    path0,
    path1,
    points0,
    points1,
    errors,
    0.5,
    dpi=300,
    save_path=save_path,
)
print(f"✅ Inference complete. Saved visualization to {save_path}")
