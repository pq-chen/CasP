from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray


def load_image(
    path: str,
    mode: str = "gray",
    size: Optional[Union[int, Tuple[int, int]]] = None,
    factor: int = 1,
    pad_to_square: bool = False,
) -> Tuple[ndarray, Optional[ndarray], ndarray]:
    if mode == "gray":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif mode == "color":
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Supported modes are `gray` and `color`.")

    h, w = image.shape[:2]
    if size is None:
        new_w, new_h = w, h
    elif isinstance(size, int):
        k = size / max(w, h)
        new_w, new_h = round(w * k), round(h * k)
    else:
        new_w, new_h = size
    new_w, new_h = new_w // factor * factor, new_h // factor * factor
    image = cv2.resize(image, (new_w, new_h))
    scale = np.array([w / new_w, h / new_h])

    mask = None
    if pad_to_square:
        length = max(new_w, new_h)
        pad_size = (length, length) if mode == "gray" else (length, length, 3)
        pad_image = np.zeros(pad_size, dtype=image.dtype)
        pad_image[:new_h, :new_w] = image
        image = pad_image
        mask = np.zeros((length, length), dtype=bool)
        mask[:new_h, :new_w] = True
    return image, mask, scale
