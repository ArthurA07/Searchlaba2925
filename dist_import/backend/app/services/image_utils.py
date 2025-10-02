from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_table_roi_bounding_box(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = img_bgr.shape[:2]
    if w * h < 0.01 * W * H:
        return None
    return int(x), int(y), int(w), int(h)


def letterbox(
    img_bgr: np.ndarray,
    new_shape: int,
    stride: int = 32,
    scaleFill: bool = False,
    auto: bool = True,
    scaleup: bool = True,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    # Adapted to match Ultralytics letterbox semantics (ratio, padding multiple of stride)
    shape = img_bgr.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r

    if scaleFill:
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw, dh = 0.0, 0.0
    else:
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw %= stride
            dh %= stride
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img_bgr, float(r), (left, top)


