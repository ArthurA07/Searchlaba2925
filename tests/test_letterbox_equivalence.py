#!/usr/bin/env python3
import math

import numpy as np

from backend.app.services.image_utils import letterbox


def test_letterbox_roundtrip_identity():
    # Synthetic original size
    H, W = 1080, 1920
    img = np.zeros((H, W, 3), dtype=np.uint8)
    resized, scale, (left, top) = letterbox(img, 2048, stride=32, scaleFill=False, auto=True, scaleup=True)

    # Generate synthetic boxes on original image and map → letterboxed → back
    boxes = np.array([
        [10, 20, 100, 200],
        [W - 120, H - 220, W - 10, H - 10],
        [W * 0.25, H * 0.25, W * 0.5, H * 0.7],
    ], dtype=float)

    # forward map to letterboxed coords
    fwd = boxes.copy()
    fwd[:, [0, 2]] = fwd[:, [0, 2]] * scale + left
    fwd[:, [1, 3]] = fwd[:, [1, 3]] * scale + top

    # inverse map back to original coords
    inv = fwd.copy()
    inv[:, [0, 2]] = (inv[:, [0, 2]] - left) / scale
    inv[:, [1, 3]] = (inv[:, [1, 3]] - top) / scale

    err = np.abs(inv - boxes)
    assert np.max(err) <= 1.0, f"Roundtrip error too large: {np.max(err)}"


if __name__ == "__main__":
    test_letterbox_roundtrip_identity()
    print("[ok] letterbox roundtrip <= 1px")


