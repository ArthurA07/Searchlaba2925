#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def largest_object_polygon(bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # remove border (paper frame) via morphological open+close
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mor = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    mor = cv2.morphologyEx(mor, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    # approximate polygon
    eps = 0.01 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return poly


def save_yolo_seg(path: Path, polys: List[Tuple[int, np.ndarray]], w: int, h: int) -> None:
    lines: List[str] = []
    for cls, pts in polys:
        xs = (pts[:, 0] / float(w)).clip(0, 1)
        ys = (pts[:, 1] / float(h)).clip(0, 1)
        coords: List[str] = []
        for x, y in zip(xs, ys):
            coords += [f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join([str(cls)] + coords))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-label single-object images with largest contour polygon")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--class-id", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prev = Path("out/preview")
    prev.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(args.images.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png"}:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        poly = largest_object_polygon(bgr)
        if poly is None:
            continue
        save_yolo_seg(args.out / (img_path.stem + ".txt"), [(args.class_id, poly.astype(np.float32))], W, H)
        # preview
        vis = bgr.copy()
        cv2.polylines(vis, [poly.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imwrite(str(prev / f"{img_path.stem}.png"), vis)

    print(f"[ok] autolabeled to: {args.out}; previews in {prev}")


if __name__ == "__main__":
    main()


