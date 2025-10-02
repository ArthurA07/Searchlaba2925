#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def detect_table_roi_bounding_box(img_bgr: np.ndarray) -> Tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    H, W = img_bgr.shape[:2]
    if w * h < 0.01 * W * H:
        return None
    return int(x), int(y), int(w), int(h)


def load_yolo_seg(label_file: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    polys: List[Tuple[int, np.ndarray]] = []
    if not label_file.exists():
        return polys
    for line in label_file.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        xs = np.array(coords[0::2]) * img_w
        ys = np.array(coords[1::2]) * img_h
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        polys.append((cls, pts))
    return polys


def save_yolo_seg(path: Path, polys: List[Tuple[int, np.ndarray]], w: int, h: int) -> None:
    lines: List[str] = []
    for cls, pts in polys:
        xs = np.clip(pts[:, 0] / float(w), 0, 1)
        ys = np.clip(pts[:, 1] / float(h), 0, 1)
        coords: List[str] = []
        for x, y in zip(xs, ys):
            coords += [f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join([str(cls)] + coords))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make ROI crops for part of train and remap masks")
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--ratio", type=float, default=0.5)
    args = ap.parse_args()

    imgs_dir = args.dataset_root / "dataset/train/images"
    labs_dir = args.dataset_root / "dataset/labels/train"
    out_imgs = args.dataset_root / "dataset/train_crops/images"
    out_labs = args.dataset_root / "dataset/labels/train_crops"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_labs.mkdir(parents=True, exist_ok=True)

    images = [p for p in imgs_dir.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    random.shuffle(images)
    take = int(len(images) * args.ratio)
    images = images[:take]

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        bbox = detect_table_roi_bounding_box(bgr)
        if bbox is None:
            continue
        x, y, w, h = bbox
        crop = bgr[y : y + h, x : x + w]
        cv2.imwrite(str(out_imgs / img_path.name), crop)
        # labels
        lab_path = labs_dir / (img_path.stem + ".txt")
        polys = load_yolo_seg(lab_path, W, H)
        new_polys: List[Tuple[int, np.ndarray]] = []
        for cls, pts in polys:
            pts2 = pts.copy()
            pts2[:, 0] = pts2[:, 0] - x
            pts2[:, 1] = pts2[:, 1] - y
            new_polys.append((cls, pts2))
        save_yolo_seg(out_labs / (img_path.stem + ".txt"), new_polys, w, h)

    print(f"[ok] crops: {out_imgs} labels: {out_labs}")


if __name__ == "__main__":
    main()


