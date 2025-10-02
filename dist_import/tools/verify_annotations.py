#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


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
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        polys.append((cls, pts))
    return polys


def overlay_masks(img_bgr: np.ndarray, polys: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    out = img_bgr.copy()
    for cls, pts in polys:
        color = tuple(int(c) for c in np.random.default_rng(cls).integers(50, 255, size=3))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(out, [pts], color=(*color,))
    overlay = cv2.addWeighted(out, 0.35, img_bgr, 0.65, 0)
    return overlay


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify YOLO-Seg annotations by overlaying masks (single+group)")
    ap.add_argument("--images-single", required=False, type=Path)
    ap.add_argument("--images-group", required=False, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num", type=int, default=30)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    singles = []
    groups = []
    if args.images_single and args.images_single.is_dir():
        singles = sorted([p for p in args.images_single.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if args.images_group and args.images_group.is_dir():
        groups = sorted([p for p in args.images_group.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    random.shuffle(singles)
    random.shuffle(groups)
    singles = singles[: args.num]
    groups = groups[: args.num]

    issues: List[str] = []

    def run_set(images: List[Path], prefix: str) -> None:
        nonlocal issues
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                issues.append(f"Cannot read image: {img_path}")
                continue
            h, w = bgr.shape[:2]
            lab_path = args.labels / (img_path.stem + ".txt")
            polys = load_yolo_seg(lab_path, w, h)
            for cls, pts in polys:
                area = float(cv2.contourArea(pts))
                if area < 10.0:
                    issues.append(f"Microscopic mask: {lab_path} cls={cls} area={area:.1f}")
            overlay = overlay_masks(bgr, polys)
            cv2.imwrite(str(args.out / f"{prefix}_{img_path.stem}.png"), overlay)

    if singles:
        run_set(singles, "single")
    if groups:
        run_set(groups, "group")

    if issues:
        (args.out / "issues.log").write_text("\n".join(issues), encoding="utf-8")
        print(f"[warn] issues: {(args.out / 'issues.log')}")
    print(f"[ok] overlays at: {args.out}")


if __name__ == "__main__":
    main()


