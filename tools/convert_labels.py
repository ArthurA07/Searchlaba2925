#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# YOLO-Seg line format: "<class_id> x1 y1 x2 y2 ..." in normalized coords [0..1]

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


def ls_to_yolo(ls_json: Path, out_dir: Path, class_map: Dict[str, int], issues: List[str]) -> None:
    data = json.loads(ls_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in data:
        img_name = Path(task.get("data", {}).get("image", "")).name
        w = int(task.get("meta", {}).get("width") or task.get("image_width") or task.get("original_width") or 0)
        h = int(task.get("meta", {}).get("height") or task.get("image_height") or task.get("original_height") or 0)
        polys: List[Tuple[int, np.ndarray]] = []
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                val = res.get("value", {})
                pts = val.get("points") or val.get("polygon")
                labels = val.get("polygonlabels") or val.get("labels")
                if not pts or not labels:
                    continue
                cls_name = labels[0]
                if cls_name not in class_map:
                    issues.append(f"Unknown class in LS: {cls_name}")
                    continue
                arr = np.array(pts, dtype=float)
                xs = arr[:, 0] * w / 100.0
                ys = arr[:, 1] * h / 100.0
                xy = np.stack([xs, ys], axis=1)
                polys.append((class_map[cls_name], xy.astype(np.float32)))
        out_path = out_dir / (Path(img_name).stem + ".txt")
        save_yolo_seg(out_path, polys, w, h)


def cvat_to_yolo(cvat_json: Path, out_dir: Path, id_map: Dict[int, int], issues: List[str]) -> None:
    data = json.loads(cvat_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = {im["id"]: im for im in data.get("images", [])}
    by_image: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for ann in data.get("annotations", []):
        if ann.get("type") != "polygon" and "segmentation" not in ann:
            continue
        img_id = ann["image_id"]
        seg = ann.get("segmentation")
        if not seg:
            continue
        coords = np.array(seg, dtype=float).reshape(-1, 2)
        cls_id = id_map.get(int(ann.get("category_id", -1)))
        if cls_id is None:
            issues.append(f"Unknown CVAT category_id: {ann.get('category_id')}")
            continue
        by_image.setdefault(img_id, []).append((cls_id, coords))
    for img_id, polys in by_image.items():
        im = imgs[img_id]
        w, h = int(im["width"]), int(im["height"])
        name = Path(im["file_name"]).stem + ".txt"
        save_yolo_seg(out_dir / name, polys, w, h)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Label Studio/CVAT to YOLO-Seg with classes.json mapping")
    ap.add_argument("mode", choices=["ls", "cvat"])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--dataset-root", required=False, type=Path, help="root with dataset/{train,val,test}/images")
    ap.add_argument("--out", required=False, type=Path, help="output dir if dataset-root is not provided")
    args = ap.parse_args()

    classes_path = Path(__file__).resolve().parents[1] / "labels/classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    slug_to_id: Dict[str, int] = {c["slug"]: int(c["id"]) for c in classes}

    issues: List[str] = []

    if args.dataset_root:
        ds = args.dataset_root
        out_train = ds / "dataset/labels/train"
        out_val = ds / "dataset/labels/val"
        out_test = ds / "dataset/labels/test"
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)
        split_map: Dict[str, Path] = {}
        for split, p in {"train": ds/"dataset/train/images", "val": ds/"dataset/val/images", "test": ds/"dataset/test/images"}.items():
            if p.is_dir():
                for img in p.glob("*.*"):
                    split_map[img.stem] = {"train": out_train, "val": out_val, "test": out_test}[split]
        tmp_out = ds / "dataset/labels/tmp"
        tmp_out.mkdir(parents=True, exist_ok=True)
        if args.mode == "ls":
            ls_to_yolo(args.input, tmp_out, slug_to_id, issues)
        else:
            id_map = {i: i for i in range(len(classes))}
            cvat_to_yolo(args.input, tmp_out, id_map, issues)
        for lab in tmp_out.glob("*.txt"):
            stem = lab.stem
            target_dir = split_map.get(stem)
            if target_dir is None:
                issues.append(f"No split for label: {stem}")
                continue
            target = target_dir / lab.name
            target.write_text(lab.read_text(encoding="utf-8"), encoding="utf-8")
        for f in tmp_out.glob("*.txt"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmp_out.rmdir()
        except Exception:
            pass
        out_dir = ds / "dataset/labels"
    else:
        out_dir = args.out if args.out else Path("out_labels")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "ls":
            ls_to_yolo(args.input, out_dir, slug_to_id, issues)
        else:
            id_map = {i: i for i in range(len(classes))}
            cvat_to_yolo(args.input, out_dir, id_map, issues)

    issues_path = out_dir / "../issues.log" if args.dataset_root else (out_dir / "issues.log")
    if issues:
        issues_path.write_text("\n".join(issues), encoding="utf-8")
        print("[warn] issues:", issues_path)
    print("[ok] converted ->", out_dir)


if __name__ == "__main__":
    main()
