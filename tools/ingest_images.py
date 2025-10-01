#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Item:
    id: str
    original_name: str
    original_relpath: str
    saved_name: str
    saved_relpath: str
    width: int
    height: int
    exif_fixed: bool
    roi_relpath: Optional[str]
    roi_bbox_xywh: Optional[Tuple[int, int, int, int]]
    kind: str = "single"  # single | group


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_class_map(class_map_path: Optional[Path]) -> Dict[str, str]:
    if not class_map_path:
        return {}
    if not class_map_path.exists():
        print(f"[warn] class-map not found: {class_map_path}")
        return {}
    try:
        if class_map_path.suffix.lower() in {".json"}:
            return json.loads(class_map_path.read_text(encoding="utf-8"))
        elif class_map_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore

            return yaml.safe_load(class_map_path.read_text(encoding="utf-8"))
        elif class_map_path.suffix.lower() in {".csv"}:
            mp: Dict[str, str] = {}
            import csv

            with open(class_map_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    mp[row["image"].strip()] = row["class"].strip()
            return mp
    except Exception as e:
        print(f"[warn] failed to parse class-map: {e}")
    return {}


def normalize_exif(im: Image.Image) -> Tuple[Image.Image, bool]:
    try:
        fixed = ImageOps.exif_transpose(im)
        return fixed, fixed is not im
    except Exception:
        return im, False


def detect_table_roi_bounding_box(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu threshold
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morph close to unify table region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Largest connected component bbox
    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # Filter out tiny bbox
    H, W = img_bgr.shape[:2]
    if w * h < 0.01 * W * H:
        return None
    return int(x), int(y), int(w), int(h)


def process_images(
    input_dir: Path,
    output_dir: Path,
    class_map: Dict[str, str],
    roi: bool,
    seed: int,
    split: Tuple[float, float, float],
) -> Tuple[List[Item], Dict[str, Dict[str, int]]]:
    originals_dir = output_dir / "originals"
    roi_dir = output_dir / "roi"
    ensure_dir(originals_dir)
    if roi:
        ensure_dir(roi_dir)

    items: List[Item] = []
    failures: List[str] = []

    # Try to read project-level .env.local for DATA_ROOT/GROUP_DIR/RULER_DIR/USE_SYMLINKS
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env.local"
    data_root = None
    group_dir_name = "Групповые для тренировки"
    ruler_dir_name = "Инструменты с линейкой"
    use_symlinks = True
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k == "DATA_ROOT":
                data_root = v
            elif k == "GROUP_DIR":
                group_dir_name = v
            elif k == "RULER_DIR":
                ruler_dir_name = v
            elif k == "USE_SYMLINKS":
                use_symlinks = v.lower() in {"1", "true", "yes"}

    # If DATA_ROOT provided, build file lists from there; else fallback to scanning input_dir
    singles_files: List[Tuple[Path, int]] = []
    group_files: List[Path] = []
    ruler_files: set[Path] = set()
    if data_root:
        dr = Path(data_root)
        # load classes for slug/id
        classes_path = root_dir / "labels/classes.json"
        classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else []
        # singles in folders 1..11 mapped to ids 0..10
        for cnum in range(1, 12):
            cdir = dr / str(cnum)
            if not cdir.is_dir():
                continue
            cid = cnum - 1
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and is_image(p):
                    singles_files.append((p, cid))
        gdir = dr / group_dir_name
        if gdir.is_dir():
            group_files = [p for p in gdir.rglob("*") if p.is_file() and is_image(p)]
        rdir = dr / ruler_dir_name
        if rdir.is_dir():
            ruler_files = set([p for p in rdir.rglob("*") if p.is_file() and is_image(p)])
        files: List[Path] = []
    else:
        files = [p for p in input_dir.rglob("*") if p.is_file() and is_image(p)]
        files.sort()

    counter = 0
    def add_item(src: Path, saved_name: str, kind: str, im2: Image.Image, width: int, height: int, exif_fixed: bool,
                 roi_relpath: Optional[str], roi_bbox: Optional[Tuple[int,int,int,int]]):
        nonlocal counter
        counter += 1
        item = Item(
            id=f"{counter:06d}",
            original_name=src.name,
            original_relpath=str(src.relative_to(input_dir)) if input_dir in src.parents else str(src),
            saved_name=saved_name,
            saved_relpath=str((originals_dir / saved_name).relative_to(output_dir)),
            width=width,
            height=height,
            exif_fixed=exif_fixed,
            roi_relpath=roi_relpath,
            roi_bbox_xywh=roi_bbox,
            kind=kind,
        )
        items.append(item)

    # Name helper from classes.json slugs
    slug_map: Dict[int, str] = {}
    cls_path = root_dir / "labels/classes.json"
    if cls_path.exists():
        for c in json.loads(cls_path.read_text(encoding="utf-8")):
            slug_map[int(c["id"])] = c["slug"]

    # ingest
    if data_root:
        # singles
        for src, cid in singles_files:
            if src in ruler_files:
                continue
            with Image.open(src) as im:
                im.load()
                im2, exif_fixed = normalize_exif(im)
                width, height = im2.size
                saved_name = f"{slug_map.get(cid, f'class{cid}')}_{(counter+1):06d}.jpg"
                dst = originals_dir / saved_name
                im2.convert("RGB").save(dst, quality=92)
                roi_relpath = None
                roi_bbox = None
                if roi:
                    bgr = cv2.cvtColor(np.array(im2.convert("RGB")), cv2.COLOR_RGB2BGR)
                    bbox = detect_table_roi_bounding_box(bgr)
                    if bbox is not None:
                        x, y, w, h = bbox
                        crop = bgr[y : y + h, x : x + w]
                        roi_path = roi_dir / saved_name
                        cv2.imwrite(str(roi_path), crop)
                        roi_relpath = str(roi_path.relative_to(output_dir))
                        roi_bbox = (x, y, w, h)
                add_item(src, saved_name, "single", im2, width, height, exif_fixed, roi_relpath, roi_bbox)
        # groups
        for src in group_files:
            with Image.open(src) as im:
                im.load()
                im2, exif_fixed = normalize_exif(im)
                width, height = im2.size
                saved_name = f"group_{(counter+1):06d}.jpg"
                dst = originals_dir / saved_name
                im2.convert("RGB").save(dst, quality=92)
                roi_relpath = None
                roi_bbox = None
                if roi:
                    bgr = cv2.cvtColor(np.array(im2.convert("RGB")), cv2.COLOR_RGB2BGR)
                    bbox = detect_table_roi_bounding_box(bgr)
                    if bbox is not None:
                        x, y, w, h = bbox
                        crop = bgr[y : y + h, x : x + w]
                        roi_path = roi_dir / saved_name
                        cv2.imwrite(str(roi_path), crop)
                        roi_relpath = str(roi_path.relative_to(output_dir))
                        roi_bbox = (x, y, w, h)
                add_item(src, saved_name, "group", im2, width, height, exif_fixed, roi_relpath, roi_bbox)
    else:
        for src in files:
            with Image.open(src) as im:
                im.load()
                im2, exif_fixed = normalize_exif(im)
                width, height = im2.size
                saved_name = f"img_{(counter+1):06d}.jpg"
                im2.convert("RGB").save(originals_dir / saved_name, quality=92)
                roi_relpath = None
                roi_bbox = None
                if roi:
                    bgr = cv2.cvtColor(np.array(im2.convert("RGB")), cv2.COLOR_RGB2BGR)
                    bbox = detect_table_roi_bounding_box(bgr)
                    if bbox is not None:
                        x, y, w, h = bbox
                        crop = bgr[y : y + h, x : x + w]
                        roi_path = roi_dir / saved_name
                        cv2.imwrite(str(roi_path), crop)
                        roi_relpath = str(roi_path.relative_to(output_dir))
                        roi_bbox = (x, y, w, h)
                add_item(src, saved_name, "single", im2, width, height, exif_fixed, roi_relpath, roi_bbox)
        try:
            with Image.open(src) as im:
                im.load()
                im2, exif_fixed = normalize_exif(im)
                width, height = im2.size
                # Unify name
                counter += 1
                saved_name = f"img_{counter:06d}.jpg"
                dst = originals_dir / saved_name
                im2.convert("RGB").save(dst, quality=92)

                roi_relpath = None
                roi_bbox = None
                if roi:
                    # Open with cv2 for ROI
                    bgr = cv2.cvtColor(np.array(im2.convert("RGB")), cv2.COLOR_RGB2BGR)
                    bbox = detect_table_roi_bounding_box(bgr)
                    if bbox is not None:
                        x, y, w, h = bbox
                        crop = bgr[y : y + h, x : x + w]
                        roi_path = roi_dir / saved_name
                        cv2.imwrite(str(roi_path), crop)
                        roi_relpath = str(roi_path.relative_to(output_dir))
                        roi_bbox = (x, y, w, h)

                item = Item(
                    id=f"{counter:06d}",
                    original_name=src.name,
                    original_relpath=str(src.relative_to(input_dir)),
                    saved_name=saved_name,
                    saved_relpath=str(dst.relative_to(output_dir)),
                    width=width,
                    height=height,
                    exif_fixed=exif_fixed,
                    roi_relpath=roi_relpath,
                    roi_bbox_xywh=roi_bbox,
                )
                items.append(item)
        except Exception as e:
            failures.append(f"{src}: {e}")

    if failures:
        (output_dir / "ingest_errors.log").write_text("\n".join(failures), encoding="utf-8")

    # Build splits (stratify if class_map provided)
    random.seed(seed)
    if data_root:
        # indices of singles/groups in items
        singles_idx = [i for i, it in enumerate(items) if it.kind == "single"]
        groups_idx = [i for i, it in enumerate(items) if it.kind == "group"]
        # stratify singles by slug prefix extracted from saved_name
        by_cls: Dict[str, List[int]] = {}
        for i in singles_idx:
            slug = items[i].saved_name.split("_")[0]
            by_cls.setdefault(slug, []).append(i)
        train, val, test = [], [], []
        for _, arr in by_cls.items():
            random.shuffle(arr)
            n = len(arr)
            n_train = int(n * split[0])
            n_val = int(n * split[1])
            train += arr[:n_train]
            val += arr[n_train : n_train + n_val]
            test += arr[n_train + n_val :]
        # groups random with guarantee val/test presence
        random.shuffle(groups_idx)
        gn = len(groups_idx)
        g_train = int(gn * split[0])
        g_val = int(gn * split[1])
        g_val_idx = groups_idx[g_train : g_train + g_val]
        g_test_idx = groups_idx[g_train + g_val :]
        if gn >= 1 and not g_val_idx:
            g_val_idx = groups_idx[:1]
        if gn >= 2 and not g_test_idx:
            g_test_idx = groups_idx[-1:]
        if g_val_idx:
            val += g_val_idx
        if g_test_idx:
            test += g_test_idx
    else:
        idxs = list(range(len(items)))
        random.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        train = idxs[:n_train]
        val = idxs[n_train : n_train + n_val]
        test = idxs[n_train + n_val :]

    # Create dataset directories with symlinks
    for split_name, subset in ("train", train), ("val", val), ("test", test):
        img_dir = output_dir / "dataset" / split_name / "images"
        roi_out = output_dir / "dataset" / split_name / "roi"
        ensure_dir(img_dir)
        ensure_dir(roi_out)
        for i in subset:
            it = items[i]
            src_img = output_dir / it.saved_relpath
            dst_img = img_dir / it.saved_name
            try:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                if use_symlinks:
                    os.symlink(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)
            if it.roi_relpath:
                src_roi = output_dir / it.roi_relpath
                dst_roi = roi_out / it.saved_name
                try:
                    if dst_roi.exists() or dst_roi.is_symlink():
                        dst_roi.unlink()
                    if use_symlinks:
                        os.symlink(src_roi, dst_roi)
                    else:
                        shutil.copy2(src_roi, dst_roi)
                except Exception:
                    shutil.copy2(src_roi, dst_roi)

    # Class report
    report: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    if class_map:
        def count_for(subset: List[int]) -> Dict[str, int]:
            mp: Dict[str, int] = {}
            for i in subset:
                it = items[i]
                c = class_map.get(it.original_name) or class_map.get(it.saved_name) or "__unknown__"
                mp[c] = mp.get(c, 0) + 1
            return dict(sorted(mp.items()))

        report["train"] = count_for(train)
        report["val"] = count_for(val)
        report["test"] = count_for(test)

    return items, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest images: EXIF-fix, ROI, manifest, split")
    ap.add_argument("--input", required=True, type=Path, help="Папка с исходными изображениями")
    ap.add_argument("--output", required=True, type=Path, help="Выходная папка для артефактов")
    ap.add_argument("--class-map", type=Path, default=None, help="JSON/YAML/CSV: image->class для отчёта и стратификации")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=(0.7, 0.2, 0.1))
    ap.add_argument("--no-roi", action="store_true", help="Не строить ROI")
    args = ap.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    roi = not args.no_roi
    split = tuple(args.split)
    assert abs(sum(split) - 1.0) < 1e-6, "split должен суммироваться к 1.0"

    ensure_dir(output_dir)
    class_map = read_class_map(args.class_map)
    items, report = process_images(input_dir, output_dir, class_map, roi, args.seed, split) 

    manifest = {
        "num_items": len(items),
        "items": [asdict(it) for it in items],
        "split": {"train": sum(1 for _ in report.get("train", {})), "val": sum(1 for _ in report.get("val", {})), "test": sum(1 for _ in report.get("test", {}))},
        "class_report": report,
        "config": {
            "data_root": data_root,
            "use_symlinks": use_symlinks,
        }
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {(output_dir / 'manifest.json')}\n[ok] originals: {(output_dir / 'originals')}\n[ok] roi: {(output_dir / 'roi') if roi else 'disabled'}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[error] {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)


