import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, UploadFile, Query
from PIL import Image

from ..services.onnx_engine import get_onnx_engine


router = APIRouter(prefix="/api/v1", tags=["v1"])


def _outputs_dir() -> Path:
    now = datetime.now()
    base = Path("outputs") / now.strftime("%Y-%m-%d")
    base.mkdir(parents=True, exist_ok=True)
    return base


@router.post("/infer")
async def infer(
    file: UploadFile = File(...),
    score_thr: float = Query(None, description="override SCORE_THR"),
    roi: bool | None = Query(None, description="override ROI (true/false)")
) -> Dict[str, Any]:
    engine = get_onnx_engine()
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    dets, meta = engine.predict(image, roi_override=roi)
    if score_thr is not None:
        dets = [d for d in dets if d.get("score", 0.0) >= score_thr]

    name_stem = Path(file.filename or "image").stem
    stamp = datetime.now().strftime("%H%M%S")
    out_dir = _outputs_dir()
    payload = {
        "detections": dets,
        "image_size": [image.width, image.height],
        "used_imgsz": meta.get("used_imgsz"),
        "roi_bbox": meta.get("roi_bbox"),
        "infer_ms": meta.get("infer_ms"),
        "engine": engine.name,
        "source_name": file.filename,
    }
    json_filename = f"{stamp}_{name_stem}.json"
    viz_filename = f"{stamp}_{name_stem}_viz.png"
    json_path = out_dir / json_filename
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Simple viz placeholder: save original image
    image.save(out_dir / viz_filename)
    # add saved paths (served via /outputs)
    saved_base = f"/outputs/{out_dir.name}/"
    payload["saved_json"] = saved_base + json_filename
    payload["saved_viz"] = saved_base + viz_filename
    return payload


_THRESHOLD: float = float(os.getenv("THRESHOLD", "0.98"))


@router.get("/config/threshold")
def get_threshold() -> Dict[str, Any]:
    return {"threshold": _THRESHOLD}


@router.post("/config/threshold")
def set_threshold(value: float = Query(..., ge=0.0, le=1.0)) -> Dict[str, Any]:
    global _THRESHOLD
    _THRESHOLD = value
    return {"threshold": _THRESHOLD}


@router.post("/match")
async def match(
    file: UploadFile = File(...),
    threshold: float = Query(None, description="override global threshold"),
) -> Dict[str, Any]:
    global _THRESHOLD
    thr = threshold if threshold is not None else _THRESHOLD
    engine = get_onnx_engine()
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    dets, meta = engine.predict(image)

    # Collapse to 11-class presence with best score per class
    num_tools = 11
    best: List[float] = [0.0] * num_tools
    for d in dets:
        name = d.get("name", "")
        if name.startswith("tool_"):
            idx = int(name.split("_")[1]) - 1
            if 0 <= idx < num_tools:
                best[idx] = max(best[idx], float(d.get("score", 0.0)))
    results = [
        {"name": f"tool_{i+1}", "present": best[i] >= thr, "score": best[i]} for i in range(num_tools)
    ]
    manual_recount = not all(best[i] >= thr for i in range(num_tools))
    payload = {
        "results": results,
        "passed_threshold": all(r["present"] for r in results),
        "manual_recount": manual_recount,
        "threshold": thr,
        "image_size": [image.width, image.height],
        "used_imgsz": meta.get("used_imgsz"),
        "roi_bbox": meta.get("roi_bbox"),
        "infer_ms": meta.get("infer_ms"),
        "engine": engine.name,
    }
    return payload


@router.get("/bench")
def bench_info() -> Dict[str, Any]:
    engine = get_onnx_engine()
    return {"engine": engine.name, "infer_size": int(os.getenv("INFER_SIZE", "2048")), "roi": os.getenv("ROI_ENABLE", "true")}


@router.post("/bench")
async def bench_run(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    engine = get_onnx_engine()
    stats: List[Dict[str, Any]] = []
    for uf in files:
        data = await uf.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        dets, meta = engine.predict(image)
        stats.append({
            "name": uf.filename,
            "infer_ms": meta.get("infer_ms"),
            "found": len(dets),
            "avg_score": (sum(d.get("score", 0.0) for d in dets) / len(dets)) if dets else 0.0,
        })
    total_ms = sum(s["infer_ms"] for s in stats if s.get("infer_ms") is not None)
    avg_ms = total_ms / max(1, len(stats))
    return {"engine": engine.name, "count": len(stats), "avg_ms": avg_ms, "items": stats}


