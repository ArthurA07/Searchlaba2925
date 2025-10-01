import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .image_utils import letterbox, pil_to_bgr, detect_table_roi_bounding_box

try:
    import onnxruntime as ort  # type: ignore
    _ORT_OK = True
except Exception:
    _ORT_OK = False

try:
    # Fallback to Ultralytics if ONNX is not available
    from ultralytics import YOLO  # type: ignore
    import cv2  # type: ignore
    _ULTRA_OK = True
except Exception:
    _ULTRA_OK = False


class ONNXEngine:
    def __init__(self) -> None:
        self.model_path = os.getenv("MODEL_PATH", "./backend/models/best.onnx")
        self.infer_size = int(os.getenv("INFER_SIZE", "2048"))
        self.roi_enable = os.getenv("ROI_ENABLE", "true").lower() in {"1", "true", "yes"}
        self.score_thr = float(os.getenv("SCORE_THR", "0.25"))
        self.iou_thr = float(os.getenv("IOU_THR", "0.5"))

        self.name = "onnx"
        self.session: Optional[ort.InferenceSession] = None

        if _ORT_OK and os.path.isfile(self.model_path):
            providers = [
                ("CUDAExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)  # type: ignore
                self.name = "onnx-ort"
            except Exception:
                # CPU only fallback
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])  # type: ignore
                self.name = "onnx-cpu"
        elif _ULTRA_OK:
            # fallback to Ultralytics (PyTorch) weights path from env if provided
            yolo_model = os.getenv("YOLO_MODEL", "yolov8s-seg.pt")
            self.yolo = YOLO(yolo_model)
            self.name = "yolo-fallback"
        else:
            self.yolo = None

    def _preprocess(self, image: Image.Image, roi_override: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        bgr = pil_to_bgr(image)
        meta: Dict[str, Any] = {
            "orig_hw": bgr.shape[:2],
            "roi_bbox": None,
            "used_imgsz": self.infer_size,
        }
        use_roi = self.roi_enable if roi_override is None else bool(roi_override)
        if use_roi:
            bbox = detect_table_roi_bounding_box(bgr)
            if bbox is not None:
                x, y, w, h = bbox
                bgr = bgr[y : y + h, x : x + w]
                meta["roi_bbox"] = [int(x), int(y), int(w), int(h)]
        lb, scale, (left, top) = letterbox(
            bgr,
            self.infer_size,
            stride=32,
            scaleFill=False,
            auto=True,
            scaleup=True,
        )
        meta.update({"scale": float(scale), "pad": [int(left), int(top)]})
        # NCHW float32 [0,1]
        inp = lb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        return inp, meta

    def _postprocess_dummy(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Minimal placeholder: no detections
        return []

    def _postprocess_onnx(self, outputs: List[np.ndarray], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Attempt to parse common YOLOv8-seg ONNX export.
        # We support two layouts produced by Ultralytics export:
        # - (1, A, 4+nc[+obj][+nm])
        # - (1, C, A) where C == 4+nc[+obj][+nm]
        # Second output (mask proto) is ignored here for stability; we return boxes + class scores.
        detections: List[Dict[str, Any]] = []
        if not outputs:
            return detections
        preds = outputs[0]
        if preds is None:
            return detections
        preds = np.array(preds)

        # Normalize to shape (A, C)
        if preds.ndim == 3:
            # (1, A, C) or (1, C, A)
            b, d1, d2 = preds.shape
            if b > 1:
                preds = preds[0]
                d1, d2 = preds.shape
            if d1 < d2:
                # (A, C)
                preds = preds
            else:
                # (C, A) -> (A, C)
                preds = preds.transpose(1, 0)
        elif preds.ndim == 2:
            # assume already (A, C)
            pass
        else:
            return detections

        if preds.ndim != 2 or preds.shape[1] < 5:
            return detections

        num_tools = int(os.getenv("NUM_TOOLS", "11"))

        # Columns mapping
        # Try [x,y,w,h, classes..., mask_coeffs...]
        C = preds.shape[1]
        cls_start = 4
        cls_end = min(4 + num_tools, C)
        # If there is an extra objectness column after boxes
        if C >= 5 + num_tools and C <= 5 + num_tools + 64:
            # layout: [x,y,w,h,obj, classes..., (mask coeffs...)]
            obj = preds[:, 4:5]
            cls_start = 5
            cls_end = min(5 + num_tools, C)
        else:
            obj = None

        x, y, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        cls_scores = preds[:, cls_start:cls_end]
        if cls_scores.size == 0:
            # Fallback: take last num_tools columns (sometimes exporter appends masks first)
            cls_scores = preds[:, -num_tools:]
        # Apply objectness if present
        if obj is not None and cls_scores.shape[0] == obj.shape[0]:
            cls_scores = cls_scores * obj

        # xywh -> xyxy (letterbox inverse applied later)
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        # xywh -> xyxy (letterbox inverse applied later)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        # unletterbox back
        left, top = float(meta.get("pad", [0, 0])[0]), float(meta.get("pad", [0, 0])[1])
        scale = float(meta.get("scale", 1.0))
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        if meta.get("roi_bbox"):
            rx, ry, _, _ = meta["roi_bbox"]
            x1 += rx; y1 += ry; x2 += rx; y2 += ry
        for i in range(preds.shape[0]):
            score = float(cls_conf[i])
            if score < self.score_thr:
                continue
            cid = int(cls_ids[i])
            detections.append({
                "class_id": cid,
                "name": f"tool_{(cid % num_tools) + 1}",
                "score": score,
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "present": True,
            })
        return detections

    def predict(self, image: Image.Image, roi_override: Optional[bool] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        inp, meta = self._preprocess(image, roi_override=roi_override)
        t0 = time.time()
        if self.session is not None:
            try:
                inputs = {self.session.get_inputs()[0].name: inp}
                out_list = self.session.run(None, inputs)
                detections = self._postprocess_onnx(out_list, meta)
            except Exception:
                detections = self._postprocess_dummy(meta)
        elif _ULTRA_OK and getattr(self, "yolo", None) is not None:
            # Fallback via Ultralytics for functionality
            bgr = pil_to_bgr(image)
            if meta.get("roi_bbox"):
                x, y, w, h = meta["roi_bbox"]
                bgr = bgr[y : y + h, x : x + w]
            import cv2  # local import for type checker

            resized, scale, (left, top) = letterbox(
                bgr, self.infer_size, stride=32, scaleFill=False, auto=True, scaleup=True
            )
            res = self.yolo.predict(source=resized, imgsz=self.infer_size, conf=self.score_thr, verbose=False)
            detections: List[Dict[str, Any]] = []
            for r in res:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                # Optional masks
                masks = getattr(r, "masks", None)
                masks_np = None
                if masks is not None and hasattr(masks, "data"):
                    try:
                        masks_np = masks.data.cpu().numpy()  # (N,H,W) float/bool
                    except Exception:
                        masks_np = None

                def to_full_image_rle(mask_letterboxed: np.ndarray) -> Optional[Dict[str, Any]]:
                    try:
                        H_res, W_res = mask_letterboxed.shape
                        # Unletterbox to ROI size
                        nh = int(round(bgr.shape[0] * scale))
                        nw = int(round(bgr.shape[1] * scale))
                        crop = mask_letterboxed[int(top): int(top)+nh, int(left): int(left)+nw]
                        roi_mask = cv2.resize(crop.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                        # Place into full image if ROI used
                        if meta.get("roi_bbox"):
                            rx, ry, rw, rh = meta["roi_bbox"]
                            full = np.zeros((meta["orig_hw"][0], meta["orig_hw"][1]), dtype=np.uint8)
                            full[ry:ry+rh, rx:rx+rw] = roi_mask
                        else:
                            full = roi_mask
                        # RLE encode (uncompressed counts)
                        h, w = full.shape
                        flat = full.flatten(order="C")
                        counts: List[int] = []
                        run = 0
                        prev = 0
                        for v in flat:
                            if v == prev:
                                run += 1
                            else:
                                counts.append(run)
                                run = 1
                                prev = v
                        counts.append(run)
                        return {"counts": counts, "size": [int(h), int(w)]}
                    except Exception:
                        return None

                for i, b in enumerate(r.boxes):
                    cls = int(b.cls.item()) if hasattr(b, "cls") else 0
                    score = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                    xyxy = b.xyxy.squeeze().tolist()
                    x1, y1, x2, y2 = xyxy
                    x1 = (x1 - left) / scale
                    y1 = (y1 - top) / scale
                    x2 = (x2 - left) / scale
                    y2 = (y2 - top) / scale
                    if meta.get("roi_bbox"):
                        rx, ry, _, _ = meta["roi_bbox"]
                        x1 += rx
                        y1 += ry
                        x2 += rx
                        y2 += ry
                    det: Dict[str, Any] = {
                        "class_id": cls,
                        "name": f"tool_{(cls % 11) + 1}",
                        "score": score,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "present": bool(score >= self.score_thr),
                    }
                    if masks_np is not None and i < masks_np.shape[0]:
                        rle = to_full_image_rle(masks_np[i] > 0.5)
                        if rle is not None:
                            det["mask_rle"] = rle
                    detections.append(det)
        else:
            detections = self._postprocess_dummy(meta)
        infer_ms = (time.time() - t0) * 1000.0
        meta["infer_ms"] = infer_ms
        return detections, meta


_ENGINE: Optional[ONNXEngine] = None


def get_onnx_engine() -> ONNXEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ONNXEngine()
    return _ENGINE


