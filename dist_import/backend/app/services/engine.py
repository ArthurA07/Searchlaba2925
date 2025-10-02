import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image


class DummyEngine:
    name = "dummy-seg"

    def __init__(self) -> None:
        self.num_tools = int(os.getenv("NUM_TOOLS", "11"))

    def predict(self, image: Image.Image, threshold: float = 0.5) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(abs(hash(image.size)) % (2**32))
        scores = rng.random(self.num_tools)
        results: List[Dict[str, Any]] = []
        for idx in range(self.num_tools):
            score = float(scores[idx])
            present = bool(score >= threshold)
            results.append({"id": idx + 1, "present": present, "score": score})
        return results


try:
    from ultralytics import YOLO  # type: ignore
    import cv2  # type: ignore

    class YOLOEngine:
        name = "yolo-seg"

        def __init__(self) -> None:
            model_path = os.getenv("YOLO_MODEL", "yolov8s-seg.pt")
            self.model = YOLO(model_path)
            self.num_tools = int(os.getenv("NUM_TOOLS", "11"))

        def _pil_to_np(self, image: Image.Image) -> np.ndarray:
            arr = np.array(image.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        def predict(self, image: Image.Image, threshold: float = 0.5) -> List[Dict[str, Any]]:
            frame = self._pil_to_np(image)
            results = self.model.predict(source=frame, imgsz=640, conf=threshold, verbose=False)
            scores = np.zeros(self.num_tools, dtype=float)
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls.item()) if hasattr(b, "cls") else 0
                    score = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                    tool_id = (cls % self.num_tools) + 1
                    scores[tool_id - 1] = max(scores[tool_id - 1], score)
            out: List[Dict[str, Any]] = []
            for idx in range(self.num_tools):
                sc = float(scores[idx])
                out.append({"id": idx + 1, "present": sc >= threshold, "score": sc})
            return out

    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False


_ENGINE: Any | None = None


def get_engine() -> Any:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    prefer_yolo = os.getenv("USE_YOLO", "1") in {"1", "true", "yes"}
    if prefer_yolo and _YOLO_AVAILABLE:
        _ENGINE = YOLOEngine()
    else:
        _ENGINE = DummyEngine()
    return _ENGINE


