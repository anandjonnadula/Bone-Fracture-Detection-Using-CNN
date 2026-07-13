"""Runtime fracture localization — degrades gracefully when not trained.

predict.py calls weights_available() before detect(); when the weights (or
the ultralytics package) are absent the pipeline simply skips the
localization step. Boxes are returned NORMALIZED (x, y = top-left; w, h)
so the viewer, PDF renderer and DB all share one representation.

Domain honesty: the detector is trained on PEDIATRIC WRIST radiographs
(GRAZPEDWRI-DX). Every surface that shows these boxes labels them
"Localization (beta — validated on wrist X-rays)".
"""

import json
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "detector.pt")
META_PATH = os.path.join(BASE_DIR, "weights", "detector_meta.json")

_model = None
_meta = None
_lock = threading.Lock()


def weights_available():
    if not os.path.exists(WEIGHTS_PATH):
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def _load():
    global _model, _meta
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(WEIGHTS_PATH)
                _meta = {}
                if os.path.exists(META_PATH):
                    with open(META_PATH) as f:
                        _meta = json.load(f)
    return _model, _meta


def detect(img_path):
    """Normalized fracture boxes: [{"x","y","w","h","conf"}, ...]."""
    model, meta = _load()
    conf = float(meta.get("confidence_threshold", 0.25))
    with _lock:  # single-instance app; YOLO predict isn't guaranteed thread-safe
        results = model.predict(img_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        w, h = r.orig_shape[1], r.orig_shape[0]
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            boxes.append({
                "x": round(x1 / w, 4),
                "y": round(y1 / h, 4),
                "w": round((x2 - x1) / w, 4),
                "h": round((y2 - y1) / h, 4),
                "conf": round(float(b.conf[0]), 4),
            })
    return boxes
