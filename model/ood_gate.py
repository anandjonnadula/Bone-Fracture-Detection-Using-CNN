"""Out-of-distribution gate: refuse inputs that are not bone X-rays.

Two cheap layers, no extra labels needed:

1. Heuristic color pre-filter (microseconds) — radiographs are effectively
   monochrome. A strongly colored image (selfie, document photo, cat) is
   rejected before any model runs.
2. Embedding distance gate — the input's MobileNetV2 GAP embedding (the
   same features Stage 1 already computes) is compared against a reference
   sample of *training* embeddings via k-NN cosine distance. Anything far
   from the training manifold (blank images, natural photos that happen to
   be grayscale, documents) is rejected.

Reference statistics live in saved_model/ood_stats.npz, produced once by
build_ood_stats.py. The acceptance threshold is the 99.5th percentile of
training-set distances (recorded in the artifact for provenance).
"""

import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(BASE_DIR, "saved_model", "ood_stats.npz")

# Mean HSV saturation above which an image is considered "colored".
SATURATION_LIMIT = 0.25

REASON_COLOR = "color_image"
REASON_NOT_XRAY = "not_xray_like"

_stats_cache = {}


def load_stats(path=STATS_PATH):
    """Load (and cache) the reference embeddings + threshold."""
    if path not in _stats_cache:
        if not os.path.exists(path):
            return None
        data = np.load(path)
        _stats_cache[path] = {
            "refs": data["refs"].astype(np.float32),  # (N, D), L2-normalized
            "k": int(data["k"]),
            "threshold": float(data["threshold"]),
            "percentile": float(data["percentile"]),
        }
    return _stats_cache[path]


def mean_saturation(arr_rgb):
    """Mean HSV saturation of an RGB uint8/float array, in [0, 1]."""
    arr = np.asarray(arr_rgb, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(sat.mean())


def knn_distance(embedding, stats):
    """Mean cosine distance to the k nearest reference embeddings."""
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return float("inf")  # degenerate embedding (e.g. blank input)
    emb = emb / norm
    dists = 1.0 - stats["refs"] @ emb
    k = min(stats["k"], len(dists))
    nearest = np.partition(dists, k - 1)[:k]
    return float(nearest.mean())


def check(arr_rgb, embed_fn, stats_path=STATS_PATH):
    """Gate an image before Stage 1.

    arr_rgb   — RGB array (any float/uint8 range) already resized for the model.
    embed_fn  — callable(arr) -> backbone GAP embedding (1D array).

    Returns (ok: bool, reason: str | None, details: dict).
    """
    sat = mean_saturation(arr_rgb)
    if sat > SATURATION_LIMIT:
        return False, REASON_COLOR, {"saturation": round(sat, 4)}

    stats = load_stats(stats_path)
    if stats is None:
        # No artifact (e.g. models retrained but stats not rebuilt): fail open,
        # but make it visible in the details for logging.
        return True, None, {"saturation": round(sat, 4), "gate": "stats_missing"}

    dist = knn_distance(embed_fn(arr_rgb), stats)
    if dist > stats["threshold"]:
        return False, REASON_NOT_XRAY, {
            "saturation": round(sat, 4),
            "knn_distance": round(dist, 4),
            "threshold": round(stats["threshold"], 4),
        }
    return True, None, {
        "saturation": round(sat, 4),
        "knn_distance": round(dist, 4),
        "threshold": round(stats["threshold"], 4),
    }
