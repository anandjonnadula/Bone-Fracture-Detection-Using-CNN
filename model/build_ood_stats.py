"""One-time script: build reference statistics for the out-of-distribution gate.

Passes every TRAINING image through the Stage-1 backbone's global-average-
pooling layer, L2-normalizes the embeddings, and stores a reference sample
for a k-NN cosine-distance check. The acceptance threshold is the 99.5th
percentile of training-set distances, sanity-checked against synthetic
obvious negatives (blank/noise/gradient images) which must land far outside.

Run from the model/ directory after (re)training Stage 1:

    python build_ood_stats.py

Output: saved_model/ood_stats.npz  (committed — it is a model artifact).
"""

import os
from datetime import UTC, datetime

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from model import predict
except ImportError:
    import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "train")
OUT_PATH = os.path.join(BASE_DIR, "saved_model", "ood_stats.npz")

IMG_SIZE = predict.IMG_SIZE
BATCH = 32
K = 10
PERCENTILE = 99.5
MAX_REFS = 2000
SEED = 1337

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_train_images():
    paths = []
    for root, _dirs, files in os.walk(TRAIN_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def build_embedder():
    entry = predict.get_model("stage1")
    model = entry["model"]
    gap = model.get_layer("gap").output
    return tf.keras.Model(model.inputs, gap), entry["legacy"]


def embed_paths(embedder, legacy, paths):
    embs = []
    for start in range(0, len(paths), BATCH):
        batch_paths = paths[start:start + BATCH]
        arrs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
            )
            arrs.append(np.asarray(img, dtype=np.float32))
        x = np.stack(arrs)
        if legacy:
            x = x / 255.0
        embs.append(embedder.predict(x, verbose=0))
        done = min(start + BATCH, len(paths))
        if done % (BATCH * 10) < BATCH:
            print(f"  embedded {done}/{len(paths)}")
    embs = np.concatenate(embs).astype(np.float32)
    embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
    return embs


def synthetic_negatives():
    """Obvious non-X-ray images that must be rejected (all grayscale, so
    they exercise the embedding gate rather than the color pre-filter)."""
    rng = np.random.default_rng(SEED)
    size = (IMG_SIZE, IMG_SIZE, 3)
    black = np.zeros(size, np.float32)
    white = np.full(size, 255.0, np.float32)
    noise = rng.uniform(0, 255, size).astype(np.float32)
    grad = np.tile(np.linspace(0, 255, IMG_SIZE, dtype=np.float32)[None, :, None], (IMG_SIZE, 1, 3))
    checker = (np.indices((IMG_SIZE, IMG_SIZE)).sum(axis=0) // 28 % 2 * 255).astype(np.float32)
    checker = np.repeat(checker[..., None], 3, axis=2)
    return {"black": black, "white": white, "noise": noise,
            "gradient": grad, "checkerboard": checker}


def main():
    paths = list_train_images()
    if not paths:
        raise SystemExit(f"No training images found under {TRAIN_DIR}")
    print(f"Embedding {len(paths)} training images...")
    embedder, legacy = build_embedder()
    embs = embed_paths(embedder, legacy, paths)

    rng = np.random.default_rng(SEED)
    n = len(embs)
    ref_idx = rng.choice(n, size=min(MAX_REFS, n), replace=False)
    refs = embs[ref_idx]

    # Training-set kNN distances (a ref's own column is masked out).
    print("Computing training-set kNN distances...")
    dists = 1.0 - embs @ refs.T  # (N, M)
    col_of = {int(g): c for c, g in enumerate(ref_idx)}
    for i, c in col_of.items():
        dists[i, c] = np.inf
    k_small = np.partition(dists, K - 1, axis=1)[:, :K]
    train_knn = k_small.mean(axis=1)

    threshold = float(np.percentile(train_knn, PERCENTILE))
    print(f"kNN distance: mean={train_knn.mean():.4f}  p{PERCENTILE}={threshold:.4f}  "
          f"max={train_knn.max():.4f}")

    # Sanity check: synthetic negatives must score far outside the threshold.
    print("Sanity-checking synthetic negatives...")
    neg_report = {}
    all_out = True
    for name, arr in synthetic_negatives().items():
        x = arr[None] / 255.0 if legacy else arr[None]
        e = embedder.predict(x, verbose=0)[0]
        norm = np.linalg.norm(e)
        if norm < 1e-8:
            d = float("inf")
        else:
            e = e / norm
            nd = 1.0 - refs @ e
            d = float(np.partition(nd, K - 1)[:K].mean())
        neg_report[name] = d
        status = "OK (rejected)" if d > threshold else "!! INSIDE threshold"
        if d <= threshold:
            all_out = False
        print(f"  {name:14s} distance={d:.4f}  {status}")

    if not all_out:
        print("WARNING: some synthetic negatives fall inside the threshold — "
              "consider a lower percentile.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        refs=refs.astype(np.float16),
        k=K,
        threshold=threshold,
        percentile=PERCENTILE,
        n_train=n,
        created_at=datetime.now(UTC).isoformat(),
    )
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"[OK] Wrote {OUT_PATH} ({size_mb:.1f} MB, {len(refs)} refs, "
          f"k={K}, threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
