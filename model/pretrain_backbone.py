"""Domain-adaptive backbone pretraining on MURA (UPGRADE_PLAN 3.2, step 2).

Trains MobileNetV2 (ImageNet init) on MURA's normal-vs-abnormal task —
study-level labels propagated to images — so the backbone sees ~40k
radiographs before it ever touches this project's small fracture data.
The output is NOT a fracture model ("abnormal" ≠ "fracture"); it is an
initialization for train_model_stage{1,2}.py via --init-weights.

MURA requires signing Stanford's research-use agreement:
https://stanfordmlgroup.github.io/competitions/mura/  (see docs/TRAINING.md)

Expected layout (the official archive):

    <mura_root>/MURA-v1.1/
        train/XR_ELBOW/patient00011/study1_negative/image1.png
        valid/...

Labels come from the study folder name ("positive" = abnormal). The split
uses MURA's own train/valid patient split, so no patient straddles splits.

Usage (from model/):

    python pretrain_backbone.py <mura_root> [--epochs-head 6] [--epochs-ft 4] [--limit N]

Outputs:
    saved_model/backbone_radiograph.keras        (full model, for inspection)
    saved_model/backbone_radiograph.weights.h5   (what --init-weights consumes)
    saved_model/backbone_radiograph_meta.json
"""

import argparse
import json
import os
from datetime import UTC, datetime

import numpy as np
import tensorflow as tf
from cnn_model import build_stage1_model, unfreeze_top_layers
from data_utils import IMG_SIZE, build_augmenter, merge_histories

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
BATCH = 32
SEED = 1337

tf.keras.utils.set_random_seed(SEED)


def collect_mura_files(mura_root, subset):
    """[(path, label)] where label 1 = abnormal (study folder 'positive')."""
    root = os.path.join(mura_root, "MURA-v1.1", subset)
    if not os.path.isdir(root):
        raise SystemExit(f"MURA layout not found: {root}")
    pairs = []
    for dirpath, _dirs, files in os.walk(root):
        label = 1.0 if "positive" in os.path.basename(dirpath) else 0.0
        for f in files:
            if f.lower().endswith(".png") and not f.startswith("."):
                pairs.append((os.path.join(dirpath, f), label))
    return pairs


def make_dataset(pairs, augment=False, shuffle=False):
    paths = [p for p, _l in pairs]
    labels = [_l for _p, _l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # bilinear, matches pipeline
        return tf.cast(img, tf.float32), label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    if augment:
        augmenter = build_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mura_root", help="Directory containing MURA-v1.1/")
    ap.add_argument("--epochs-head", type=int, default=6)
    ap.add_argument("--epochs-ft", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional cap on training images (smoke runs)")
    args = ap.parse_args()

    train_pairs = collect_mura_files(args.mura_root, "train")
    val_pairs = collect_mura_files(args.mura_root, "valid")
    if args.limit:
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(train_pairs))[:args.limit]
        train_pairs = [train_pairs[i] for i in idx]
    n_abnormal = sum(label for _p, label in train_pairs)
    print(f"MURA: {len(train_pairs)} train / {len(val_pairs)} valid images "
          f"({n_abnormal:.0f} abnormal in train)")

    train_ds = make_dataset(train_pairs, augment=True, shuffle=True)
    val_ds = make_dataset(val_pairs)

    # Same architecture (and layer names) as Stage 1 so the weights file
    # loads cleanly into both stage models (the stage-2 head is skipped).
    model, base = build_stage1_model()
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=metrics)
    print("\n=== Phase 1: head training (backbone frozen) ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=2, restore_best_weights=True)],
                   verbose=2)

    unfreeze_top_layers(base, n_layers=40)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy", metrics=metrics)
    print("\n=== Phase 2: fine-tuning top backbone layers (BatchNorm frozen) ===")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=2, restore_best_weights=True)],
                   verbose=2)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, "backbone_radiograph.keras"))
    model.save_weights(os.path.join(SAVE_DIR, "backbone_radiograph.weights.h5"))

    history = merge_histories(h1, h2)
    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "task": "MURA normal-vs-abnormal (domain-adaptive pretraining)",
        "backbone": "MobileNetV2 (ImageNet -> MURA)",
        "train_images": len(train_pairs),
        "valid_images": len(val_pairs),
        "val_auc": max(history.get("val_auc", [0.0])),
        "note": ("Initialization only — 'abnormal' is not 'fracture'; never "
                 "evaluate this model on the fracture test sets directly."),
        "history": history,
    }
    with open(os.path.join(SAVE_DIR, "backbone_radiograph_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n[OK] Saved backbone_radiograph.keras / .weights.h5 / _meta.json")
    print("Next: python train_model_stage1.py --init-weights "
          "saved_model/backbone_radiograph.weights.h5   (same for stage 2)")


if __name__ == "__main__":
    main()
