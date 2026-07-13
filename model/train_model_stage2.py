"""Stage 2 trainer — 12-way fracture-type classification.

Improvements over the original script:
  * Fine-tuning now unfreezes only the TOP backbone layers and keeps every
    BatchNormalization layer frozen. The old script set the whole MobileNetV2
    trainable — retraining BN statistics on ~1.3k images with batch size 32
    wrecks the pretrained features.
  * Label smoothing (0.1) — strong regularizer for a 12-class problem on a
    small dataset.
  * Class weights for the mild imbalance (82-159 images per class).
  * Proper validation split from the TRAIN set; the test set is only touched
    once, for the final report (the old script validated on test).
  * Small-dataset caching for fast epochs, real history + per-class test
    report saved to stage2_meta.json.
"""

import argparse
import json
import os
from datetime import UTC, datetime

import numpy as np
import tensorflow as tf
from cnn_model import build_stage2_model, unfreeze_top_layers
from data_utils import (
    build_augmenter,
    class_weights_from_counts,
    count_per_class,
    load_test,
    load_train_val,
    merge_histories,
    prepare,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "train")
TEST_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "test")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 15

tf.keras.utils.set_random_seed(1337)


def collect_probs_and_labels(model, ds):
    y_true, y_prob = [], []
    for batch_x, batch_y in ds:
        y_prob.append(model.predict_on_batch(batch_x))
        y_true.append(np.argmax(batch_y.numpy(), axis=1))
    return np.concatenate(y_true), np.concatenate(y_prob, axis=0)


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 2 trainer")
    ap.add_argument(
        "--init-weights", default=None,
        help="Optional weights file to initialize from (e.g. "
             "saved_model/backbone_radiograph.weights.h5 from pretrain_backbone.py); "
             "the classification head is skipped via skip_mismatch.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    counts = count_per_class(TRAIN_DIR)
    print("Train counts:", counts)

    train_ds, val_ds, class_names = load_train_val(TRAIN_DIR, label_mode="categorical")
    print("Class names (folder order):", class_names)
    test_ds, _ = load_test(TEST_DIR, label_mode="categorical")

    augmenter = build_augmenter()
    # Dataset is small (~1.3k images) — cache decoded images in memory.
    train_prepped = prepare(train_ds, augmenter=augmenter, cache_in_memory=True)
    val_prepped = prepare(val_ds, cache_in_memory=True)
    test_prepped = prepare(test_ds)

    class_weights = class_weights_from_counts([counts[name] for name in class_names])
    print("Class weights:", class_weights)

    model, base = build_stage2_model(num_classes=len(class_names))
    if args.init_weights:
        model.load_weights(args.init_weights, skip_mismatch=True)
        print(f"Initialized weights from {args.init_weights} (head skipped)")
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    metrics = [
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
    ]

    # ---- Phase 1: train the classification head on frozen features ----
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
    print("\n=== Phase 1: head training (backbone frozen) ===")
    h1 = model.fit(
        train_prepped,
        validation_data=val_prepped,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ],
        verbose=2,
    )

    # ---- Phase 2: fine-tune top of the backbone at a low LR ----
    unfreeze_top_layers(base, n_layers=40)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss, metrics=metrics)
    print("\n=== Phase 2: fine-tuning top backbone layers (BatchNorm frozen) ===")
    h2 = model.fit(
        train_prepped,
        validation_data=val_prepped,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ],
        verbose=2,
    )

    # ---- Final evaluation on the held-out test set ----
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n=== Final evaluation (held-out test set) ===")
    y_true, y_prob = collect_probs_and_labels(model, test_prepped)
    y_pred = np.argmax(y_prob, axis=1)
    top1 = float((y_pred == y_true).mean())
    top3 = float(
        np.mean([t in row for t, row in zip(y_true, np.argsort(y_prob, axis=1)[:, -3:], strict=False)])
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"Top-1 accuracy: {top1:.4f}   Top-3 accuracy: {top3:.4f}")

    # ---- Save ----
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "stage2_model.keras")
    model.save(model_path)

    # Keep class_indices.json for backward compatibility with the app.
    with open(os.path.join(BASE_DIR, "class_indices.json"), "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f)

    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "backbone": ("MobileNetV2 (radiograph-pretrained)" if args.init_weights
                     else "MobileNetV2 (ImageNet)"),
        "init_weights": args.init_weights,
        "input_size": 224,
        "output_semantics": "softmax over fracture types; raw 0-255 RGB input",
        "class_names": class_names,
        "train_counts": counts,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "label_smoothing": 0.1,
        "history": merge_histories(h1, h2),
        "test_metrics": {
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "per_class": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        },
    }
    with open(os.path.join(SAVE_DIR, "stage2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Stage 2 model saved to {model_path}")
    print("[OK] Metadata + history saved to stage2_meta.json")


if __name__ == "__main__":
    main()
