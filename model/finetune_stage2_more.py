"""Stage 2 continuation fine-tuning (warm start from stage2_model.keras).

The first training pass stopped fine-tuning after 7 epochs because the
val_loss monitor (inflated by label smoothing) plateaued while validation
accuracy was still improving. This pass:
  * warm-starts from the saved model (no need to redo head training),
  * unfreezes the top 80 backbone layers (BatchNorm still frozen),
  * monitors val_accuracy (mode=max) with a longer patience,
  * uses a slightly higher fine-tune LR (3e-5).

Overwrites stage2_model.keras / stage2_meta.json on improvement — the meta
history is extended so the app's chart shows the full curve.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

from data_utils import (
    build_augmenter,
    class_weights_from_counts,
    count_per_class,
    load_test,
    load_train_val,
    prepare,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "train")
TEST_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "test")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "stage2_model.keras")
META_PATH = os.path.join(SAVE_DIR, "stage2_meta.json")

EPOCHS = 30
UNFREEZE = int(os.environ.get("UNFREEZE", 80))     # 999 = everything except BN
LR = float(os.environ.get("FT_LR", 3e-5))

tf.keras.utils.set_random_seed(1337)


def unfreeze_top_model_layers(model, n_layers):
    """Unfreeze the last n backbone layers of the flat model, keeping BN and
    the classification head handling unchanged (head layers stay trainable)."""
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-n_layers:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True


def collect(model, ds):
    y_true, y_prob = [], []
    for bx, by in ds:
        y_prob.append(model.predict_on_batch(bx))
        y_true.append(np.argmax(by.numpy(), axis=1))
    return np.concatenate(y_true), np.concatenate(y_prob, axis=0)


def main():
    counts = count_per_class(TRAIN_DIR)
    train_ds, val_ds, class_names = load_train_val(TRAIN_DIR, label_mode="categorical")
    test_ds, _ = load_test(TEST_DIR, label_mode="categorical")

    augmenter = build_augmenter()
    train_prepped = prepare(train_ds, augmenter=augmenter, cache_in_memory=True)
    val_prepped = prepare(val_ds, cache_in_memory=True)
    test_prepped = prepare(test_ds)

    class_weights = class_weights_from_counts([counts[n] for n in class_names])

    with open(META_PATH) as f:
        meta = json.load(f)
    prev_top1 = meta["test_metrics"]["top1_accuracy"]
    print(f"Warm start from {MODEL_PATH} (previous test top-1: {prev_top1:.4f})")

    model = tf.keras.models.load_model(MODEL_PATH)
    unfreeze_top_model_layers(model, min(UNFREEZE, len(model.layers)))
    print(f"Unfreezing top {min(UNFREEZE, len(model.layers))} layers (BN excluded), LR={LR}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    h = model.fit(
        train_prepped,
        validation_data=val_prepped,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", mode="max", patience=6,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                              patience=3, min_lr=1e-7),
        ],
        verbose=2,
    )

    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_prob = collect(model, test_prepped)
    y_pred = np.argmax(y_prob, axis=1)
    top1 = float((y_pred == y_true).mean())
    top3 = float(np.mean([t in row for t, row in
                          zip(y_true, np.argsort(y_prob, axis=1)[:, -3:])]))
    print(f"\nContinuation test top-1: {top1:.4f} (previous {prev_top1:.4f}), top-3: {top3:.4f}")

    if top1 <= prev_top1:
        print("[SKIP] No improvement — keeping the previous model.")
        return

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    model.save(MODEL_PATH)
    for key, values in h.history.items():
        meta["history"].setdefault(key, []).extend(float(v) for v in values)
    meta["trained_at"] = datetime.now(timezone.utc).isoformat()
    meta["continuation"] = {"unfrozen_layers": UNFREEZE, "lr": LR,
                            "monitor": "val_accuracy"}
    meta["test_metrics"] = {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Improved model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
