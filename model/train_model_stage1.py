"""Stage 1 trainer — binary fracture / no-fracture detection.

Improvements over the original scratch-CNN script:
  * MobileNetV2 transfer learning (ImageNet) instead of a 11M-param scratch
    CNN — better features, ~4x smaller saved model.
  * Class weights: the training set is heavily imbalanced
    (~705 fracture vs ~4640 no-fracture); without weighting, the model
    maximizes accuracy by under-calling fractures — the most dangerous
    failure mode in this domain.
  * Proper validation split carved from the TRAIN set. The old script used
    the test set for validation, which leaks it into model selection.
  * Two-phase training: frozen-backbone head training, then fine-tuning of
    the top backbone layers (BatchNorm kept frozen) at a low LR.
  * EarlyStopping + ReduceLROnPlateau + best-weights restore.
  * Decision-threshold tuning on the validation set (a fixed 0.5 threshold
    is rarely optimal under class imbalance).
  * Real training history + test metrics saved to stage1_meta.json — the
    web app displays these instead of hard-coded dummy numbers.

Label semantics: the model outputs sigmoid P(fracture). Folders sort as
['fracture', 'no_fracture'] -> labels 0/1, so labels are remapped (1 - y)
to make 'fracture' the positive class.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cnn_model import build_stage1_model, unfreeze_top_layers
from data_utils import (
    build_augmenter,
    class_weights_from_counts,
    count_per_class,
    load_test,
    load_train_val,
    merge_histories,
    prepare,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "train")
TEST_DIR = os.path.join(BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "test")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 8

tf.keras.utils.set_random_seed(1337)


def remap_fracture_positive(ds):
    """Folder order gives fracture=0; flip so the sigmoid means P(fracture)."""
    return ds.map(lambda x, y: (x, 1.0 - y), num_parallel_calls=tf.data.AUTOTUNE)


def collect_probs_and_labels(model, ds):
    y_true, y_prob = [], []
    for batch_x, batch_y in ds:
        y_prob.append(model.predict_on_batch(batch_x).reshape(-1))
        y_true.append(batch_y.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def tune_threshold(y_true, y_prob):
    """Pick the threshold that maximizes balanced accuracy on validation."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        score = (tpr + tnr) / 2
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def binary_metrics(y_true, y_prob, threshold):
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_fracture": float(precision_score(y_true, pred, zero_division=0)),
        "recall_fracture": float(recall_score(y_true, pred, zero_division=0)),
        "f1_fracture": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def main():
    counts = count_per_class(TRAIN_DIR)
    print("Train counts:", counts)

    train_ds, val_ds, class_names = load_train_val(TRAIN_DIR, label_mode="binary")
    print("Class names (folder order):", class_names)
    test_ds, _ = load_test(TEST_DIR, label_mode="binary")

    train_ds = remap_fracture_positive(train_ds)
    val_ds = remap_fracture_positive(val_ds)
    test_ds = remap_fracture_positive(test_ds)

    augmenter = build_augmenter()
    train_prepped = prepare(train_ds, augmenter=augmenter)
    val_prepped = prepare(val_ds)
    test_prepped = prepare(test_ds)

    # After remap: label 1 = fracture, label 0 = no_fracture
    class_weights = class_weights_from_counts(
        [counts["no_fracture"], counts["fracture"]]
    )
    print("Class weights {0: no_fracture, 1: fracture}:", class_weights)

    model, base = build_stage1_model()
    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    # ---- Phase 1: train the classification head on frozen features ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=metrics,
    )
    print("\n=== Phase 1: head training (backbone frozen) ===")
    h1 = model.fit(
        train_prepped,
        validation_data=val_prepped,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Phase 2: fine-tune top of the backbone at a low LR ----
    unfreeze_top_layers(base, n_layers=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=metrics,
    )
    print("\n=== Phase 2: fine-tuning top backbone layers (BatchNorm frozen) ===")
    h2 = model.fit(
        train_prepped,
        validation_data=val_prepped,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ],
        verbose=2,
    )

    # ---- Threshold tuning on validation, final evaluation on test ----
    print("\n=== Threshold tuning (validation set) ===")
    val_true, val_prob = collect_probs_and_labels(model, val_prepped)
    threshold, val_bal_acc = tune_threshold(val_true, val_prob)
    print(f"Tuned threshold: {threshold:.2f} (val balanced accuracy {val_bal_acc:.4f})")

    print("\n=== Final evaluation (held-out test set) ===")
    test_true, test_prob = collect_probs_and_labels(model, test_prepped)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_metrics_05 = binary_metrics(test_true, test_prob, 0.5)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # ---- Save ----
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "stage1_model.keras")
    model.save(model_path)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "backbone": "MobileNetV2 (ImageNet)",
        "input_size": 224,
        "positive_class": "fracture",
        "output_semantics": "sigmoid = P(fracture); raw 0-255 RGB input",
        "threshold": threshold,
        "val_balanced_accuracy_at_threshold": float(val_bal_acc),
        "train_counts": counts,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "history": merge_histories(h1, h2),
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": test_metrics_05,
    }
    with open(os.path.join(SAVE_DIR, "stage1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Stage 1 model saved to {model_path}")
    print(f"[OK] Metadata + history saved to stage1_meta.json")


if __name__ == "__main__":
    main()
