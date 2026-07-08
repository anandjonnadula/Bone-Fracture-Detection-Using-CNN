"""Offline evaluation of the two-stage pipeline on the held-out test sets.

Usage:
    python evaluate_models.py --tag baseline            # evaluate whatever is loadable
    python evaluate_models.py --stage1 saved_model/stage1_model.h5 --tag baseline
    python evaluate_models.py --tag retrained

Handles both model generations automatically:
  * legacy .h5  -> inputs scaled 1/255; stage-1 sigmoid means P(no_fracture)
  * new .keras  -> raw 0-255 inputs; stage-1 sigmoid means P(fracture),
                   decision threshold read from stage1_meta.json

Writes saved_model/eval_<stage>_<tag>.json and prints a summary.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from data_utils import load_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
STAGE1_TEST = os.path.join(BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "test")
STAGE2_TEST = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "test")


def load_any(path):
    """Load a model; fall back to the tf_keras legacy loader for old .h5 files."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        if path.endswith(".h5"):
            print(f"  keras load failed ({type(e).__name__}); trying tf_keras legacy loader...")
            import tf_keras

            return tf_keras.models.load_model(path, compile=False)
        raise


def default_model_path(stage):
    keras_path = os.path.join(SAVE_DIR, f"stage{stage}_model.keras")
    h5_path = os.path.join(SAVE_DIR, f"stage{stage}_model.h5")
    if os.path.exists(keras_path):
        return keras_path
    if os.path.exists(h5_path):
        return h5_path
    raise FileNotFoundError(f"No stage-{stage} model in {SAVE_DIR}")


def predict_all(model, ds, legacy):
    probs, labels = [], []
    for batch_x, batch_y in ds:
        x = batch_x / 255.0 if legacy else batch_x
        probs.append(model.predict_on_batch(x))
        labels.append(batch_y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def evaluate_stage1(model_path, tag):
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    legacy = model_path.endswith(".h5")
    print(f"\n=== Stage 1 evaluation: {os.path.basename(model_path)} "
          f"({'legacy' if legacy else 'new'}) ===")
    model = load_any(model_path)

    ds, class_names = load_test(STAGE1_TEST, label_mode="binary")
    print("  Folder order:", class_names)  # ['fracture', 'no_fracture']

    probs, labels = predict_all(model, ds, legacy)
    sigmoid = probs.reshape(-1)
    folder_label = labels.reshape(-1)  # 0 = fracture, 1 = no_fracture

    y_true = 1.0 - folder_label  # 1 = fracture (positive class)
    p_fracture = (1.0 - sigmoid) if legacy else sigmoid

    threshold = 0.5
    if not legacy:
        meta_path = os.path.join(SAVE_DIR, "stage1_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                threshold = float(json.load(f).get("threshold", 0.5))

    pred = (p_fracture >= threshold).astype(int)
    y_int = y_true.astype(int)
    results = {
        "model": os.path.basename(model_path),
        "legacy": legacy,
        "threshold": threshold,
        "n_test": int(len(y_int)),
        "accuracy": float((pred == y_int).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, pred)),
        "precision_fracture": float(precision_score(y_int, pred, zero_division=0)),
        "recall_fracture": float(recall_score(y_int, pred, zero_division=0)),
        "f1_fracture": float(f1_score(y_int, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_int, p_fracture)),
        "confusion_matrix_rows_true_no-frac_frac": confusion_matrix(y_int, pred).tolist(),
    }

    for k, v in results.items():
        print(f"  {k}: {v}")
    out = os.path.join(SAVE_DIR, f"eval_stage1_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved {out}")
    return results


def evaluate_stage2(model_path, tag):
    from sklearn.metrics import classification_report, confusion_matrix

    legacy = model_path.endswith(".h5")
    print(f"\n=== Stage 2 evaluation: {os.path.basename(model_path)} "
          f"({'legacy' if legacy else 'new'}) ===")
    model = load_any(model_path)

    ds, class_names = load_test(STAGE2_TEST, label_mode="categorical")
    probs, labels = predict_all(model, ds, legacy)
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)

    top1 = float((y_pred == y_true).mean())
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3 = float(np.mean([t in row for t, row in zip(y_true, top3_idx)]))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    results = {
        "model": os.path.basename(model_path),
        "legacy": legacy,
        "n_test": int(len(y_true)),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"  top1_accuracy: {top1:.4f}")
    print(f"  top3_accuracy: {top3:.4f}")
    print(f"  macro_f1: {results['macro_f1']:.4f}")
    out = os.path.join(SAVE_DIR, f"eval_stage2_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved {out}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=None, help="path to stage-1 model")
    ap.add_argument("--stage2", default=None, help="path to stage-2 model")
    ap.add_argument("--only", choices=["1", "2"], default=None)
    ap.add_argument("--tag", default="eval", help="label for output json files")
    args = ap.parse_args()

    if args.only in (None, "1"):
        evaluate_stage1(args.stage1 or default_model_path(1), args.tag)
    if args.only in (None, "2"):
        evaluate_stage2(args.stage2 or default_model_path(2), args.tag)


if __name__ == "__main__":
    main()
