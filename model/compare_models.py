"""Before/after comparison of the original (.h5) and retrained (.keras) models.

Evaluates every model on the SAME held-out test sets and reports metrics on:
  * the full test set, and
  * the leak-free subset (test images that never had byte-identical copies in
    the training folders — see dedupe_train_test.py). The original models were
    trained WITH those duplicates, so the clean subset is the only fair
    measure of their generalization.

Usage:  python compare_models.py
Writes saved_model/comparison_report.json and prints a summary table.
"""

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))

IMG_SIZE = 224
BATCH = 32


def load_any(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        import tf_keras

        return tf_keras.models.load_model(path, compile=False)


def iter_test_files(test_dir):
    """Yields (path, class_index) with classes in alphabetical folder order."""
    classes = sorted(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )
    for idx, cls in enumerate(classes):
        folder = os.path.join(test_dir, cls)
        for f in sorted(os.listdir(folder)):
            yield os.path.join(folder, f), idx


def load_batch(paths):
    """Mirrors the deployed preprocessing in predict.py (BILINEAR resize)."""
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        arrs.append(np.asarray(img, dtype=np.float32))
    return np.stack(arrs)


def predict_files(model, paths, legacy):
    probs = []
    for i in range(0, len(paths), BATCH):
        x = load_batch(paths[i:i + BATCH])
        if legacy:
            x = x / 255.0
        probs.append(model.predict_on_batch(x))
    return np.concatenate(probs, axis=0)


def leaked_set(tag):
    path = os.path.join(SAVE_DIR, f"test_leakage_{tag}.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        rel_paths = json.load(f)
    return {os.path.normpath(os.path.join(DATASET_DIR, p)) for p in rel_paths}


def stage1_metrics(y_true, p_frac, threshold):
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

    pred = (p_frac >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((pred == y_true).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "recall_fracture": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1_fracture": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_frac)), 4) if len(set(y_true)) > 1 else None,
    }


def stage2_metrics(y_true, probs):
    from sklearn.metrics import f1_score

    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    return {
        "n": int(len(y_true)),
        "top1_accuracy": round(float((y_pred == y_true).mean()), 4),
        "top3_accuracy": round(float(np.mean([t in row for t, row in zip(y_true, top3, strict=False)])), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def compare_stage1(report):
    test_dir = os.path.join(DATASET_DIR, "stage1_BinaryClassification", "test")
    files = list(iter_test_files(test_dir))
    paths = [p for p, _ in files]
    folder_labels = np.array([c for _, c in files])  # 0=fracture, 1=no_fracture
    y_true = (1 - folder_labels).astype(int)         # 1 = fracture
    leaked = leaked_set("stage1")
    clean_mask = np.array([os.path.normpath(p) not in leaked for p in paths])
    print(f"\nStage 1: {len(paths)} test images "
          f"({int(clean_mask.sum())} leak-free / {int((~clean_mask).sum())} were duplicated in train)")

    new_threshold = 0.5
    meta_path = os.path.join(SAVE_DIR, "stage1_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            new_threshold = float(json.load(f).get("threshold", 0.5))

    for label, fname, legacy, threshold in [
        ("original_h5", "stage1_model.h5", True, 0.5),
        ("retrained_keras", "stage1_model.keras", False, new_threshold),
    ]:
        path = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        model = load_any(path)
        sigm = predict_files(model, paths, legacy).reshape(-1)
        p_frac = (1.0 - sigm) if legacy else sigm
        report["stage1"][label] = {
            "threshold": threshold,
            "full_test": stage1_metrics(y_true, p_frac, threshold),
            "clean_test": stage1_metrics(y_true[clean_mask], p_frac[clean_mask], threshold),
        }
        r = report["stage1"][label]
        print(f"  {label}: full acc {r['full_test']['accuracy']:.4f} "
              f"(bal {r['full_test']['balanced_accuracy']:.4f}, AUC {r['full_test']['roc_auc']}) | "
              f"clean acc {r['clean_test']['accuracy']:.4f} "
              f"(bal {r['clean_test']['balanced_accuracy']:.4f}, AUC {r['clean_test']['roc_auc']})")


def compare_stage2(report):
    test_dir = os.path.join(DATASET_DIR, "stage2_MultiClassification", "test")
    files = list(iter_test_files(test_dir))
    paths = [p for p, _ in files]
    y_true = np.array([c for _, c in files])
    leaked = leaked_set("stage2")
    clean_mask = np.array([os.path.normpath(p) not in leaked for p in paths])
    print(f"\nStage 2: {len(paths)} test images "
          f"({int(clean_mask.sum())} leak-free / {int((~clean_mask).sum())} were duplicated in train)")

    for label, fname, legacy in [
        ("original_h5", "stage2_model.h5", True),
        ("retrained_keras", "stage2_model.keras", False),
    ]:
        path = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        model = load_any(path)
        probs = predict_files(model, paths, legacy)
        report["stage2"][label] = {
            "full_test": stage2_metrics(y_true, probs),
            "clean_test": stage2_metrics(y_true[clean_mask], probs[clean_mask]),
        }
        r = report["stage2"][label]
        print(f"  {label}: full top1 {r['full_test']['top1_accuracy']:.4f} "
              f"top3 {r['full_test']['top3_accuracy']:.4f} | "
              f"clean top1 {r['clean_test']['top1_accuracy']:.4f} "
              f"top3 {r['clean_test']['top3_accuracy']:.4f}")


def main():
    report = {"stage1": {}, "stage2": {}}
    compare_stage1(report)
    compare_stage2(report)
    out = os.path.join(SAVE_DIR, "comparison_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {out}")


if __name__ == "__main__":
    main()
