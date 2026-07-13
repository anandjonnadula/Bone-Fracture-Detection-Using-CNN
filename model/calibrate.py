"""Post-hoc temperature scaling + abstention band for both stages.

No retraining needed: recreates the validation split (same seed as
training), collects the saved models' probabilities, inverts them to
logits, and fits a single scalar temperature T minimizing NLL. Results —
temperature, ECE before/after, calibrated decision threshold, and the
abstention band — are written into stage*_meta.json, which predict.py
consumes at inference time. Calibration is per-model: re-run this script
after every retrain.

Run from the model/ directory:

    python calibrate.py
"""

import json
import os
from datetime import UTC, datetime

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from model import predict
    from model.data_utils import load_train_val, prepare
except ImportError:
    import predict
    from data_utils import load_train_val, prepare

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
STAGE1_TRAIN = os.path.join(BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "train")
STAGE2_TRAIN = os.path.join(BASE_DIR, "..", "dataset", "stage2_MultiClassification", "train")

# Pragmatic abstention band around the calibrated threshold (see UPGRADE_PLAN
# 1.1): validation balanced accuracy is ~99.7%, so a data-driven band would
# collapse to zero width — the fixed margins are the honest choice here.
BAND_BELOW = 0.10
BAND_ABOVE = 0.13

EPS = 1e-7


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece_binary(y_true, p, n_bins=15):
    """Expected calibration error over the positive-class probability."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def ece_multiclass(y_true_idx, probs, n_bins=15):
    """Top-1 confidence ECE."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        err += (mask.sum() / total) * abs(correct[mask].mean() - conf[mask].mean())
    return float(err)


def fit_temperature_binary(logits, labels):
    def nll(t):
        p = np.clip(_sigmoid(logits / t), EPS, 1 - EPS)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return float(res.x)


def fit_temperature_softmax(log_probs, labels_idx):
    def nll(t):
        z = log_probs / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -np.mean(logp[np.arange(len(labels_idx)), labels_idx])

    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return float(res.x)


def collect_val_probs(model_entry, val_ds):
    """Model probabilities + labels over a (batched) validation dataset."""
    model, legacy = model_entry["model"], model_entry["legacy"]
    probs, labels = [], []
    for batch_x, batch_y in val_ds:
        x = batch_x.numpy()
        if legacy:
            x = x / 255.0
        probs.append(model.predict(x, verbose=0))
        labels.append(batch_y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _update_meta(stage, updates):
    meta_path = os.path.join(SAVE_DIR, f"{stage}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update(updates)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Updated {meta_path}")


def calibrate_stage1():
    print("=== Stage 1: temperature scaling + abstention band ===")
    entry = predict.get_model("stage1")
    _train_ds, val_ds, _names = load_train_val(STAGE1_TRAIN, label_mode="binary")
    # Same label remap as training: sigmoid means P(fracture).
    val_ds = val_ds.map(lambda x, y: (x, 1.0 - y))
    val_ds = prepare(val_ds)

    probs, labels = collect_val_probs(entry, val_ds)
    p_raw = probs.reshape(-1)
    if entry["legacy"]:
        p_raw = 1.0 - p_raw
    y = labels.reshape(-1)

    logits = _logit(p_raw)
    temperature = fit_temperature_binary(logits, y)
    p_cal = _sigmoid(logits / temperature)

    ece_before = ece_binary(y, p_raw)
    ece_after = ece_binary(y, p_cal)
    print(f"T={temperature:.4f}  ECE before={ece_before:.4f}  after={ece_after:.4f}")

    raw_thr = float(entry["meta"].get("threshold", 0.5))
    thr_cal = float(_sigmoid(_logit(np.array([raw_thr]))[0] / temperature))
    lo = max(thr_cal - BAND_BELOW, 0.02)
    hi = min(thr_cal + BAND_ABOVE, 0.97)

    # Report what the band does on validation (forced-decision quality).
    in_band = (p_cal >= lo) & (p_cal <= hi)
    forced = (p_cal >= thr_cal).astype(float)
    correct = (forced == y)
    stats = {
        "val_abstain_fraction": float(in_band.mean()),
        "val_accuracy_outside_band": float(correct[~in_band].mean()) if (~in_band).any() else None,
        "val_accuracy_inside_band_forced": float(correct[in_band].mean()) if in_band.any() else None,
    }
    print(f"threshold(raw)={raw_thr:.3f} -> calibrated={thr_cal:.3f}  band=[{lo:.3f}, {hi:.3f}]")
    print(f"band stats: {stats}")

    _update_meta("stage1", {
        "calibration": {
            "temperature": temperature,
            "method": "temperature_scaling",
            "val_ece_before": ece_before,
            "val_ece_after": ece_after,
            "calibrated_at": datetime.now(UTC).isoformat(),
            **stats,
        },
        "threshold_calibrated": thr_cal,
        "abstain_low": lo,
        "abstain_high": hi,
    })


def calibrate_stage2():
    print("\n=== Stage 2: temperature scaling ===")
    entry = predict.get_model("stage2")
    _train_ds, val_ds, _names = load_train_val(STAGE2_TRAIN, label_mode="categorical")
    val_ds = prepare(val_ds)

    probs, labels = collect_val_probs(entry, val_ds)
    y_idx = labels.argmax(axis=1)

    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    temperature = fit_temperature_softmax(log_probs, y_idx)

    z = log_probs / temperature
    z = z - z.max(axis=1, keepdims=True)
    p_cal = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    ece_before = ece_multiclass(y_idx, probs)
    ece_after = ece_multiclass(y_idx, p_cal)
    print(f"T={temperature:.4f}  top-1 ECE before={ece_before:.4f}  after={ece_after:.4f}")

    _update_meta("stage2", {
        "calibration": {
            "temperature": temperature,
            "method": "temperature_scaling",
            "val_ece_before": ece_before,
            "val_ece_after": ece_after,
            "calibrated_at": datetime.now(UTC).isoformat(),
        },
        "abstain_top1": predict.STAGE2_ABSTAIN_TOP1,
    })


if __name__ == "__main__":
    calibrate_stage1()
    calibrate_stage2()
