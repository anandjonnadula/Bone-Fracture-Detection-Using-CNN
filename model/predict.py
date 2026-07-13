"""Inference pipeline: OOD gate → Stage 1 (calibrated) → Stage 2 → localization → Grad-CAM.

Key behaviors:
  * Probabilities are temperature-calibrated (see calibrate.py); every scan
    gets a verdict of fracture | no_fracture | uncertain based on an
    abstention band chosen on validation. Uncertain scans are auto-routed
    to the doctor queue by the app layer.
  * confidence_tier() is the single source of truth for the plain-language
    wording shown on the results page, dashboards, doctor queue and PDF —
    never duplicate the mapping in templates.
  * An out-of-distribution gate runs before Stage 1 so selfies/documents/
    cats are refused instead of receiving a confident fracture probability.
  * Grad-CAM is exported twice: a merged JPEG for the PDF, and a transparent
    RGBA PNG (*_cam.png) the interactive viewer can blend over the original.
  * Supports the new .keras models (raw 0-255 input, sigmoid = P(fracture),
    tuned decision threshold) while remaining backward compatible with the
    legacy .h5 models (1/255 input, sigmoid = P(no_fracture)).
  * Models load lazily on first prediction and inference is guarded by a
    lock (the app serves with 1 process + N threads).
"""

import json
import math
import os
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

try:  # imported as `model.predict` by the app, as `predict` by sibling scripts
    from model import ood_gate
except ImportError:  # pragma: no cover - script-mode fallback
    import ood_gate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
IMG_SIZE = 224

# Stage-2 abstention: below this calibrated top-1 probability the type is
# reported as "unclear — top-3 candidates shown" instead of asserting a class.
STAGE2_ABSTAIN_TOP1 = 0.45

VERDICT_FRACTURE = "fracture"
VERDICT_NO_FRACTURE = "no_fracture"
VERDICT_UNCERTAIN = "uncertain"

OOD_REJECT_MESSAGE = (
    "This doesn't appear to be a bone X-ray (radiograph). "
    "Please upload a plain X-ray image in JPG/PNG format."
)

# -----------------------------
# Class labels / severity / recommendations
# -----------------------------
DEFAULT_LABELS = [
    "avulsion", "comminuted", "compression", "dislocation", "greenstick",
    "hairline", "impacted", "intra_articular", "longitudinal", "oblique",
    "pathological", "spiral",
]

severity_dict = {
    "hairline": "Mild",
    "greenstick": "Mild",
    "avulsion": "Moderate",
    "spiral": "Moderate",
    "oblique": "Moderate",
    "compression": "Moderate",
    "impacted": "Moderate",
    "pathological": "Severe",
    "dislocation": "Severe",
    "longitudinal": "Severe",
    "intra_articular": "Severe",
    "comminuted": "Severe",
}

recommendation_dict = {
    "Mild": "Rest the affected area and avoid strain. Follow basic care and monitor symptoms.",
    "Moderate": "Consult an orthopedic specialist. Immobilization or minor treatment may be required.",
    "Severe": "Immediate medical attention is required. Possible surgery or advanced treatment needed.",
}

RECOMMENDATION_NO_FRACTURE = "No medical action required at this time based on the scan."
RECOMMENDATION_UNCERTAIN = (
    "The screening result falls in the model's uncertainty band. A specialist "
    "review has been requested automatically — please await the doctor's assessment."
)
RECOMMENDATION_TYPE_UNCLEAR = (
    "A fracture was detected but its type could not be determined confidently. "
    "Consult an orthopedic specialist for classification and treatment."
)


def recommendation_for(verdict, severity):
    """Recommendation text for a STORED scan row (results page revisits)."""
    if verdict == VERDICT_NO_FRACTURE:
        return RECOMMENDATION_NO_FRACTURE
    if verdict == VERDICT_UNCERTAIN:
        return RECOMMENDATION_UNCERTAIN
    if verdict == VERDICT_FRACTURE:
        if severity in recommendation_dict:
            return recommendation_dict[severity]
        return RECOMMENDATION_TYPE_UNCLEAR
    return None  # rejected / legacy rows


def _clean_label(folder_name):
    """'Intra-articular fracture' -> 'intra_articular', 'Fracture Dislocation' -> 'dislocation'."""
    label = folder_name.lower()
    label = label.replace(" fracture", "").replace("-crush", "").replace("fracture ", "")
    label = label.strip().replace("-", "_").replace(" ", "_")
    return label


def _load_class_labels():
    indices_path = os.path.join(BASE_DIR, "class_indices.json")
    if os.path.exists(indices_path):
        try:
            with open(indices_path) as f:
                indices = json.load(f)
            ordered = [k for k, _ in sorted(indices.items(), key=lambda kv: kv[1])]
            return [_clean_label(name) for name in ordered]
        except Exception as e:
            print("Error loading class_indices.json:", e)
    return DEFAULT_LABELS


class_labels = _load_class_labels()


def display_name(label):
    return label.replace("_", " ").title()


# -----------------------------
# Lazy model loading (new .keras preferred, legacy .h5 fallback)
# -----------------------------
_models = {}
_predict_lock = threading.Lock()
_load_lock = threading.Lock()


def _load_single(stage):
    """Returns dict {model, legacy: bool, meta: dict}."""
    keras_path = os.path.join(SAVE_DIR, f"{stage}_model.keras")
    h5_path = os.path.join(SAVE_DIR, f"{stage}_model.h5")
    meta_path = os.path.join(SAVE_DIR, f"{stage}_meta.json")

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Could not read {meta_path}:", e)

    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path)
        return {"model": model, "legacy": False, "meta": meta}

    if os.path.exists(h5_path):
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
        except Exception:
            import tf_keras  # legacy Keras 2 loader

            model = tf_keras.models.load_model(h5_path, compile=False)
        return {"model": model, "legacy": True, "meta": meta}

    raise FileNotFoundError(
        f"No model found for {stage}: expected {keras_path} or {h5_path}. "
        f"Run the training scripts in the model/ folder first."
    )


def get_model(stage):
    if stage not in _models:
        with _load_lock:
            if stage not in _models:  # double-checked: threads race on first request
                _models[stage] = _load_single(stage)
    return _models[stage]


def warm_models():
    """Load both stages eagerly (called once at app startup / first worker)."""
    get_model("stage1")
    get_model("stage2")


# -----------------------------
# Image preprocessing
# -----------------------------
def _load_image_array(img_path):
    """RGB image as float32 array (IMG_SIZE, IMG_SIZE, 3) in 0-255.

    BILINEAR resize matches the tf.data pipeline the models were trained
    with — a mismatched resize kernel costs measurable accuracy.
    """
    img = Image.open(img_path).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
    )
    return np.asarray(img, dtype=np.float32)


def _model_input(arr, legacy):
    """New models embed rescaling; legacy models trained on 1/255 input."""
    x = arr / 255.0 if legacy else arr
    return np.expand_dims(x, axis=0)


# -----------------------------
# Calibration (temperature scaling — see calibrate.py)
# -----------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p, eps=1e-7):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def calibrate_binary(p, temperature):
    """Apply temperature scaling to a sigmoid probability."""
    if not temperature or temperature == 1.0:
        return p
    return _sigmoid(_logit(p) / temperature)


def calibrate_softmax(probs, temperature):
    """Re-temper a softmax distribution (log-probs recover logits up to a constant)."""
    if not temperature or temperature == 1.0:
        return probs
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logp -= logp.max()
    exp = np.exp(logp)
    return exp / exp.sum()


def stage1_params(meta=None):
    """Decision parameters in CALIBRATED probability space.

    calibrate.py stores `threshold_calibrated` + `abstain_low/high`; when a
    model has not been calibrated yet, fall back to the raw tuned threshold
    with the pragmatic default band [thr − 0.10, thr + 0.13].
    """
    if meta is None:
        meta = get_model("stage1")["meta"]
    temperature = (meta.get("calibration") or {}).get("temperature")
    raw_thr = float(meta.get("threshold", 0.5))
    thr = float(meta.get("threshold_calibrated",
                         calibrate_binary(raw_thr, temperature) if temperature else raw_thr))
    lo = float(meta.get("abstain_low", max(thr - 0.10, 0.02)))
    hi = float(meta.get("abstain_high", min(thr + 0.13, 0.97)))
    return {"threshold": thr, "abstain_low": lo, "abstain_high": hi,
            "temperature": temperature}


def verdict_from_prob(p_cal, params):
    if p_cal < params["abstain_low"]:
        return VERDICT_NO_FRACTURE
    if p_cal > params["abstain_high"]:
        return VERDICT_FRACTURE
    return VERDICT_UNCERTAIN


# The ONE place tier wording lives — results page, dashboards, doctor queue
# and PDF all render these strings (never duplicate them in templates).
TIER_TEXTS = {
    "fracture_high": "Fracture detected — high confidence",
    "fracture_likely": "Fracture likely",
    "uncertain": ("Uncertain — specialist review recommended "
                  "(this scan has been flagged automatically)"),
    "clear_moderate": "No fracture detected — moderate confidence",
    "clear_high": "No fracture detected — high confidence",
    "rejected": "Not scored — image doesn't appear to be a bone X-ray",
}


def tier_text_for(slug):
    return TIER_TEXTS.get(slug, "")


def confidence_tier(p, meta=None):
    """Plain-language confidence tier — the ONE mapping used everywhere.

    p is the CALIBRATED fracture probability in [0, 1]. Returns (slug, text).
    """
    params = stage1_params(meta)
    lo, hi = params["abstain_low"], params["abstain_high"]
    if p >= max(hi, 0.85):
        slug = "fracture_high"
    elif p > hi:
        slug = "fracture_likely"
    elif p >= lo:
        slug = "uncertain"
    elif p > 0.10:
        slug = "clear_moderate"
    else:
        slug = "clear_high"
    return (slug, TIER_TEXTS[slug])


# -----------------------------
# OOD gate plumbing
# -----------------------------
_embedder = None


def _get_embedder():
    """Sub-model producing the Stage-1 GAP embedding (features already computed)."""
    global _embedder
    if _embedder is None:
        entry = get_model("stage1")
        model = entry["model"]
        try:
            gap_out = model.get_layer("gap").output
        except ValueError:
            gap_out = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    gap_out = layer.output
                    break
            if gap_out is None:  # last resort: pool the last conv block manually
                conv = model.get_layer(_find_last_conv_layer(model)).output
                gap_out = tf.keras.layers.GlobalAveragePooling2D()(conv)
        _embedder = {
            "model": tf.keras.Model(model.inputs, gap_out),
            "legacy": entry["legacy"],
        }
    return _embedder


def embed_image(arr):
    """Backbone GAP embedding for an image array (used by the OOD gate)."""
    emb = _get_embedder()
    with _predict_lock:
        return emb["model"].predict(
            _model_input(arr, emb["legacy"]), verbose=0
        )[0]


def check_ood(arr):
    """Run the OOD gate on a preprocessed image array."""
    return ood_gate.check(arr, embed_image)


# -----------------------------
# Grad-CAM
# -----------------------------
def _find_last_conv_layer(model):
    if "out_relu" in [layer.name for layer in model.layers]:
        return "out_relu"
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def _compute_cam(img_path, entry, class_index=None):
    """Raw normalized CAM heatmap in [0,1] at IMG_SIZE resolution, or None."""
    model, legacy = entry["model"], entry["legacy"]
    layer_name = _find_last_conv_layer(model)
    if layer_name is None:
        print("Grad-CAM: no 4D conv layer found")
        return None

    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    arr = _load_image_array(img_path)
    x = tf.convert_to_tensor(_model_input(arr, legacy))

    with _predict_lock, tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            idx = int(tf.argmax(preds[0])) if class_index is None else class_index
            loss = preds[:, idx]
        grads = tape.gradient(loss, conv_out)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.squeeze(conv_out @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    import cv2

    return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)), arr


def generate_gradcam(img_path, entry, class_index=None):
    """Export Grad-CAM as (merged_jpg_path, transparent_png_path).

    The merged JPEG feeds the PDF (needs a flat picture); the RGBA PNG is a
    transparent layer for the interactive viewer (activation → opacity).
    Never writes over the original file. Returns (None, None) on failure.
    """
    try:
        computed = _compute_cam(img_path, entry, class_index)
        if computed is None:
            return None, None
        cam, arr = computed

        import cv2

        root, _ = os.path.splitext(img_path)

        heat = np.uint8(255 * cam)
        heat_color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        # 1) Merged overlay for the PDF (honest label burned in).
        original_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.6, heat_color_bgr, 0.4, 0)
        cv2.putText(overlay, "Model Attention", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        merged_path = f"{root}_gradcam.jpg"
        cv2.imwrite(merged_path, overlay)

        # 2) Transparent RGBA layer for the interactive viewer.
        rgb = heat_color_bgr[..., ::-1]
        alpha = (np.clip(cam, 0, 1) * 255 * 0.85).astype("uint8")
        rgba = np.dstack([rgb, alpha])
        overlay_path = f"{root}_cam.png"
        Image.fromarray(rgba).save(overlay_path)

        return merged_path, overlay_path
    except Exception as e:
        print("Grad-CAM generation failed:", e)
        return None, None


# -----------------------------
# Fracture localization (Stage 1.5 — optional detector, see model/detector/)
# -----------------------------
def _detect_boxes(img_path):
    """Run the wrist-fracture detector if its weights exist; else None.

    None  = detector unavailable (weights not trained/downloaded)
    []    = detector ran and found no region
    list  = normalized boxes [{"x","y","w","h","conf"}, ...]
    """
    try:
        try:
            from model.detector.infer_detector import detect, weights_available
        except ImportError:
            from detector.infer_detector import detect, weights_available
        if not weights_available():
            return None
        return detect(img_path)
    except Exception as e:
        print("Detector inference failed:", e)
        return None


# -----------------------------
# Stage inference
# -----------------------------
def stage1_infer(arr):
    """Calibrated Stage-1 screening. Returns probabilities + verdict."""
    stage1 = get_model("stage1")
    with _predict_lock:
        raw = float(stage1["model"].predict(
            _model_input(arr, stage1["legacy"]), verbose=0)[0][0])
    if stage1["legacy"]:
        # Legacy model: sigmoid = P(no_fracture), fixed 0.5 threshold.
        p_raw = 1.0 - raw
    else:
        p_raw = raw

    params = stage1_params(stage1["meta"])
    p_cal = calibrate_binary(p_raw, params["temperature"])
    verdict = verdict_from_prob(p_cal, params)
    tier, tier_text = confidence_tier(p_cal, stage1["meta"])
    return {
        "p_raw": p_raw,
        "p_cal": p_cal,
        "params": params,
        "verdict": verdict,
        "tier": tier,
        "tier_text": tier_text,
    }


def stage2_infer(arr):
    """Calibrated Stage-2 typing with horizontal-flip TTA and abstention."""
    stage2 = get_model("stage2")
    with _predict_lock:
        probs = stage2["model"].predict(
            _model_input(arr, stage2["legacy"]), verbose=0)[0]
        flip_probs = stage2["model"].predict(
            _model_input(arr[:, ::-1, :], stage2["legacy"]), verbose=0)[0]
    probs = (probs + flip_probs) / 2.0

    temperature = (stage2["meta"].get("calibration") or {}).get("temperature")
    probs = calibrate_softmax(probs, temperature)

    order = np.argsort(probs)[::-1]
    top3 = [
        {"label": display_name(class_labels[i]), "prob": round(float(probs[i]) * 100, 2)}
        for i in order[:3]
    ]
    best = int(order[0])
    top1_prob = float(probs[best])
    type_unclear = top1_prob < STAGE2_ABSTAIN_TOP1
    return {
        "top3": top3,
        "best_index": best,
        "fracture_type": class_labels[best],
        "top1_prob": top1_prob,
        "type_unclear": type_unclear,
    }


# -----------------------------
# Full pipeline
# -----------------------------
def _noop_status(_stage):
    pass


def predict_fracture(img_path, status_cb=None, run_detector=True):
    """Run the complete pipeline on one image.

    status_cb, when given, is called with a stage name at each boundary:
    preprocessing → ood_check → stage1 → stage2 → localizing → explaining.
    Returns a result dict; if the OOD gate rejects, the dict has
    rejected=True and no probabilities.
    """
    status_cb = status_cb or _noop_status

    status_cb("preprocessing")
    arr = _load_image_array(img_path)

    status_cb("ood_check")
    ok, reason, ood_details = check_ood(arr)
    if not ok:
        return {
            "rejected": True,
            "reject_reason": reason,
            "reject_message": OOD_REJECT_MESSAGE,
            "ood": ood_details,
        }

    status_cb("stage1")
    s1 = stage1_infer(arr)
    params = s1["params"]
    verdict = s1["verdict"]

    common = {
        "rejected": False,
        "verdict": verdict,
        "fracture_prob": round(s1["p_cal"] * 100, 2),
        "fracture_prob_raw": round(s1["p_raw"] * 100, 2),
        "threshold": round(params["threshold"] * 100, 2),
        "abstain_low": round(params["abstain_low"] * 100, 2),
        "abstain_high": round(params["abstain_high"] * 100, 2),
        "tier": s1["tier"],
        "tier_text": s1["tier_text"],
        "ood": ood_details,
        "detections": None,
    }

    if verdict == VERDICT_NO_FRACTURE:
        return {
            **common,
            "result": "No Fracture Detected",
            "fracture_detected": False,
            "confidence": round((1.0 - s1["p_cal"]) * 100, 2),
            "fracture_type": None,
            "type_unclear": False,
            "top3": [],
            "gradcam": None,
            "cam_overlay": None,
            "severity": "None",
            "recommendation": RECOMMENDATION_NO_FRACTURE,
        }

    # Stage 2 runs for `fracture` and — informationally — for `uncertain`.
    status_cb("stage2")
    s2 = stage2_infer(arr)

    detections = None
    if run_detector:
        status_cb("localizing")
        detections = _detect_boxes(img_path)

    status_cb("explaining")
    stage1_entry = get_model("stage1")
    merged, cam_overlay = generate_gradcam(img_path, stage1_entry)
    if merged is None:
        merged, cam_overlay = generate_gradcam(
            img_path, get_model("stage2"), class_index=s2["best_index"]
        )

    if verdict == VERDICT_UNCERTAIN:
        return {
            **common,
            "result": "Uncertain — Specialist Review Requested",
            "fracture_detected": False,
            "confidence": round(s2["top1_prob"] * 100, 2),
            "fracture_type": None,
            "type_unclear": True,
            "top3": s2["top3"],
            "gradcam": merged,
            "cam_overlay": cam_overlay,
            "detections": detections,
            "severity": "None",
            "recommendation": RECOMMENDATION_UNCERTAIN,
        }

    if s2["type_unclear"]:
        return {
            **common,
            "result": "Fracture Detected (type unclear)",
            "fracture_detected": True,
            "confidence": round(s2["top1_prob"] * 100, 2),
            "fracture_type": None,
            "type_unclear": True,
            "top3": s2["top3"],
            "gradcam": merged,
            "cam_overlay": cam_overlay,
            "detections": detections,
            "severity": "Unknown",
            "recommendation": RECOMMENDATION_TYPE_UNCLEAR,
        }

    fracture_type = s2["fracture_type"]
    severity = severity_dict.get(fracture_type, "Unknown")
    return {
        **common,
        "result": f"Fracture Detected: {display_name(fracture_type)}",
        "fracture_detected": True,
        "confidence": round(s2["top1_prob"] * 100, 2),
        "fracture_type": display_name(fracture_type),
        "type_unclear": False,
        "top3": s2["top3"],
        "gradcam": merged,
        "cam_overlay": cam_overlay,
        "detections": detections,
        "severity": severity,
        "recommendation": recommendation_dict.get(severity, "Consult a medical professional."),
    }


# -----------------------------
# Real model performance (from training metadata)
# -----------------------------
def get_model_performance():
    """Training history + test metrics + calibration for both stages."""
    out = {}
    for stage in ("stage1", "stage2"):
        meta_path = os.path.join(SAVE_DIR, f"{stage}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    out[stage] = json.load(f)
                continue
            except Exception as e:
                print(f"Could not read {meta_path}:", e)
        out[stage] = None
    return out
