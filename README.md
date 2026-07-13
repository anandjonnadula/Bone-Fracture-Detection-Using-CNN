# 🦴 Bone Fracture Detection Using CNN

[![CI](https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN/actions/workflows/ci.yml)

An end-to-end deep-learning web application that detects bone fractures from X-ray (radiograph) images using a **calibrated two-stage CNN pipeline**, classifies the fracture type, refuses non-X-ray inputs, explains predictions with an interactive Grad-CAM viewer, and generates PDF diagnostic reports — wrapped in a role-based clinical portal (Patient / Doctor / Admin) with asynchronous processing, doctor annotation tools, and real DICOM support.

Released under the [MIT License](LICENSE). **Not a medical device** — see the disclaimer below.

👉 Start with the [accuracy investigation](#-the-accuracy-investigation--what-was-actually-wrong) — the train/test-leakage forensics that shaped every honesty rule in this project.

---

## 📌 System Pipeline

| Step | Component | What it does |
|---|---|---|
| **OOD gate** | Color pre-filter + k-NN on backbone embeddings | Refuses selfies/documents/photos instead of scoring them (`model/ood_gate.py`) |
| **Stage 1** | MobileNetV2 (transfer learning) | Binary *fracture / no-fracture*, temperature-calibrated probability, validation-tuned threshold, **abstention band** that auto-routes uncertain scans to the doctor queue |
| **Stage 2** | MobileNetV2 (transfer learning) | 12-class fracture-type classification (top-3 shown; abstains to "type unclear" below 45% calibrated top-1) |
| **Stage 1.5** *(optional)* | YOLO detector (GRAZPEDWRI-DX) | Bounding box around the fracture — beta, wrist-validated; auto-enabled once weights are trained ([docs/TRAINING.md](docs/TRAINING.md)) |
| **Explainability** | Grad-CAM | Transparent heatmap layer over the original in an interactive viewer (opacity / zoom / pan) |
| **Report** | ReportLab | PDF with calibrated probability, tier wording, threshold + abstention band, doctor's diagnosis and annotations |

Uploads return instantly; a live progress stepper (Preprocessing → Image check → Screening → Typing → Explainability → Report) polls the async job. Every scan receives a **verdict** — `fracture | no_fracture | uncertain | rejected` — and a **plain-language confidence tier** used consistently across the results page, dashboards, doctor queue and PDF (one function in `model/predict.py`, never duplicated in templates).

**Tech stack:** Flask · TensorFlow 2.20 / Keras 3 · OpenCV · SQLite · ReportLab · pydicom · Chart.js (vendored) · Vanilla JS/CSS · Docker · gunicorn

## 🚀 Quick Start

### Docker (any OS — recommended)

```bash
docker compose up
```

Open <http://localhost:5000>. Data persists in named volumes (`db-data`, `uploads`) across restarts. This also permanently sidesteps the Windows 260-character path problem below.

### Bare metal (Windows)

> ⚠️ **The virtualenv must live at a SHORT path** (e.g. `%USERPROFILE%\.venvs\bfd`).
> Installing TensorFlow into a venv *inside* this deeply-nested project folder exceeds
> the Windows 260-character path limit and silently produces a broken install.

```bat
:: 1. Create the environment (once)
python -m venv %USERPROFILE%\.venvs\bfd
%USERPROFILE%\.venvs\bfd\Scripts\pip install -r requirements.txt

:: 2. Launch the app
run.bat                       :: or: %USERPROFILE%\.venvs\bfd\Scripts\python app.py
```

**Accounts & roles**

| Role | How to get one | Access |
|---|---|---|
| Patient | Public sign-up at `/register` | Upload scans, own records, request specialist review, facility locator |
| Doctor | `/register-clinical` with the clinical key (`CLINICAL_KEY` env var, default `DOC-VVIT-2026`) | Review queue, annotation tools, full history, approve/reject AI findings |
| Admin | `/register-clinical` with the admin key (`ADMIN_KEY` env var, default `ADM-VVIT-2026`) | Everything + `/admin` analytics + model card |

**Configuration (environment variables)** — `SECRET_KEY`, `CLINICAL_KEY`, `ADMIN_KEY`, `DATABASE_PATH`, `MEDIA_DIR`, `PORT`, `FLASK_DEBUG` (set `0` in production), `SECURE_COOKIES=1` behind HTTPS, `DEMO_MODE=1` for public demos ([docs/DEPLOY.md](docs/DEPLOY.md)), `PRELOAD_MODELS=1` to warm models at boot. Development fallbacks exist so it runs out of the box.

**Public demo:** `DEMO_MODE=1` seeds demo accounts (shown on the login page), adds a *"try a sample X-ray"* gallery — including one deliberately non-X-ray image that demos the OOD gate — disables clinical registration, enforces 3 uploads/min + 5 MB caps, and wipes data every 24 h. Deploying to Hugging Face Spaces (free) is a `git push`: see [docs/DEPLOY.md](docs/DEPLOY.md).

## 🔒 Security model

* **Private medical images.** Uploads live in a non-public `media/` directory served only through the authorizing `/media/<file>` route (owner or doctor/admin) — a one-time migration moved legacy files out of the publicly-served `static/uploads/` and rewrote DB paths.
* **CSRF protection** on every POST (Flask-WTF); JSON APIs take the `X-CSRFToken` header.
* **Security headers** including a nonce-based Content-Security-Policy with no CDN dependencies (Chart.js and the Inter font are vendored; the only remote hosts are the facility-locator's map/directory services).
* **Rate limiting** (Flask-Limiter): 10 uploads/min (3 in demo), 5 registrations/min; plus the existing per-username login throttle. HttpOnly/SameSite session cookies, `SECURE_COOKIES=1` behind HTTPS.
* **DICOM without PHI.** `.dcm` files are validated by parsing (Modality CR/DX/DR only), converted to PNG with correct VOI-LUT windowing and MONOCHROME1 inversion, and **the original file is deleted** — only whitelisted non-identifying tags (body part, view, modality, bit depth) are kept.

## 🧪 Tests & CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest -m "not model"   # fast suite: auth, uploads, review flow, jobs API, DICOM, calibration routing
pytest -m model         # smoke tests with the real TensorFlow models (committed, so CI runs them too)
```

GitHub Actions runs lint + both suites + a Docker image build on every push.

## 🗂️ Project Structure

```
├── app.py                      # Flask app: auth, security, uploads, media, dashboards, APIs
├── db.py                       # SQLite access + idempotent migrations (incl. media move)
├── jobs.py                     # Async pipeline: ThreadPoolExecutor + jobs table
├── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
├── demo/seed_demo.py           # DEMO_MODE account seeding
├── docs/TRAINING.md            # External datasets, backbone pretraining, detector (licenses!)
├── docs/DEPLOY.md              # Hugging Face Spaces deployment
├── model/
│   ├── cnn_model.py            # MobileNetV2 architectures (flat graph -> Grad-CAM friendly)
│   ├── data_utils.py           # tf.data pipelines, augmentation, class weights
│   ├── train_model_stage{1,2}.py  # Trainers (+ --init-weights for the radiograph backbone)
│   ├── predict.py              # Pipeline: OOD gate -> calibrated stages -> CAM; tier wording
│   ├── report.py               # PDF generation + server-side annotation/box rendering
│   ├── calibrate.py            # Temperature scaling + abstention band (per-model, re-run after training)
│   ├── ood_gate.py / build_ood_stats.py   # Out-of-distribution gate + its artifact builder
│   ├── dicom_utils.py          # DICOM validation/conversion, no-PHI policy
│   ├── external_data.py / prepare_fracatlas.py   # Hash-gated external data admission
│   ├── pretrain_backbone.py    # MURA domain-adaptive pretraining
│   ├── detector/               # GRAZPEDWRI-DX wrist-fracture detector (prepare/train/infer)
│   ├── evaluate_models.py / compare_models.py    # Honest evaluation on leak-free test sets
│   ├── sanitize_dataset.py / dedupe_train_test.py  # One-time dataset repairs
│   └── saved_model/            # *.keras models, *_meta.json (incl. calibration), ood_stats.npz
├── tests/                      # pytest suite + fixtures (see Tests & CI)
├── templates/                  # Jinja2 pages (async stepper, interactive viewer, annotation UI)
└── static/                     # css/, js/ (viewer, annotate, processing), vendored chart.js + fonts, samples/
```

Retraining: see [docs/TRAINING.md](docs/TRAINING.md). After **any** retrain, re-run
`calibrate.py` and `build_ood_stats.py` — calibration and OOD statistics are per-model artifacts.

---

# 🔬 The Accuracy Investigation — What Was Actually Wrong

Before retraining, the dataset and evaluation itself had to be fixed. Three findings:

### 1. Train/test leakage (the big one)
**277 of the 594 Stage-1 test images (47%) were byte-identical copies of training images.**
Stage 2 had 20 of 355 leaked. The original models were literally tested on images they had
memorized, so their reported accuracy was inflated. Fixed by `dedupe_train_test.py`:
duplicated **train** files were quarantined to `.dataset_backups/leaked_train_duplicates/`
(the test set stayed untouched), and the leaked-test lists were saved so any model can also
be scored on the **leak-free subset** — the only fair basis for old-vs-new comparison.
The same content-hash gate now guards every external dataset (`model/external_data.py`).

### 2. Corrupt images crashed strict decoders
12 truncated/mis-encoded images (masked by PIL's tolerant mode in the old code) were repaired
by `sanitize_dataset.py`; originals are backed up in `.dataset_backups/`.

### 3. Methodology bugs in the original training
* **validated on the test set** (model selection leaked the test data),
* **no class weighting** despite a 1:6.6 fracture/no-fracture imbalance,
* Stage 2 fine-tuned **the whole backbone including BatchNorm** on ~1.3k images,
* the app displayed **hard-coded fake accuracy curves** to users.

## 📊 Results — Before vs After Retraining

*Old = original `.h5` models (trained with the leaked duplicates). New = retrained `.keras`
models (trained on deduplicated data). "Leak-free test" = the test images that never had
copies in any training folder — the honest number.*

### Stage 1 — Fracture Detector

*The "leak-free" subset here is 317 fracture images. A crucial detail: **all 268 no-fracture
test images had byte-identical copies in the old model's training data**, so the old model's
specificity was never honestly measurable — while its fracture-detection rate on genuinely
unseen images is shown below.*

| Metric (full 594-image test) | Old (scratch .h5) | ImageNet MobileNetV2 | **Radiograph-pretrained + FracAtlas (deployed)** |
|---|---|---|---|
| Balanced accuracy | 94.3% ¹ | 87.0% | **90.4%** |
| Fracture recall | 95.4% ¹ | 99.4% | 98.8% |
| Fracture precision | — | 82.7% | **87.0%** |
| ROC AUC | 0.987 ¹ | 0.972 | **0.976** |
| Decision threshold | fixed 0.5 | 0.32 (tuned) | 0.36 (tuned) |
| Model file size | 128 MB | 22 MB | 22 MB |

The **deployed** Stage 1 uses a MobileNetV2 backbone domain-adaptively pretrained on
**MURA** (~37k radiographs, normal/abnormal, val AUC 0.83) then trained on the
deduplicated fracture data **plus 3,968 hash-gated FracAtlas images** — see
[docs/TRAINING.md](docs/TRAINING.md). Versus the ImageNet backbone it lifts balanced
accuracy +3.4 pts and precision +4.3 pts at a slightly higher (more selective) threshold.
On the **leak-free** subset (317 fracture images never seen in training) its unseen-fracture
recall is 96.9% vs the ImageNet model's 99.4%: it trades a little raw recall for precision
and overall balance — and the calibrated **abstention band** routes the borderline cases to
a doctor rather than silently deciding. ¹ *old `.h5` numbers are inflated — it trained on 277
of these 594 test images (47%).*

### Stage 2 — Fracture-Type Classifier (12 classes)

| Metric (leak-free test, n=335) | Old (full-unfreeze MobileNetV2) | **New (this pipeline)** |
|---|---|---|
| Top-1 accuracy (with deployed flip-TTA) | 41.2% ³ | 38.5% |
| Top-1 accuracy (single view) | 40.6% ³ | 37.6% |
| Top-3 accuracy (with TTA) | 64.8% | 64.2% |

³ *Upward-biased: the old model used **this test set as its training-time validation** —
early stopping selected its weights against these exact images for ~80 epochs. The ±5-point
95% confidence interval at n=335 makes the two models statistically indistinguishable;
only the new one's number was earned without ever touching the test set.*

The deployed Stage 2 (ImageNet backbone) reaches 38.6% top-1; validation accuracy plateaus
at ~40%, this dataset's practical ceiling (~1,150 training images for 12 fine-grained classes).

**A tried-and-rejected experiment (documented honestly):** we also trained Stage 2 from the
MURA radiograph backbone — the same backbone that *helped* Stage 1. It **regressed** Stage 2
to 31.6% top-1 (leak-free) / 29.6% (full). MURA's binary *abnormal-vs-normal* pretraining
washes out the fine texture cues needed to tell 12 fracture **types** apart — the
"abnormal ≠ fracture" risk the plan called out. So the shipped models are **best-of-both**:
the radiograph-pretrained Stage 1 (a clear win) with the ImageNet Stage 2 (better for
fine-grained typing). The honest path to a better Stage 2 is **more labeled type data**, not a
different backbone. Full old-vs-new-vs-radiograph numbers live in `comparison_report.json`.

Regenerate anytime with: `python model/compare_models.py`
(writes `model/saved_model/comparison_report.json`).

### Calibration & abstention (Phase 1 upgrade)

Raw network confidences are not probabilities. Both stages are **temperature-scaled**
on their validation splits (`model/calibrate.py`), and the numbers live in the meta files:

| | Temperature T | Val ECE before → after |
|---|---|---|
| Stage 1 (radiograph-pretrained) | 1.02 | 0.095 → 0.096 |
| Stage 2 (ImageNet) | 1.44 | 0.133 → 0.100 |

The deployed Stage 1 is already near-calibrated (T≈1.0), so temperature scaling barely
moves it; the **abstention band** does the heavy lifting. Stage 1 abstains inside the
calibrated band **[26.3%, 49.3%]** around its calibrated threshold (36.3%): on validation
~10% of scans fall in the band, where forced decisions are only **50% accurate** (a
coin-flip) vs **89% outside** — exactly the scans auto-flagged to a doctor. The OOD gate's
threshold (99.5th percentile of the 9,021-image training set's embedding distances = 0.553)
rejects all synthetic negatives (blank/noise/gradient images score 0.48–0.64).

### What the new training pipeline does differently

| Area | Change |
|---|---|
| Architecture | Both stages: MobileNetV2 (ImageNet) grafted as a **flat graph** with preprocessing baked in — Grad-CAM reaches real conv layers, saved model is ~4× smaller than the old 128 MB scratch CNN |
| Imbalance | Class weights (Stage 1: fracture errors cost ~6.3× more; Stage 2: per-class balancing) |
| Validation | Proper 15% split carved from **train**; test set touched exactly once |
| Training | Two-phase: frozen-backbone head training → fine-tune top 40 layers with **BatchNorm frozen**, EarlyStopping (best-weights restore) + ReduceLROnPlateau |
| Stage 1 threshold | Tuned on validation for balanced accuracy instead of a blind 0.5 |
| Stage 2 regularization | Label smoothing 0.1 + X-ray-appropriate augmentation (no vertical flips) |
| Honesty | Real history/metrics/calibration saved to `*_meta.json` and displayed in-app; fake chart removed |

---

# 🛠️ Application Features (beyond the models)

### Prediction trust
* **Verdicts + tiers everywhere**: the same plain-language wording ("Fracture detected — high confidence" … "No fracture detected — high confidence") on results, dashboards, doctor queue and PDF, all derived from one function.
* **Probability bar with a threshold marker and shaded abstention band** so the number is interpretable at a glance; uncertain scans say explicitly that a doctor has been asked to review.
* **OOD gate**: a photo upload never reaches Stage 1 — the user gets a friendly "this doesn't appear to be a bone X-ray" message and rejections are logged.

### UX
* **Async inference**: upload returns instantly; a live stepper shows pipeline progress; refreshing is harmless; failures and rejections get clear copy.
* **Interactive Grad-CAM viewer**: transparent heatmap layer over the *original* radiograph with an opacity slider (keyboard: `M`), wheel/pinch zoom, drag pan, double-click reset; legacy records fall back to the merged image.
* **Dark reading mode**: scan pages default to a deep-dark reading palette (radiographs read best against dark); a "☀ Lights" toggle persists per user; explicit global theme choice always wins.
* **Doctor annotations**: arrow / ellipse / freehand / label tools drawn in image space (they survive zoom/pan), stored as normalized vector JSON, flattened **server-side** into the regenerated PDF, and visible to the patient after review.

### Platform
* Role-based portal (Patient / Doctor / Admin) with audit trail (`reviewed_by` / `reviewed_at`), admin analytics (scan volume, verdict mix incl. uncertain/rejected, fracture-type distribution), honest model card at `/model-info` (now including calibration, ECE and OOD-gate parameters), fracture-type education page, facility locator.
* Everything above sits on the hardened base from earlier iterations: upload validation, UUID filenames, idempotent migrations, named column access, structured logging, error pages, login throttling.

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](LICENSE).
Optional training-time dependency note: Ultralytics YOLO (fracture localization) is AGPL-3.0 — see [docs/TRAINING.md](docs/TRAINING.md).

## ⚠️ Disclaimer

This is an **academic project** built for learning purposes. It is **not a medical device**,
has not undergone clinical validation, and must never be used for actual diagnosis.
Always consult qualified medical professionals. DICOM uploads are converted immediately and
no patient-identifying data is ever stored.
