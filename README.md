# 🦴 Bone Fracture Detection Using CNN

An end-to-end deep-learning web application that detects bone fractures from X-ray (radiograph) images using a **two-stage CNN pipeline**, classifies the fracture type, estimates severity, explains predictions with Grad-CAM heatmaps, and generates downloadable PDF diagnostic reports — wrapped in a role-based clinical portal (Patient / Doctor / Admin).

Released under the [MIT License](LICENSE).

---

## 📌 System Pipeline

| Stage | Model | Task |
|---|---|---|
| **Stage 1** | MobileNetV2 (transfer learning) | Binary: *Fracture* vs *No Fracture*, with a validation-tuned decision threshold |
| **Stage 2** | MobileNetV2 (transfer learning) | 12-class fracture-type classification (Avulsion, Comminuted, Compression, Dislocation, Greenstick, Hairline, Impacted, Intra-articular, Longitudinal, Oblique, Pathological, Spiral) |

**Pipeline:** Upload X-ray → Stage 1 estimates P(fracture) → if above threshold, Stage 2 ranks the fracture types (top-3 shown) → severity (Mild/Moderate/Severe) + care recommendation → Grad-CAM heatmap of model attention → PDF report → record saved; a doctor can review, annotate, approve/reject, and the PDF is regenerated with their diagnosis.

**Tech stack:** Flask · TensorFlow 2.20 / Keras 3 · OpenCV · SQLite · ReportLab · Chart.js · Vanilla JS/CSS

## 🚀 Quick Start (Windows)

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

Then open http://127.0.0.1:5000.

**Accounts & roles**

| Role | How to get one | Access |
|---|---|---|
| Patient | Public sign-up at `/register` | Upload scans, own records, request specialist review, facility locator |
| Doctor | `/register-clinical` with the clinical key (`CLINICAL_KEY` env var, default `DOC-VVIT-2026`) | Review queue, full history, approve/reject AI findings |
| Admin | `/register-clinical` with the admin key (`ADMIN_KEY` env var, default `ADM-VVIT-2026`) | Everything + `/admin` analytics dashboard |

**Configuration (environment variables)** — `SECRET_KEY`, `CLINICAL_KEY`, `ADMIN_KEY`, `FLASK_DEBUG` (set `0` in production). Development fallbacks exist so it runs out of the box.

**Not included in the repository** (gitignored): the training `dataset/` and its
`.dataset_backups/`, the legacy `.h5` models, `database.db` (created automatically on
first run), and `static/uploads/`. The retrained `.keras` models and their metadata ARE
included, so the app predicts out of the box — the dataset is only needed for retraining.

## 🗂️ Project Structure

```
├── app.py                      # Flask app: auth, uploads, dashboards, admin analytics
├── run.bat                     # Launcher (uses the short-path venv)
├── database.db                 # SQLite (users, scans, doctor reviews)
├── requirements.txt            # Pinned, known-good versions
├── model/
│   ├── cnn_model.py            # MobileNetV2 architectures (flat graph -> Grad-CAM friendly)
│   ├── data_utils.py           # tf.data pipelines, augmentation, class weights
│   ├── train_model_stage1.py   # Stage-1 trainer (class weights, threshold tuning)
│   ├── train_model_stage2.py   # Stage-2 trainer (partial unfreeze, label smoothing)
│   ├── predict.py              # Inference + Grad-CAM + PDF generation
│   ├── evaluate_models.py      # Test-set evaluation for any saved model
│   ├── compare_models.py       # Old-vs-new comparison (full + leak-free test sets)
│   ├── sanitize_dataset.py     # One-time repair of corrupt images
│   ├── dedupe_train_test.py    # One-time removal of train/test duplicates
│   └── saved_model/            # *.keras models + *_meta.json (history, metrics, threshold)
├── dataset/                    # stage1 (binary) & stage2 (12-class) train/test folders
├── templates/                  # Jinja2 pages, all extending base.html
└── static/                     # css/style.css, js/app.js, images/, uploads/
```

Retraining: `cd model` then `python train_model_stage2.py` and `python train_model_stage1.py`
(use the short-path venv's python). Each writes a `.keras` model plus a `_meta.json` with real
training history, test metrics, and (stage 1) the tuned decision threshold — the web app reads
these for its charts and the model-info page.

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

| Metric (deployment pipeline) | Old (scratch CNN) | **New (MobileNetV2)** |
|---|---|---|
| Detection rate on **unseen** fractures | 95.3% (missed 15/317) | **99.4% (missed 2/317)** |
| Full test-set accuracy | 94.4% ¹ | 88.7% ² |
| ROC AUC (full test) | 0.987 ¹ | 0.978 |
| Decision threshold | fixed 0.5 | 0.32, tuned on validation |
| Model file size | 128 MB | **22 MB** |

¹ *Inflated: the old model trained on 277 of these 594 test images (47%).*
² *Honest: the new model never saw any test image; it deliberately trades some
no-fracture precision for near-perfect fracture recall — the clinically safer error.*

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

The new Stage 2 improved from 30.7% → 38.6% top-1 across training iterations
(head training → 80-layer fine-tune → flip-TTA); validation accuracy plateaus at ~40%,
which is this dataset's practical ceiling (~1,150 training images for 12 fine-grained
classes). The honest path to further gains is **more data per class**, not more epochs.

Regenerate anytime with: `python model/compare_models.py`
(writes `model/saved_model/comparison_report.json`).

### What the new training pipeline does differently

| Area | Change |
|---|---|
| Architecture | Both stages: MobileNetV2 (ImageNet) grafted as a **flat graph** with preprocessing baked in — Grad-CAM reaches real conv layers, saved model is ~4× smaller than the old 128 MB scratch CNN |
| Imbalance | Class weights (Stage 1: fracture errors cost ~6.3× more; Stage 2: per-class balancing) |
| Validation | Proper 15% split carved from **train**; test set touched exactly once |
| Training | Two-phase: frozen-backbone head training → fine-tune top 40 layers with **BatchNorm frozen**, EarlyStopping (best-weights restore) + ReduceLROnPlateau |
| Stage 1 threshold | Tuned on validation for balanced accuracy instead of a blind 0.5 |
| Stage 2 regularization | Label smoothing 0.1 + X-ray-appropriate augmentation (no vertical flips) |
| Honesty | Real history/metrics saved to `*_meta.json` and displayed in-app; fake chart removed |

---

# 🛠️ Everything Else That Was Fixed / Improved

### Backend (Flask)
* **Upload security**: JPG/PNG whitelist, real-image verification (PIL), 15 MB cap, and UUID-prefixed filenames — previously any file type was accepted (stored-XSS risk via `/static/uploads`) and same-named uploads silently **overwrote other patients' images**, corrupting old records.
* **Secrets** (`SECRET_KEY`, clinical/admin keys) moved to environment variables.
* `/learn` and `/model-info` used to crash (templates never existed) — both pages now exist (fracture-type education + an honest model card).
* The orphaned `admin` role now has a real **`/admin` analytics dashboard** (scan volume, fracture-type distribution, review turnaround, user counts).
* Grad-CAM previously **overwrote the original upload** for `.png`/`.jpeg` files (`str.replace(".jpg", …)` no-op) — fixed with proper path handling; heatmap honestly labeled "Model Attention".
* Flash messages are now actually rendered (errors used to vanish); login throttling (5 attempts / 10 min); minimum password length; structured logging; error pages (404/413/500).
* PDF reports include the care recommendation, top-3 probabilities, record number, medical disclaimer — and are **regenerated with the doctor's diagnosis** after review.

### Database (SQLite)
* Foreign keys ON + `REFERENCES users(id)`; indexes on `user_id` and `review_status`.
* Idempotent column migrations (replaces blind try/except ALTERs).
* New audit columns: `reviewed_by`, `reviewed_at`, plus `gradcam_path`, `pdf_path`, `fracture_prob`.
* All templates/queries use **named column access** — positional `case[10]` indexing is gone.

### Frontend / UI & UX
* One **`base.html`** replaces ~1,700 lines of copy-pasted nav/theme markup across 9 pages; nav is role-aware from a single place and **responsive** (hamburger on mobile).
* `login.html` had two `<body>` tags; fake "Sign in with Google" and dead "Forgot Password" stubs removed, as was `script.js` (dead code targeting elements that no longer existed).
* Results page shows **honest per-scan data**: stage-1 fracture probability vs threshold, top-3 fracture-type probability bars, and real training curves clearly labeled as training history.
* Upload zone: working **drag & drop**, client-side type/size validation, truthful copy (JPG/PNG 15 MB — the old UI promised unsupported DICOM).
* Patient dashboard now links each record's Grad-CAM and PDF; doctor queue sorts patient-requested reviews first and shows the audit trail.
* Accessibility: aria-labels on icon buttons, visible focus rings, severity shown as text+color, `aria-live` flash region. Favicon added; theme applies before first paint (no flash).
* Facility locator: working "use my location" (geolocation + reverse-geocode) replacing the old stub.

### Project / DevOps
* The committed `venv/` pointed at another machine's Python (`C:\Users\tanma\…`) — removed; see Quick Start for the working setup (short path is mandatory for TensorFlow on Windows).
* `requirements.txt` fully pinned; `tf_keras` included only for loading the legacy `.h5` models.
* Medical disclaimer on every page footer, the model card, and PDFs.

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](LICENSE).

## ⚠️ Disclaimer

This is an **academic project** built for learning purposes. It is **not a medical device**,
has not undergone clinical validation, and must never be used for actual diagnosis.
Always consult qualified medical professionals.
