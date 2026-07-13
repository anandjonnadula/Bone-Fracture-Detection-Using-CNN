# 🚀 Upgrade Plan — Bone Fracture Detection Using CNN

This document specifies every modification to be made to the project, in a dependency-aware
order, ending with the live demo. Each item lists: **goal → files touched → implementation
steps → definition of done**. Code snippets are sketches to guide implementation, not
drop-in files.

---

## Recommended Order (Phases)

| Phase | Items | Why this order |
|---|---|---|
| **0 — Foundation** | Docker · Tests + CI · Security hardening | Everything after this lands on a reproducible, tested, secure base |
| **1 — Prediction trust** | Calibration & abstention · Plain-language confidence tiers · OOD gate | No retraining needed; changes how results are *interpreted and shown* |
| **2 — UX** | Async inference · Interactive Grad-CAM · Dark reading mode · Doctor annotations | Builds on the new result pipeline from Phase 1 |
| **3 — Data & ML** | DICOM support · External datasets / backbone pretraining · GRAZPEDWRI-DX localization | Longest phase; benefits from CI + Docker being in place |
| **4 — Ship** | Live demo deployment | Final step, once everything above is merged |

Cross-cutting rule for all phases: **every new column/table goes through the idempotent
migration mechanism**, and **every new dataset goes through the same hash-based dedupe
discipline** (`dedupe_train_test.py`) that fixed the original leakage.

---

# Phase 0 — Foundation

## 0.1 Docker

**Goal:** one-command, OS-independent run. Permanently eliminates the Windows 260-char
path problem and becomes the deployment unit for the live demo.

**Files:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `requirements.txt` (one swap), `README.md`.

**Steps**

1. Swap `opencv-python` → `opencv-python-headless` in `requirements.txt` (no GUI libs
   needed server-side; avoids installing `libGL` in the image).
2. Create the image:

   ```dockerfile
   FROM python:3.11-slim

   ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt gunicorn

   COPY . .

   ENV FLASK_DEBUG=0
   EXPOSE 5000
   # 1 worker (models load once), threads for concurrent requests
   CMD ["gunicorn", "-w", "1", "--threads", "4", "-t", "180", "-b", "0.0.0.0:5000", "app:app"]
   ```

3. `.dockerignore`: `dataset/`, `.dataset_backups/`, `venv*/`, `*.h5`, `database.db`,
   `static/uploads/`, `.git/`, `__pycache__/`. Keep `model/saved_model/*.keras` +
   `*_meta.json` **in** the image (only ~22 MB).
4. `docker-compose.yml` with named volumes for persistence:

   ```yaml
   services:
     app:
       build: .
       ports: ["5000:5000"]
       environment:
         - SECRET_KEY=${SECRET_KEY}
         - CLINICAL_KEY=${CLINICAL_KEY}
         - ADMIN_KEY=${ADMIN_KEY}
       volumes:
         - db-data:/app/data          # move database.db into /app/data (see note)
         - uploads:/app/media         # see Security 0.3 — uploads leave static/
   volumes:
     db-data:
     uploads:
   ```

5. In `app.py`, read the DB path and upload dir from env vars
   (`DATABASE_PATH`, `MEDIA_DIR`) with the current locations as defaults, so the same
   code runs bare-metal and in Docker.
6. Ensure models are loaded **once at import time** (module-level), since gunicorn uses
   `-w 1 --threads N`; guard inference with a `threading.Lock` if needed.

**Done when:** `docker compose up` on a clean machine serves the app at
`http://localhost:5000`, predictions work, and data survives a container restart.

---

## 0.2 Tests + CI

**Goal:** a pytest suite covering the security-critical paths plus a model smoke test,
running on every push via GitHub Actions.

**Files:** `tests/` (new), `.github/workflows/ci.yml`, `pyproject.toml` (ruff config), `requirements-dev.txt`.

**Steps**

1. `tests/conftest.py`: app fixture with a **temporary SQLite DB** and temp media dir;
   helper fixtures `patient_client`, `doctor_client`, `admin_client` (registered + logged in).
2. Test modules:
   - `test_auth.py` — register/login/logout, clinical key required for doctor/admin,
     wrong key rejected, login throttling kicks in after 5 attempts, role-gated routes
     return 403/redirect for the wrong role.
   - `test_upload_validation.py` — rejects non-image bytes with an image extension,
     rejects >15 MB, rejects disallowed extensions, accepts a valid PNG/JPG,
     stored filename is UUID-prefixed, second upload never overwrites the first.
   - `test_review_flow.py` — patient uploads → appears in doctor queue → doctor
     approves with diagnosis → status/audit columns set → PDF regenerated.
   - `test_model_smoke.py` (marked `@pytest.mark.model`) — run the real pipeline on
     2 fixture images (one fracture, one normal) from `tests/fixtures/`; assert it
     returns a probability in [0,1], a verdict, and writes a Grad-CAM + PDF.
   - Later phases add `test_ood_gate.py`, `test_dicom.py`, `test_jobs_api.py`.
3. CI workflow:

   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11", cache: pip }
         - run: pip install -r requirements.txt -r requirements-dev.txt
         - run: ruff check .
         - run: pytest -m "not model"        # fast suite
         - run: pytest -m model              # smoke test (models are committed, so this works)
     docker:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: docker build -t bfd .
   ```

4. Add the CI badge to `README.md`.

**Done when:** CI is green on `main`; a deliberately broken upload check makes it red.

---

## 0.3 Security hardening

**Goal:** close the remaining web-app gaps: CSRF, exposed patient images, missing
headers, unlimited request rates.

**Files:** `app.py`, all templates with forms, `requirements.txt` (+`Flask-WTF`, `Flask-Limiter`), migration.

**Steps**

1. **CSRF** — enable `CSRFProtect(app)` (Flask-WTF); add
   `<input type="hidden" name="csrf_token" value="{{ csrf_token() }}">` to every POST
   form; include the token as a header for any JS `fetch` POSTs (upload, annotations,
   job polling won't need it for GETs).
2. **Private media (the big one)** — X-rays currently live under `/static/uploads`,
   which is publicly served: anyone with the URL sees a patient's scan.
   - Move uploads to a non-static dir (`MEDIA_DIR`, e.g. `./media` / the Docker volume).
   - Serve through an authorizing route:

     ```python
     @app.route("/media/<path:filename>")
     @login_required
     def media(filename):
         scan = scan_for_file(filename)          # look up by stored path
         if scan is None or not can_view(g.user, scan):   # owner, or doctor/admin
             abort(404)
         return send_from_directory(MEDIA_DIR, filename)
     ```
   - Update all templates/PDF paths; write a one-time migration that moves existing
     files from `static/uploads/` and rewrites DB paths.
3. **Headers** — one `@app.after_request` hook (or Flask-Talisman):
   `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
   `Referrer-Policy: same-origin`, and a `Content-Security-Policy` that allows only
   `'self'` plus the Chart.js CDN you actually use (better: vendor Chart.js into
   `static/js/` and drop the CDN so CSP is pure `'self'`).
4. **Session cookies** — `SESSION_COOKIE_HTTPONLY=True`, `SESSION_COOKIE_SAMESITE="Lax"`,
   `SESSION_COOKIE_SECURE=True` when behind HTTPS (env-controlled).
5. **Rate limiting** — Flask-Limiter (in-memory storage is fine, single instance):
   e.g. `10/minute` on `/upload`, `5/minute` on `/register*`, keep the existing login
   throttle or migrate it to the limiter.
6. Verify password hashing uses `werkzeug.security` (or `argon2-cffi`) — upgrade if not.

**Done when:** an unauthenticated request to a scan image URL returns 404/403; every
POST without a CSRF token fails; headers visible in devtools; tests from 0.2 cover it.

---

# Phase 1 — Prediction trust

## 1.1 Calibration & abstention

**Goal:** make the probabilities *mean something* (temperature scaling), and give the
model an explicit "I'm not sure" output that auto-routes to the doctor queue.

**Files:** `model/calibrate.py` (new), `model/predict.py`, `model/train_model_stage*.py`
(save val logits), `*_meta.json` schema, `app.py`, migration (`verdict` column on scans).

**Steps**

1. **Temperature scaling (no retraining needed).** New script `model/calibrate.py`:
   - Recreate the validation split (same seed as training), collect raw logits
     (add a helper that returns pre-sigmoid/pre-softmax outputs; if the saved model
     only outputs probabilities, invert with `logit = log(p/(1-p))`).
   - Fit a single scalar `T` minimizing NLL on validation:

     ```python
     from scipy.optimize import minimize_scalar
     res = minimize_scalar(lambda t: nll(logits / t, labels), bounds=(0.5, 5.0), method="bounded")
     ```
   - Write `calibration: {temperature: T, method: "temperature_scaling", val_ece_before, val_ece_after}`
     into the model's `_meta.json`. Do this for both stages.
2. **Abstention band (Stage 1).** Define an uncertainty margin around the tuned
   threshold, chosen on validation (e.g. the band where balanced accuracy of a forced
   decision drops below ~85%; a pragmatic start: calibrated `p ∈ [thr − 0.10, thr + 0.13]`).
   Store `abstain_low` / `abstain_high` in `_meta.json`.
3. **Pipeline behavior** in `predict.py`:
   - `p < abstain_low` → verdict `no_fracture`
   - `p > abstain_high` → verdict `fracture` → run Stage 2
   - otherwise → verdict `uncertain` → **still run Stage 2** (informational), set
     `review_status = 'requested'` automatically, and mark the record.
4. **Stage 2 abstention:** if the calibrated top-1 probability < e.g. 0.45, label the
   type as "unclear — top-3 candidates shown" rather than asserting a class.
5. Add `verdict` (and calibrated `fracture_prob`) to the scans table via the idempotent
   migration; PDF and results page consume the verdict.
6. Report ECE / reliability-diagram numbers in the model-info page (you already show
   honest metrics — calibration belongs there).

**Done when:** every scan has a verdict of `fracture | no_fracture | uncertain`;
uncertain scans appear pre-flagged in the doctor queue; `_meta.json` carries `T` and
band; unit test asserts the three routing branches.

---

## 1.2 Plain-language confidence tiers

**Goal:** translate calibrated probabilities into wording a patient understands,
consistently across the results page, dashboard, doctor queue, and PDF.

**Files:** `model/predict.py` (single source of truth), `templates/` (results, dashboards), `static/css/style.css`, PDF generation.

**Steps**

1. One function, used everywhere (never duplicate the mapping in templates):

   ```python
   def confidence_tier(p, meta):
       thr, lo, hi = meta["threshold"], meta["abstain_low"], meta["abstain_high"]
       if p >= max(hi, 0.85): return ("fracture_high",   "Fracture detected — high confidence")
       if p >  hi:            return ("fracture_likely", "Fracture likely")
       if p >= lo:            return ("uncertain",       "Uncertain — specialist review recommended (this scan has been flagged automatically)")
       if p >  0.10:          return ("clear_moderate",  "No fracture detected — moderate confidence")
       return                  ("clear_high",            "No fracture detected — high confidence")
   ```

2. UI: tier shown as **text + color + icon** (never color alone — you already follow
   this for severity); the probability bar gets a **threshold marker** and a shaded
   **abstention band** so the number is interpretable at a glance.
3. Every tier string ends near the existing medical disclaimer; the `uncertain` tier
   explicitly says a doctor has been asked to review.
4. Stage 2 top-3 bars get the same treatment: if abstaining, header reads
   "Fracture type unclear — most likely candidates:".

**Done when:** the same tier wording appears on results page, patient dashboard row,
doctor queue, and PDF, all derived from the one function.

---

## 1.3 Out-of-distribution gate (before Stage 1)

**Goal:** refuse to score inputs that aren't bone X-rays (selfies, documents, cats),
instead of emitting a confident fracture probability for them.

**Files:** `model/ood_gate.py` (new), `model/build_ood_stats.py` (new, one-time),
`model/saved_model/ood_stats.npz` (new artifact), `model/predict.py`, `app.py`, results template, tests.

**Approach:** two cheap layers — a heuristic pre-filter plus a feature-space distance
check on the MobileNetV2 embeddings you already compute. No new labels needed.

**Steps**

1. **Heuristic pre-filter** (microseconds): radiographs are effectively monochrome.
   Compute per-channel correlation / mean saturation; if the image is strongly colored
   (e.g. mean HSV saturation > ~0.25), reject immediately with reason `"color_image"`.
2. **Embedding gate:**
   - `build_ood_stats.py` (run once, like your other one-time scripts): pass every
     **training** image through the Stage-1 backbone's global-average-pooling layer,
     L2-normalize embeddings, store the mean vector + either (a) mean/covariance for a
     Mahalanobis distance or (b) the full matrix for a k-NN cosine distance
     (k≈10). Save to `saved_model/ood_stats.npz`.
   - Pick the acceptance threshold as e.g. the 99.5th percentile of training-set
     distances, then sanity-check it against ~100 obvious negatives (photos from any
     personal folder / a COCO sample) — negatives should score far outside. Record the
     threshold + percentile in the npz/meta.
3. **Pipeline:** `predict.py` runs `ood_gate.check(img)` before Stage 1. On rejection,
   no probabilities are computed or stored; the scan record (if kept at all) gets
   status `rejected_not_xray`.
4. **UI:** friendly, non-accusatory message — "This doesn't appear to be a bone X-ray
   (radiograph). Please upload a plain X-ray image in JPG/PNG format." Log rejections
   (useful demo-abuse telemetry later).
5. **Tests:** fixture set of ~5 non-X-ray images must be rejected; the 2 X-ray fixtures
   must pass. Also assert a **blank/black image** is rejected (distance will be large).

**Done when:** a photo upload never reaches Stage 1, the user sees the friendly
message, and the fixtures test locks the behavior in.

---

# Phase 2 — UX

## 2.1 Async inference with visible progress

**Goal:** upload returns instantly; the browser shows a live stepper
(Preprocessing → Fracture screening → Type classification → Explainability → Report)
instead of a frozen page. Also unlocks the localization stage (Phase 3) without
worsening latency.

**Design choice:** no Celery/Redis — a `ThreadPoolExecutor` plus a `jobs` table is the
right size for a single-instance SQLite app, and it survives your gunicorn
`-w 1 --threads N` model.

**Files:** `app.py` (or new `jobs.py`), migration (jobs table), `templates/results.html`
→ split into `processing.html` + results partial, `static/js/app.js`, tests.

**Steps**

1. **Schema:**

   ```sql
   CREATE TABLE IF NOT EXISTS jobs (
     id         TEXT PRIMARY KEY,           -- uuid4
     scan_id    INTEGER REFERENCES scans(id),
     user_id    INTEGER REFERENCES users(id),
     status     TEXT NOT NULL DEFAULT 'queued',
       -- queued | preprocessing | ood_check | stage1 | stage2 | explaining | reporting | done | failed | rejected
     error      TEXT,
     created_at TEXT DEFAULT CURRENT_TIMESTAMP,
     updated_at TEXT
   );
   ```

2. **Worker:** module-level `ThreadPoolExecutor(max_workers=2)`. The upload route
   validates + saves the file, inserts scan + job rows, submits
   `run_pipeline(job_id)`, and redirects to `/processing/<job_id>`. `run_pipeline`
   updates `jobs.status` at each stage boundary and wraps everything in
   try/except → `failed` + logged traceback. Each thread opens its **own** SQLite
   connection (never share connections across threads).
3. **API:** `GET /api/jobs/<id>` → `{status, error, scan_id}` (owner/doctor/admin only).
4. **Frontend:** `/processing/<job_id>` renders the stepper skeleton; JS polls every
   1 s (give up after ~3 min → show retry). On `done` redirect to the results page;
   on `rejected` show the OOD message; on `failed` show the error page copy.
5. **Ordering with OOD gate:** the gate is just the second step of the pipeline —
   users see "Checking image type…" before "Screening for fracture…".

**Done when:** a slow prediction never blocks the HTTP worker; refreshing the
processing page is harmless; `test_jobs_api.py` covers happy path + authz on the
polling endpoint.

---

## 2.2 Interactive Grad-CAM

**Goal:** replace the static baked-in heatmap with an overlay the user controls —
opacity slider, zoom, pan — over the *original* radiograph.

**Files:** `model/predict.py`, `templates/results.html` (+ doctor review page), `static/js/viewer.js` (new), `static/css/style.css`.

**Steps**

1. **Backend — export the heatmap as a transparent layer**, not a merged JPEG:

   ```python
   cam = normalize(cam_raw)                              # [0,1], resized to image size
   rgb  = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)[..., ::-1]
   alpha = (np.clip(cam, 0, 1) * 255 * 0.85).astype("uint8")   # activation → opacity
   rgba = np.dstack([rgb, alpha])
   Image.fromarray(rgba).save(heatmap_png_path)          # *_cam.png, PNG keeps alpha
   ```

   Keep generating the merged image too — the PDF still needs a flat picture.
2. **Frontend viewer** (`viewer.js`, vanilla, ~100 lines):
   - Container with two stacked `<img>` layers (original below, `*_cam.png` above,
     `pointer-events: none` on the overlay).
   - **Opacity slider** (`<input type="range">`) drives the overlay's `style.opacity`;
     an "M" keyboard shortcut / button toggles 0 ↔ last value.
   - **Zoom/pan:** wheel zoom toward cursor + drag to pan by updating one
     `transform: translate(x,y) scale(s)` on an inner wrapper; double-click resets.
     (If you'd rather not hand-roll it, vendor the tiny `@panzoom/panzoom` lib into
     `static/js/` — no CDN, per the CSP decision in 0.3.)
   - Touch: pinch-zoom + one-finger pan (pointer events make this ~20 extra lines).
3. Keep the honest label ("Model Attention — not a clinical annotation") pinned to the
   viewer chrome, and add it to the doctor review page too (same component).
4. Accessibility: slider gets a visible label + `aria-valuetext` ("Heatmap 40% visible").

**Done when:** results and review pages show the draggable/zoomable viewer; the PDF is
unchanged; old records with only merged images still render (fallback when `*_cam.png`
is absent).

---

## 2.3 Dark reading mode by default on scan pages

**Goal:** radiographs read best against dark surroundings — scan-viewing pages
(results, doctor review) default to a deep-dark theme while respecting an explicit
user choice.

**Files:** `templates/base.html`, results + review templates, `static/css/style.css`, the pre-paint theme script.

**Steps**

1. You already apply the theme before first paint — extend that script with a
   **page-level default**: templates set `<body data-theme-default="dark">` on scan
   pages; the script resolves `localStorage user choice ▸ page default ▸ global default`.
2. Add a **"reading" palette** on top of your dark theme for the viewer region:
   near-black canvas around the image (e.g. `#0b0e11`), reduced-chrome cards, dimmed
   non-essential UI, slightly desaturated accent colors so the heatmap's colors stay
   the most saturated thing on screen.
3. A small "☀ Lights" toggle near the viewer flips just that page (persisted per user).
4. Re-check contrast ratios (WCAG AA) for tier badges and the probability bars on the
   dark palette; verify no first-paint flash on these pages.

**Done when:** opening any results/review page shows dark by default, a user's explicit
light choice sticks, and Grad-CAM + charts remain legible.

---

## 2.4 Doctor annotation tools

**Goal:** doctors mark up the radiograph (arrow, ellipse, freehand, short label) during
review; annotations are stored as vectors, rendered onto the regenerated PDF, and
visible to the patient after approval.

**Files:** doctor review template, `static/js/annotate.js` (new), `app.py` (save/load endpoints), migration (`scan_annotations` table), PDF generation in `predict.py`/report module.

**Steps**

1. **Schema (vector-first — re-renderable at any resolution, editable later):**

   ```sql
   CREATE TABLE IF NOT EXISTS scan_annotations (
     id         INTEGER PRIMARY KEY,
     scan_id    INTEGER NOT NULL REFERENCES scans(id),
     doctor_id  INTEGER NOT NULL REFERENCES users(id),
     data       TEXT NOT NULL,      -- JSON, see below
     created_at TEXT DEFAULT CURRENT_TIMESTAMP
   );
   ```

   ```json
   { "image_w": 1024, "image_h": 812,
     "shapes": [
       {"type":"ellipse","cx":0.42,"cy":0.31,"rx":0.06,"ry":0.04,"color":"#ff5252","width":3},
       {"type":"arrow","x1":0.60,"y1":0.55,"x2":0.45,"y2":0.34,"color":"#ff5252","width":3},
       {"type":"path","points":[[0.1,0.2],[0.12,0.22]],"color":"#ffd740","width":2},
       {"type":"label","x":0.62,"y":0.57,"text":"hairline, distal radius","color":"#ff5252"}
     ]}
   ```

   Coordinates normalized 0–1 → resolution-independent.
2. **Canvas editor** (`annotate.js`): a `<canvas>` layered inside the *same viewer* from
   2.2 (annotations must survive zoom/pan — draw in image space, apply the shared
   transform). Toolbar: select tool, arrow, ellipse, freehand, text, color (2–3 preset
   colors is enough), undo, clear, **Save**. Save POSTs the JSON (CSRF header) to
   `POST /api/scans/<id>/annotations` (doctor/admin only).
3. **Server-side flattening:** on review submission, render the vectors onto a copy of
   the original with Pillow (`ImageDraw` — ellipse/line/polygon + a bundled TTF for
   labels) → `*_annotated.png`, embed **that** image in the regenerated PDF alongside
   the doctor's diagnosis. Don't trust a client-exported PNG; the server render is the
   canonical one.
4. **Patient view:** after approval, the results page viewer gains an "annotations"
   layer toggle (render the same JSON on a read-only canvas).
5. Audit: annotations are additive per review; `reviewed_by/reviewed_at` you already
   have completes the trail.

**Done when:** a doctor can circle a fracture, save, approve — and both the patient's
results page and the regenerated PDF show the marked-up image.

---

# Phase 3 — Data & ML

## 3.1 Real DICOM support

**Goal:** accept `.dcm` uploads (the format X-rays actually leave machines in), convert
correctly to 8-bit for the pipeline, keep useful metadata, and store **no PHI**.

**Files:** `model/dicom_utils.py` (new), `app.py` upload route, client-side validation in `static/js/app.js`, migration (`body_part`, `view_position`, `source_format` columns), `requirements.txt` (+`pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg` — for compressed transfer syntaxes), tests + a couple of sample `.dcm` fixtures (pydicom ships test files you can use).

**Steps**

1. **Validation:** extend the whitelist with `.dcm`; verify by parsing, not extension:

   ```python
   ds = pydicom.dcmread(path)          # raises on non-DICOM
   if getattr(ds, "Modality", "") not in {"CR", "DX", "DR"}:
       reject("Only plain radiographs (CR/DX) are supported.")
   ```

   Raise the size cap for DICOM only (e.g. 50 MB).
2. **Correct conversion** (`dicom_utils.to_png`) — the two classic pitfalls are VOI LUT
   and MONOCHROME1:

   ```python
   from pydicom.pixel_data_handlers.util import apply_voi_lut
   arr = apply_voi_lut(ds.pixel_array, ds).astype("float32")
   if ds.PhotometricInterpretation == "MONOCHROME1":
       arr = arr.max() - arr                       # invert: bone should be bright
   arr -= arr.min(); arr /= max(arr.ptp(), 1e-6)
   png = (arr * 255).astype("uint8")
   ```

3. **PHI policy:** convert immediately, keep **only the PNG** plus whitelisted
   non-identifying tags (`BodyPartExamined`, `ViewPosition`, bit depth, modality) in new
   DB columns; **delete the original `.dcm` after conversion** (or, if you want to keep
   it, anonymize with pydicom by blanking the patient module tags first). Document this
   in the model card / disclaimer.
4. The rest of the pipeline is untouched — OOD gate, Stage 1, etc. all consume the PNG.
   Show body part / view on the results page and PDF when present.
5. Update upload-zone copy ("JPG / PNG / DICOM (.dcm)"), client-side checks, and the
   README (which once falsely promised DICOM — now it's real; say so).

**Done when:** a compressed `.dcm` (JPEG-Lossless transfer syntax) converts correctly,
a MONOCHROME1 fixture comes out non-inverted, non-radiograph DICOM is rejected with a
clear message, and no PHI persists anywhere.

---

## 3.2 More data + radiograph-pretrained backbone

**Goal:** attack the real bottleneck you identified (~1,150 Stage-2 training images).
Two levers: (a) a backbone pretrained on *radiographs* instead of ImageNet-only, and
(b) more labeled images where labels transfer directly.

**Datasets** (check each one's license/registration terms before use — MURA in
particular requires signing Stanford's research-use agreement):

| Dataset | Size / labels | Use here |
|---|---|---|
| **MURA** (Stanford) | ~40k upper-extremity radiographs, normal/abnormal per study | Backbone **pretraining** (labels are "abnormal", not "fracture" — don't feed into Stage 1 directly) |
| **FracAtlas** | ~4k X-rays, fracture/non-fracture (+ localization masks for the fractured subset) | Direct **Stage 1** training data; masks useful for 3.3 sanity checks |
| **GRAZPEDWRI-DX** | ~20k pediatric wrist radiographs with bounding boxes | Detector training (3.3); images also usable for pretraining |
| RSNA challenge sets | Mostly CT / other tasks | Optional; skip unless a plain-radiograph fracture set fits |

**Files:** `model/pretrain_backbone.py` (new), `model/data_utils.py` (multi-source loaders), `train_model_stage*.py` (accept `--init-weights`), docs.

**Steps**

1. **Dedupe discipline first:** run every external image through the same content-hash
   index as `dedupe_train_test.py` against your **test sets** before it may enter any
   training folder. This is non-negotiable — it's the mistake this project already
   paid for once. Also dedupe external sets against *each other* (they overlap in the wild).
2. **Domain-adaptive pretraining** (`pretrain_backbone.py`):
   - MobileNetV2 (ImageNet init) + small head, trained on **MURA normal-vs-abnormal**
     (study-level labels propagated to images), standard two-phase schedule (frozen →
     top-layers unfrozen, BatchNorm frozen — reuse your existing recipe).
   - Save `saved_model/backbone_radiograph.keras` + meta. Optional stretch: replace the
     supervised proxy with SimCLR-style self-supervised pretraining over MURA +
     GRAZPEDWRI-DX images combined — more work, use only if the supervised version
     plateaus.
3. **Stage 1:** retrain with `--init-weights backbone_radiograph.keras` and FracAtlas
   merged into training (it maps 1:1 onto fracture/no-fracture). Recompute class
   weights; re-tune the threshold and (from 1.1) re-fit temperature + abstention band —
   calibration is per-model.
4. **Stage 2:** retrain from the radiograph backbone. Expect the bigger relative gain
   here — that's the head that was starved. Keep label smoothing + your augmentation
   policy.
5. **Honest evaluation, unchanged:** score on the same leak-free test sets via
   `compare_models.py`; add a third column ("radiograph-pretrained") to the README
   results tables. Report deltas with the same CI caveats you already use.

**Done when:** the comparison report shows old vs current vs radiograph-pretrained on
identical leak-free test sets, and the meta files record exactly which external data
each model saw.

---

## 3.3 Fracture localization — GRAZPEDWRI-DX detector

**Goal:** a Stage 1.5 that draws a **bounding box around the fracture** — stronger
evidence than Grad-CAM and a genuine step past Stage 2's classification ceiling.

**Files:** `model/detector/` (new: `prepare_grazpedwri.py`, `train_detector.py`, `infer_detector.py`, weights), `model/predict.py` (pipeline integration), viewer (boxes layer), PDF, migration (`detections` JSON column or table), `requirements.txt`.

**Steps**

1. **Framework choice:** Ultralytics YOLO (v8/11, `n` or `s` size) is the pragmatic
   pick — GRAZPEDWRI-DX is distributed with YOLO-format labels and training is a few
   lines. Note Ultralytics is **AGPL-3.0**; that's compatible with an open academic
   project but document it (your MIT code + AGPL dependency at runtime). If that's
   unacceptable, YOLOX (Apache-2.0) is the fallback.
2. **Data prep** (`prepare_grazpedwri.py`): download from the official source, build a
   **patient-level** train/val/test split (never split by image — one patient's views
   must not straddle splits; this is the detection version of your leakage lesson).
   Start with a single `fracture` class; the other annotated classes (metal, cast,
   periosteal reaction…) can be a later multi-class experiment.
3. **Train** at 640 px, standard augmentation minus vertical flips (consistent with
   your Stage 2 policy); export best weights + a small meta JSON (mAP@50, per-split
   counts, confidence threshold chosen on val).
4. **Pipeline integration** (`predict.py`): after Stage 1 fires (or verdict is
   `uncertain`), run the detector; store boxes as normalized JSON
   `[{"x":…, "y":…, "w":…, "h":…, "conf":…}]`.
   - Boxes found → new "Detected region(s)" layer in the 2.2 viewer (toggle, like
     annotations) + drawn on the PDF image.
   - No boxes but Stage 1 positive → show that disagreement honestly ("screening
     positive; no focal region localized — common outside wrist studies").
5. **Scope honesty (important):** the detector is trained on **pediatric wrist**
   radiographs. Label the layer "Localization (beta — validated on wrist X-rays)" in
   the UI, model card, and PDF. If DICOM metadata (3.1) says the body part isn't
   wrist/forearm, either skip the detector or keep the caveat prominent.
6. Latency lands inside the async pipeline (2.1) as a new `localizing` step; a nano
   model on CPU adds well under a second.

**Done when:** a wrist-fracture fixture produces a sensible box end-to-end (viewer +
PDF), the model card documents the domain limitation, and detector metrics live in its
meta JSON.

---

# Phase 4 — Live demo (final step)

**Goal:** a public, safe, zero-cost demo. Target: **Hugging Face Spaces (Docker Space)**
— free CPU tier, and your image from 0.1 deploys as-is. (Render/Railway free tiers work
too but sleep aggressively.)

**Files:** `demo/seed_demo.py` (new), `app.py` (`DEMO_MODE` behaviors), `static/samples/` (bundled sample X-rays), Space config (`README.md` front-matter with `sdk: docker`, port 5000 → HF expects 7860, so parameterize the port via `PORT` env), docs.

**Steps**

1. **`DEMO_MODE=1` env flag** switches on:
   - **Seeded accounts** shown right on the login page:
     `demo-patient / demo-doctor / demo-admin` (random passwords printed on the page —
     it's a demo). Clinical/admin **registration keys disabled** in demo mode so nobody
     mints real roles.
   - **Sample gallery:** an "Or try a sample X-ray" strip on the upload page (4–6
     bundled radiographs: clear fracture, subtle fracture, normal, one deliberately
     OOD image to show the gate working). One click runs the full pipeline — this is
     what most visitors will actually do.
   - **Banner on every page:** "Public demo — do not upload real patient images. Data
     is wiped periodically."
   - Aggressive limits: uploads 3/minute/IP (Flask-Limiter from 0.3), 5 MB cap,
     nightly wipe of demo DB + media (a tiny scheduler thread, or just rely on Space
     restarts — storage is ephemeral on the free tier, which is a feature here).
2. **Space setup:** push the repo (or a `deploy` branch) to a Docker Space; set
   `SECRET_KEY` + `DEMO_MODE=1` as Space secrets/variables; make the container listen
   on `$PORT` (default 7860 on Spaces).
3. **Resource check:** with the 22 MB Keras models + YOLO-nano, TF-CPU fits the free
   tier; keep gunicorn at `-w 1 --threads 2` there. First-request model load is hidden
   by the async pipeline's "queued" state.
4. **Polish for visitors:** README gets the live-demo link, a 20-second GIF
   (upload → progress stepper → interactive Grad-CAM + boxes → PDF), an architecture
   diagram, and the CI badge. The leakage-investigation section is your differentiator —
   link it from the top.

**Done when:** a stranger with the link can, in under a minute: click a sample →
watch the progress stepper → play with the Grad-CAM slider → open the PDF → log into
the demo-doctor account and annotate + approve that scan.

---

## Appendix A — New dependencies

| Package | For |
|---|---|
| `gunicorn` | Docker/production server (0.1) |
| `pytest`, `ruff` (dev) | Tests + CI (0.2) |
| `Flask-WTF`, `Flask-Limiter` | CSRF + rate limiting (0.3) |
| `scipy` | Temperature fitting (1.1) — likely already a transitive dep |
| `pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg` | DICOM (3.1) |
| `ultralytics` (or `yolox`) | Detector (3.3) |
| `opencv-python-headless` | replaces `opencv-python` (0.1) |

## Appendix B — Schema changes (all via the idempotent migration)

- `scans`: `verdict`, `body_part`, `view_position`, `source_format`, `detections` (JSON)
- New tables: `jobs` (2.1), `scan_annotations` (2.4)
- Path rewrite migration when uploads move out of `static/` (0.3)

## Appendix C — Per-phase exit checklist

- [ ] **P0** `docker compose up` works · CI green · media served only through the auth route · CSRF on all forms
- [ ] **P1** verdicts + tiers everywhere · uncertain auto-flags review · non-X-rays rejected · `_meta.json` has T + band
- [ ] **P2** async stepper live · Grad-CAM slider/zoom on results & review · dark default on scan pages · doctor markup lands in PDF
- [ ] **P3** `.dcm` end-to-end with no PHI stored · radiograph-pretrained models beat baseline on leak-free tests (documented either way) · wrist detector boxed in UI + PDF with domain caveat
- [ ] **P4** live demo link in README · sample gallery · demo limits + wipe · GIF + diagram + badges

---

*Keep the project's defining habit throughout: every claim in the README stays backed by
a script anyone can rerun (`compare_models.py`, `calibrate.py`, detector meta), and the
medical disclaimer travels with every new surface (viewer, annotations, demo banner).*
