# Training Guide — data, pretraining, detector

Everything the shipped models were trained with is reproducible from the
scripts in `model/`. This guide covers the optional Phase-3 upgrades:
external datasets, radiograph-pretrained backbones, and the wrist-fracture
localization detector. **None of these are required to run the app** — the
pipeline degrades gracefully (no detector weights → localization is skipped).

Training environment:

```bash
pip install -r requirements.txt -r requirements-train.txt
```

## The one non-negotiable rule: dedupe before training

This project already paid for train/test leakage once — 47% of the Stage-1
test set had byte-identical twins in the training folders (see the
leakage investigation in the README and `model/dedupe_train_test.py`).

Every external image therefore passes a content-hash gate
(`model/external_data.py`) against **both stages' test sets** before it may
enter any training folder, and external sets are deduped against each other.
The `prepare_*` scripts below apply the gate automatically — never copy
images into `dataset/` by hand.

## External datasets

| Dataset | Size / labels | Used for | Access |
|---|---|---|---|
| **MURA** (Stanford) | ~40k upper-extremity radiographs, normal/abnormal per study | Backbone **pretraining** only — "abnormal" ≠ "fracture", never feed it into Stage 1 directly | Requires signing Stanford's research-use agreement: <https://stanfordmlgroup.github.io/competitions/mura/> |
| **FracAtlas** | ~4k X-rays, fracture/non-fracture (+ masks for the fractured subset) | Direct **Stage 1** training data | CC-BY 4.0: <https://doi.org/10.6084/m9.figshare.22363012> |
| **GRAZPEDWRI-DX** | ~20k pediatric wrist radiographs with bounding boxes | **Detector** (Stage 1.5) | CC-BY 4.0: <https://doi.org/10.6084/m9.figshare.14825193> |
| RSNA challenge sets | Mostly CT / other tasks | Skipped — no plain-radiograph fracture set fits | — |

Check each license/registration term before downloading; cite the datasets
in any publication.

## 1. Radiograph-pretrained backbone (MURA)

```bash
cd model
python pretrain_backbone.py <path-to-mura-download>      # writes saved_model/backbone_radiograph.*
```

Two-phase schedule (frozen head training → top-40-layer fine-tuning with
BatchNorm frozen), identical layer names to the stage models so the weights
file initializes both. A SimCLR-style self-supervised variant over
MURA + GRAZPEDWRI-DX is the documented stretch option if the supervised
proxy plateaus.

## 2. More Stage-1 data (FracAtlas)

```bash
cd model
python prepare_fracatlas.py <path-to-fracatlas-download> --dry-run   # inspect
python prepare_fracatlas.py <path-to-fracatlas-download>             # merge (gate applied)
```

## 3. Retrain both stages

```bash
cd model
python train_model_stage1.py --init-weights saved_model/backbone_radiograph.weights.h5
python train_model_stage2.py --init-weights saved_model/backbone_radiograph.weights.h5

# Calibration + OOD statistics are PER-MODEL artifacts — always re-run:
python calibrate.py
python build_ood_stats.py

# Honest evaluation on the same leak-free test sets:
python compare_models.py
```

The meta JSONs record `init_weights` and `external_images`, so every model
documents exactly which data it saw. Report old vs new results side by side
(the README results tables get a "radiograph-pretrained" column) with the
same confidence-interval caveats already used there — including when the
result is a wash.

## 4. Fracture localization detector (GRAZPEDWRI-DX)

**Step 1 — build the patient-level split.** The figshare distribution keeps
its YOLO labels under `folder_structure/yolov5/labels/`, apart from the
images, so point the prep script at each explicitly:

```bash
cd model/detector
python prepare_grazpedwri.py <graz-root> \
    --images-dir <graz-root>/images \
    --labels-dir <graz-root>/folder_structure/yolov5/labels \
    --out <graz-root>/yolo_fracture
```

(If your mirror already has `<graz-root>/images` + `<graz-root>/labels/`,
just run `python prepare_grazpedwri.py <graz-root>` with no overrides.)
This writes `yolo_fracture/{images,labels}/{train,val,test}/` + `dataset.yaml`.

**Step 2 — train.** GPU strongly recommended (YOLO on CPU is impractical —
days for 100 epochs over ~15k images). Two routes:

- **Local GPU:** `python train_detector.py <graz-root>/yolo_fracture/dataset.yaml`
- **Free Colab GPU (recommended if you have no GPU):** zip `yolo_fracture/`
  to Google Drive and run [`docs/colab_train_detector.ipynb`](colab_train_detector.ipynb)
  — it mirrors `train_detector.py`'s hyperparameters exactly and downloads
  `detector.pt` + `detector_meta.json` for you to drop into
  `model/detector/weights/`.

- The split is **patient-level** — one patient's views never straddle
  train/val/test. This is the detection version of the leakage lesson.
- Single `fracture` class to start; the other annotated classes (metal,
  cast, periosteal reaction…) are a later multi-class experiment.
- Augmentation excludes vertical flips, consistent with the classifiers.
- Output: `model/detector/weights/detector.pt` + `detector_meta.json`
  (mAP@50, per-split counts, confidence threshold). The web pipeline
  detects the weights automatically and adds a `localizing` step; a nano
  model adds well under a second on CPU.

**Domain honesty:** the detector is validated on *pediatric wrist*
radiographs only. The UI, PDF and model card all label it
"Localization (beta — validated on wrist X-rays)", and boxes are advisory
evidence — never a diagnosis.

### Licensing note (Ultralytics)

Ultralytics YOLO is **AGPL-3.0**. This MIT-licensed project uses it as an
optional runtime dependency (in `requirements-train.txt`, not the Docker
image by default); if you deploy with the detector enabled, your service
must comply with AGPL terms. If that's unacceptable, YOLOX (Apache-2.0) is
the documented fallback — `infer_detector.py` is the only integration point
you'd need to swap.
