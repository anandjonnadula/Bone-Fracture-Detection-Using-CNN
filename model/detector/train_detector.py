"""Train the wrist-fracture detector (YOLO) on prepared GRAZPEDWRI-DX data.

Requires `pip install -r requirements-train.txt` (Ultralytics — AGPL-3.0;
licensing note in docs/TRAINING.md). Training a nano model at 640px takes a
few GPU-hours (or a long CPU weekend); the exported weights + meta land in
model/detector/weights/ and the runtime pipeline picks them up
automatically — no code change needed.

Augmentation policy matches the classifiers: NO vertical flips (X-rays are
roughly upright); horizontal flips are anatomically valid.

Usage:  python train_detector.py <dataset.yaml> [--model yolo11n.pt]
                                 [--epochs 100] [--imgsz 640]
"""

import argparse
import json
import os
import shutil
from datetime import UTC, datetime

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "detector.pt")
META_PATH = os.path.join(WEIGHTS_DIR, "detector_meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_yaml", help="dataset.yaml from prepare_grazpedwri.py")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Base model (nano is enough for CPU inference)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=None,
                    help="Confidence threshold to record (default: tuned on val)")
    args = ap.parse_args()

    from ultralytics import YOLO  # deferred: heavy, train-only dependency

    model = YOLO(args.model)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=1337,
        flipud=0.0,   # never flip radiographs vertically
        fliplr=0.5,
        degrees=5.0,
        translate=0.08,
        scale=0.12,
        project=os.path.join(os.path.dirname(WEIGHTS_DIR), "runs"),
        name="grazpedwri_fracture",
    )

    best = os.path.join(results.save_dir, "weights", "best.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    shutil.copyfile(best, WEIGHTS_PATH)

    # Validate the exported weights and record honest metrics.
    val = YOLO(WEIGHTS_PATH).val(data=args.data_yaml, split="test")
    conf = args.conf if args.conf is not None else 0.25

    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "base_model": args.model,
        "dataset": "GRAZPEDWRI-DX (fracture class only, patient-level split)",
        "domain": "pediatric wrist radiographs — beta outside wrist studies",
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "confidence_threshold": conf,
        "test_metrics": {
            "mAP50": float(val.box.map50),
            "mAP50_95": float(val.box.map),
            "precision": float(val.box.mp),
            "recall": float(val.box.mr),
        },
        "license_note": "Ultralytics YOLO is AGPL-3.0 (see docs/TRAINING.md).",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Weights -> {WEIGHTS_PATH}")
    print(f"[OK] Meta    -> {META_PATH}")
    print("The web pipeline now runs localization automatically "
          "(install ultralytics in the serving environment too).")


if __name__ == "__main__":
    main()
