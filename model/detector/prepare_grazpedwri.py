"""Prepare GRAZPEDWRI-DX for YOLO training with a PATIENT-LEVEL split.

GRAZPEDWRI-DX (~20k pediatric wrist radiographs, CC-BY 4.0) ships with
YOLO-format labels for 9 classes. This script:

  1. builds a patient-level 80/10/10 train/val/test split — one patient's
     views must NEVER straddle splits (the detection version of this
     project's train/test-leakage lesson);
  2. keeps only the `fracture` class and remaps it to class 0 (the other
     annotated classes — metal, cast, periosteal reaction… — are a later
     multi-class experiment);
  3. emits a self-contained Ultralytics dataset:

        <out>/images/{train,val,test}/*.png   (hardlinks when possible)
        <out>/labels/{train,val,test}/*.txt   (fracture-only, class 0)
        <out>/dataset.yaml

Download from the official source (registration-free figshare record):
https://doi.org/10.6084/m9.figshare.14825193 — see docs/TRAINING.md.

Expected input layout:

    <graz_root>/images/*.png          (e.g. 0001_1297860395_01_WRI-L1_M014.png)
    <graz_root>/labels/YOLO/*.txt     (or labels/*.txt)

Some GRAZPEDWRI mirrors keep the YOLO labels apart from the images (e.g. the
figshare `folder_structure/yolov5/labels/`). Use --images-dir / --labels-dir
to point at each explicitly in that case.

The patient id is the first underscore-separated token of the filename.

Usage:
    python prepare_grazpedwri.py <graz_root> [--out <dir>]
    python prepare_grazpedwri.py <graz_root> --images-dir <imgs> --labels-dir <lbls>
"""

import argparse
import os
import random
import shutil
from collections import defaultdict

SEED = 1337
FRACTURE_CLASS_ID = 3  # index of 'fracture' in the official 9-class label set
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def find_labels_dir(root):
    for cand in (os.path.join(root, "labels", "YOLO"), os.path.join(root, "labels")):
        if os.path.isdir(cand):
            return cand
    raise SystemExit(f"No labels directory found under {root}")


def patient_of(filename):
    return filename.split("_", 1)[0]


def link_or_copy(src, dst):
    try:
        os.link(src, dst)  # hardlink: no admin rights needed, no duplication
    except OSError:
        shutil.copyfile(src, dst)


def filter_label_file(src, dst):
    """Keep only fracture boxes, remapped to class 0. Returns #kept, #dropped."""
    kept = dropped = 0
    lines_out = []
    if os.path.exists(src):
        with open(src) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if int(parts[0]) == FRACTURE_CLASS_ID:
                    lines_out.append(" ".join(["0", *parts[1:]]))
                    kept += 1
                else:
                    dropped += 1
    with open(dst, "w") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graz_root", help="Extracted GRAZPEDWRI-DX directory")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <graz_root>/yolo_fracture)")
    ap.add_argument("--images-dir", default=None,
                    help="Override the images directory")
    ap.add_argument("--labels-dir", default=None,
                    help="Override the YOLO labels directory")
    args = ap.parse_args()

    images_dir = args.images_dir or os.path.join(args.graz_root, "images")
    labels_dir = args.labels_dir or find_labels_dir(args.graz_root)
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    out_dir = args.out or os.path.join(args.graz_root, "yolo_fracture")

    images = sorted(f for f in os.listdir(images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    # ---- patient-level split ----
    by_patient = defaultdict(list)
    for img in images:
        by_patient[patient_of(img)].append(img)
    patients = sorted(by_patient)
    random.Random(SEED).shuffle(patients)

    n = len(patients)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    split_patients = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    kept_boxes = dropped_boxes = 0
    counts = {}
    for split, pats in split_patients.items():
        img_out = os.path.join(out_dir, "images", split)
        lbl_out = os.path.join(out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        n_split = 0
        for pat in pats:
            for img in by_patient[pat]:
                stem = os.path.splitext(img)[0]
                dst_img = os.path.join(img_out, img)
                if not os.path.exists(dst_img):
                    link_or_copy(os.path.join(images_dir, img), dst_img)
                k, d = filter_label_file(
                    os.path.join(labels_dir, stem + ".txt"),
                    os.path.join(lbl_out, stem + ".txt"))
                kept_boxes += k
                dropped_boxes += d
                n_split += 1
        counts[split] = n_split

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"# GRAZPEDWRI-DX, fracture-only, patient-level split (seed {SEED})\n"
            f"path: {os.path.abspath(out_dir)}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: fracture\n"
        )

    print(f"Patients: {n} -> " +
          ", ".join(f"{s}: {len(p)} patients / {counts[s]} images"
                    for s, p in split_patients.items()))
    print(f"Boxes: kept {kept_boxes} fracture, dropped {dropped_boxes} other-class")
    print(f"[OK] Wrote {yaml_path}")
    print(f"Next: python train_detector.py {yaml_path}")


if __name__ == "__main__":
    main()
