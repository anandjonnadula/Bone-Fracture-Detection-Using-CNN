"""One-time dataset sanitizer.

The dataset (scraped from public sources) contains a handful of truncated
JPEGs and oddly-encoded PNGs. The original training code masked this with
PIL's LOAD_TRUNCATED_IMAGES, but tf.data's native decoder is strict and
aborts mid-epoch. This script finds every image TensorFlow cannot decode and
re-encodes it losslessly-as-possible via PIL (tolerant mode). Originals are
backed up to .dataset_backups/ (outside the training folders) before being
replaced.

Usage:  python sanitize_dataset.py
"""

import os
import shutil

import tensorflow as tf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
BACKUP_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", ".dataset_backups"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def tf_decodable(path):
    """True if TensorFlow's strict decoder fully accepts the file."""
    try:
        data = tf.io.read_file(path)
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        _ = img.shape
        return True
    except Exception:
        return False


def reencode(path):
    """Re-save the image via PIL (tolerant), backing up the original."""
    rel = os.path.relpath(path, DATASET_DIR)
    backup_path = os.path.join(BACKUP_DIR, rel)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)

    img = Image.open(path)
    img = img.convert("RGB")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        img.save(path, format="PNG", icc_profile=None)
    else:
        img.save(path, format="JPEG", quality=95, icc_profile=None)


def main():
    checked = fixed = failed = 0
    failures = []
    for root, _, files in os.walk(DATASET_DIR):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            path = os.path.join(root, name)
            checked += 1
            if checked % 2000 == 0:
                print(f"  scanned {checked} images ({fixed} repaired so far)...", flush=True)
            if tf_decodable(path):
                continue
            try:
                reencode(path)
                if tf_decodable(path):
                    fixed += 1
                    print(f"  [FIXED] {os.path.relpath(path, DATASET_DIR)}", flush=True)
                else:
                    raise RuntimeError("still undecodable after re-encode")
            except Exception as e:
                failed += 1
                failures.append(path)
                print(f"  [FAILED] {os.path.relpath(path, DATASET_DIR)}: {e}", flush=True)

    print(f"\nScanned {checked} images: repaired {fixed}, unrepairable {failed}.")
    if failures:
        print("Unrepairable files (consider deleting them):")
        for p in failures:
            print("  ", p)
    if fixed:
        print(f"Originals backed up under {BACKUP_DIR}")


if __name__ == "__main__":
    main()
