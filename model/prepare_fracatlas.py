"""Merge FracAtlas into Stage-1 training data — with the dedupe gate.

FracAtlas (https://doi.org/10.6084/m9.figshare.22363012) is ~4,083 X-rays
labeled fracture / non-fracture, which maps 1:1 onto Stage 1's task. Its
license (CC-BY 4.0) permits this use with attribution (see docs/TRAINING.md).

Expected download layout (the official zip):

    <fracatlas_root>/
        images/Fractured/*.jpg
        images/Non_fractured/*.jpg

Every image passes through the content-hash gate against BOTH stages' test
sets before it may enter dataset/stage1_BinaryClassification/train/ — the
leakage lesson this project already paid for once.

Usage (from model/):

    python prepare_fracatlas.py <fracatlas_root> [--dry-run]

Afterwards retrain Stage 1 (optionally from the radiograph backbone) and
RE-RUN calibrate.py + build_ood_stats.py — both are per-model artifacts.
"""

import argparse
import os
import shutil

from external_data import IMAGE_EXTS, filter_against_test_sets, winlong

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_TRAIN = os.path.normpath(os.path.join(
    BASE_DIR, "..", "dataset", "stage1_BinaryClassification", "train"))

SOURCES = {
    # FracAtlas folder name -> stage1 class folder
    "Fractured": "fracture",
    "Non_fractured": "no_fracture",
}


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(winlong(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def _decodable(path):
    """True only if the image fully decodes.

    FracAtlas ships some truncated JPEGs (valid header, missing image data).
    PIL's tolerant mode would hide them, but the tf.data training pipeline's
    strict decoder crashes on them — so reject anything that doesn't fully
    load, matching the dedupe/sanitize discipline the rest of the repo uses.
    """
    from PIL import Image
    try:
        with Image.open(winlong(path)) as im:
            im.load()  # forces a full decode; raises on truncation
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fracatlas_root", help="Path to the extracted FracAtlas download")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied without touching the dataset")
    args = ap.parse_args()

    images_root = os.path.join(args.fracatlas_root, "images")
    if not os.path.isdir(images_root):
        raise SystemExit(f"Not a FracAtlas layout: {images_root} missing "
                         "(expected images/Fractured + images/Non_fractured)")

    admitted_hashes = set()
    total_copied = total_leaked = total_corrupt = 0
    for src_name, dst_class in SOURCES.items():
        src_dir = os.path.join(images_root, src_name)
        if not os.path.isdir(src_dir):
            print(f"!! Skipping missing folder {src_dir}")
            continue
        candidates = list_images(src_dir)
        clean, leaked = filter_against_test_sets(candidates, admitted_hashes)
        print(f"{src_name}: {len(candidates)} images -> {len(clean)} clean, "
              f"{len(leaked)} blocked (test-set or duplicate collision)")
        total_leaked += len(leaked)

        dst_dir = os.path.join(STAGE1_TRAIN, dst_class)
        os.makedirs(winlong(dst_dir), exist_ok=True)
        corrupt = 0
        for p in clean:
            dst = os.path.join(dst_dir, f"fracatlas_{os.path.basename(p)}")
            if os.path.exists(winlong(dst)):
                continue
            if not _decodable(p):  # skip truncated/undecodable images
                corrupt += 1
                continue
            if not args.dry_run:
                shutil.copyfile(winlong(p), winlong(dst))
            total_copied += 1
        if corrupt:
            print(f"  {corrupt} {src_name} images skipped (truncated/undecodable)")
        total_corrupt += corrupt

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"\n[OK] {verb} {total_copied} images into {STAGE1_TRAIN} "
          f"({total_leaked} blocked by the dedupe gate, "
          f"{total_corrupt} skipped as undecodable).")
    print("Next: retrain stage 1, then re-run calibrate.py and build_ood_stats.py.")


if __name__ == "__main__":
    main()
