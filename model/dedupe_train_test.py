"""Remove train/test leakage from the dataset.

Byte-identical copies of test images were found inside the training folders
(277 of 594 Stage-1 test images, 20 of 355 Stage-2 test images). Any model
trained on those folders is partially evaluated on images it saw during
training, inflating test accuracy.

This script:
  1. moves the duplicated TRAIN files into .dataset_backups/leaked_train_duplicates/
     (the test set stays untouched so metrics remain comparable), and
  2. writes saved_model/test_leakage_stage{1,2}.json listing which TEST files
     have (or had) train-side twins, so evaluations can also report metrics on
     the leak-free subset.

Usage:  python dedupe_train_test.py
"""

import hashlib
import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
BACKUP_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", ".dataset_backups", "leaked_train_duplicates")
)
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")


def winlong(path):
    """Extended-length path prefix so file ops survive Windows' 260-char limit."""
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def file_hashes(folder):
    out = {}
    for root, _, files in os.walk(winlong(folder)):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out.setdefault(hashlib.md5(fh.read()).hexdigest(), []).append(p)
    return out


def dedupe(stage_folder, tag):
    train_dir = os.path.join(DATASET_DIR, stage_folder, "train")
    test_dir = os.path.join(DATASET_DIR, stage_folder, "test")
    # Train files moved out by a previous (possibly interrupted) run still
    # count as train-side twins when recording which TEST files were leaked.
    moved_dir = os.path.join(BACKUP_DIR, stage_folder, "train")

    train_hashes = file_hashes(train_dir)
    already_moved = file_hashes(moved_dir) if os.path.isdir(moved_dir) else {}
    test_hashes = file_hashes(test_dir)

    leaked = (set(train_hashes) | set(already_moved)) & set(test_hashes)
    moved = 0
    leaked_test_files = []

    for h in leaked:
        for test_path in test_hashes[h]:
            rel_test = os.path.relpath(test_path, winlong(DATASET_DIR))
            leaked_test_files.append(rel_test.replace("\\", "/"))
        for train_path in train_hashes.get(h, []):
            rel = os.path.relpath(train_path, winlong(DATASET_DIR))
            dest = os.path.join(BACKUP_DIR, rel)
            os.makedirs(winlong(os.path.dirname(dest)), exist_ok=True)
            shutil.move(train_path, winlong(dest))
            moved += 1

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_json = os.path.join(SAVE_DIR, f"test_leakage_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(sorted(leaked_test_files), f, indent=2)

    print(f"{stage_folder}: moved {moved} duplicated train files out "
          f"({len(leaked)} unique images); "
          f"{len(leaked_test_files)} test files recorded in {os.path.basename(out_json)}")


def main():
    dedupe("stage1_BinaryClassification", "stage1")
    dedupe("stage2_MultiClassification", "stage2")
    # Report post-dedupe class counts
    for stage in ("stage1_BinaryClassification", "stage2_MultiClassification"):
        train_dir = winlong(os.path.join(DATASET_DIR, stage, "train"))
        counts = {
            d: len(os.listdir(os.path.join(train_dir, d)))
            for d in sorted(os.listdir(train_dir))
            if os.path.isdir(os.path.join(train_dir, d))
        }
        print(f"  {stage} train counts after dedupe: {counts}")


if __name__ == "__main__":
    main()
