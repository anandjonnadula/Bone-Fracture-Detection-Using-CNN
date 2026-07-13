"""Dedupe discipline for EXTERNAL datasets (UPGRADE_PLAN 3.2, step 1).

This project already paid once for train/test leakage (47% of the Stage-1
test set had byte-identical train twins — see dedupe_train_test.py). The
rule for every new data source is therefore non-negotiable:

  * an external image may enter a training folder ONLY if its content hash
    does not collide with any TEST image of either stage, and
  * external sets are deduped against each other (they overlap in the wild).

All checks are content-hash based (MD5 over file bytes), matching the
mechanism that fixed the original leak.
"""

import hashlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))

TEST_DIRS = [
    os.path.join(DATASET_DIR, "stage1_BinaryClassification", "test"),
    os.path.join(DATASET_DIR, "stage2_MultiClassification", "test"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def winlong(path):
    """Extended-length path prefix so file ops survive Windows' 260-char limit."""
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def file_hash(path):
    with open(winlong(path), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def hash_index(folder):
    """{md5: [paths]} for every image file under folder (recursive)."""
    out = {}
    for root, _dirs, files in os.walk(winlong(folder)):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                p = os.path.join(root, f)
                out.setdefault(file_hash(p), []).append(p)
    return out


def test_set_hashes():
    """Union of content hashes across BOTH stages' test sets."""
    hashes = set()
    for d in TEST_DIRS:
        if os.path.isdir(d):
            hashes |= set(hash_index(d))
    return hashes


def filter_against_test_sets(candidate_paths, forbidden=None):
    """Split candidates into (clean, leaked) against the test-set hashes.

    `forbidden` lets callers extend the block-list with hashes from other
    external sets already admitted (dedupe external-vs-external) — pass the
    same set across calls and it accumulates admitted hashes.
    """
    forbidden = set() if forbidden is None else forbidden
    forbidden |= test_set_hashes()
    clean, leaked = [], []
    for p in candidate_paths:
        h = file_hash(p)
        if h in forbidden:
            leaked.append(p)
        else:
            forbidden.add(h)  # also dedupes candidates against each other
            clean.append(p)
    return clean, leaked
