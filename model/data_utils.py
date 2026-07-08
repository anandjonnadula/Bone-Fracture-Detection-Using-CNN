"""Shared dataset pipeline for training and evaluation.

Uses tf.data + image_dataset_from_directory (the modern replacement for the
deprecated ImageDataGenerator). Images are yielded as raw 0-255 float tensors;
pixel normalization lives inside the models (see cnn_model.py).
"""

import os

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 1337

AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter():
    """X-ray-appropriate augmentation.

    Horizontal flips are anatomically valid (left/right limbs); vertical
    flips are not. Rotations/shifts stay small — X-rays are roughly upright.
    Brightness/contrast jitter simulates exposure differences between
    radiography machines.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.045, fill_mode="nearest", seed=SEED),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
            tf.keras.layers.RandomZoom(0.12, fill_mode="nearest", seed=SEED),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0.0, 255.0), seed=SEED),
            tf.keras.layers.RandomContrast(0.12, seed=SEED),
        ],
        name="augmenter",
    )


def load_train_val(data_dir, label_mode, validation_split=0.15, batch_size=BATCH_SIZE):
    """Train/validation datasets from one directory (stratified by shuffle+seed).

    Returns (train_ds, val_ds, class_names). Datasets are unbatched-order
    stable across calls thanks to the fixed seed.
    """
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
        validation_split=validation_split,
        subset="both",
    )
    return train_ds, val_ds, train_ds.class_names


def load_test(data_dir, label_mode, batch_size=BATCH_SIZE):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )
    return ds, ds.class_names


def prepare(ds, augmenter=None, cache_in_memory=False):
    """Attach augmentation / caching / prefetching to a dataset."""
    if cache_in_memory:
        ds = ds.cache()
    if augmenter is not None:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(AUTOTUNE)


def count_per_class(data_dir):
    """Number of image files per class subfolder (alphabetical order)."""
    counts = {}
    for name in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, name)
        if os.path.isdir(sub):
            counts[name] = len(
                [f for f in os.listdir(sub) if not f.startswith(".")]
            )
    return counts


def class_weights_from_counts(counts_in_label_order):
    """Balanced class weights: w_c = total / (num_classes * n_c)."""
    counts = np.asarray(counts_in_label_order, dtype=np.float64)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def merge_histories(*histories):
    """Concatenate epoch metric lists from successive fit() phases."""
    merged = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged
