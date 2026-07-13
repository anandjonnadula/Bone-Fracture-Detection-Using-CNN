"""OOD gate: obvious non-X-rays are refused; real radiographs pass."""

import numpy as np
import pytest
from conftest import FRACTURE_XRAY, NORMAL_XRAY

from model import ood_gate, predict

pytestmark = pytest.mark.model  # embedding checks load the Stage-1 model

SIZE = (predict.IMG_SIZE, predict.IMG_SIZE, 3)


def _non_xray_images():
    rng = np.random.default_rng(42)
    color_photo = rng.uniform(0, 255, SIZE).astype(np.float32)  # colored noise
    sunset = np.zeros(SIZE, np.float32)
    sunset[..., 0] = 240  # heavily saturated red -> color pre-filter
    black = np.zeros(SIZE, np.float32)
    white = np.full(SIZE, 255.0, np.float32)
    gray_noise = np.repeat(
        rng.uniform(0, 255, SIZE[:2]).astype(np.float32)[..., None], 3, axis=2)
    return {
        "color_photo": color_photo,
        "saturated_red": sunset,
        "blank_black": black,
        "blank_white": white,
        "grayscale_noise": gray_noise,
    }


def test_non_xray_images_rejected():
    for name, arr in _non_xray_images().items():
        ok, reason, details = predict.check_ood(arr)
        assert not ok, f"{name} should be rejected ({details})"
        assert reason in (ood_gate.REASON_COLOR, ood_gate.REASON_NOT_XRAY)


def test_blank_black_image_rejected_by_distance():
    ok, reason, _ = predict.check_ood(np.zeros(SIZE, np.float32))
    assert not ok
    assert reason == ood_gate.REASON_NOT_XRAY  # monochrome -> embedding gate


def test_color_image_rejected_by_prefilter():
    arr = np.zeros(SIZE, np.float32)
    arr[..., 1] = 220  # pure green
    ok, reason, details = predict.check_ood(arr)
    assert not ok and reason == ood_gate.REASON_COLOR
    assert details["saturation"] > ood_gate.SATURATION_LIMIT


def test_real_xrays_pass():
    for path in (FRACTURE_XRAY, NORMAL_XRAY):
        arr = predict._load_image_array(path)
        ok, reason, details = predict.check_ood(arr)
        assert ok, f"{path} was rejected: {reason} ({details})"


def test_stats_artifact_sanity():
    stats = ood_gate.load_stats()
    assert stats is not None, "ood_stats.npz missing — run build_ood_stats.py"
    assert stats["refs"].ndim == 2 and len(stats["refs"]) >= 500
    assert 0 < stats["threshold"] < 2
    # references are L2-normalized (float16 storage tolerance)
    norms = np.linalg.norm(stats["refs"], axis=1)
    assert np.allclose(norms, 1.0, atol=5e-2)
