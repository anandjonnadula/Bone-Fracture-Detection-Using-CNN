"""Calibration math + the three verdict routing branches + tier wording."""

import numpy as np
import pytest

from model import predict

META = {
    "threshold": 0.32,
    "threshold_calibrated": 0.30,
    "abstain_low": 0.20,
    "abstain_high": 0.45,
    "calibration": {"temperature": 1.2},
}


def test_verdict_three_branches():
    params = predict.stage1_params(META)
    assert predict.verdict_from_prob(0.05, params) == "no_fracture"
    assert predict.verdict_from_prob(0.19, params) == "no_fracture"
    assert predict.verdict_from_prob(0.20, params) == "uncertain"
    assert predict.verdict_from_prob(0.30, params) == "uncertain"
    assert predict.verdict_from_prob(0.45, params) == "uncertain"
    assert predict.verdict_from_prob(0.46, params) == "fracture"
    assert predict.verdict_from_prob(0.99, params) == "fracture"


def test_stage1_params_defaults_without_calibration():
    params = predict.stage1_params({"threshold": 0.5})
    assert params["threshold"] == 0.5
    assert params["abstain_low"] == pytest.approx(0.4)
    assert params["abstain_high"] == pytest.approx(0.63)


def test_confidence_tier_five_levels():
    assert predict.confidence_tier(0.95, META)[0] == "fracture_high"
    assert predict.confidence_tier(0.60, META)[0] == "fracture_likely"
    assert predict.confidence_tier(0.30, META)[0] == "uncertain"
    assert predict.confidence_tier(0.15, META)[0] == "clear_moderate"
    assert predict.confidence_tier(0.05, META)[0] == "clear_high"


def test_tier_wording_single_source_of_truth():
    for p in (0.95, 0.6, 0.3, 0.15, 0.05):
        slug, text = predict.confidence_tier(p, META)
        assert text == predict.tier_text_for(slug)
        assert text  # never empty


def test_calibrate_binary():
    assert predict.calibrate_binary(0.7, None) == 0.7
    assert predict.calibrate_binary(0.7, 1.0) == 0.7
    # T > 1 softens: probabilities move toward 0.5
    assert 0.5 < predict.calibrate_binary(0.9, 2.0) < 0.9
    assert 0.1 < predict.calibrate_binary(0.1, 2.0) < 0.5
    # T < 1 sharpens
    assert predict.calibrate_binary(0.9, 0.5) > 0.9


def test_calibrate_softmax_properties():
    probs = np.array([0.7, 0.2, 0.1])
    out = predict.calibrate_softmax(probs, 1.5)
    assert out.sum() == pytest.approx(1.0)
    assert list(np.argsort(out)) == list(np.argsort(probs))  # order preserved
    assert out[0] < probs[0]  # T > 1 softens the winner


def test_recommendation_for_all_verdicts():
    assert predict.recommendation_for("no_fracture", "None") \
        == predict.RECOMMENDATION_NO_FRACTURE
    assert predict.recommendation_for("uncertain", "None") \
        == predict.RECOMMENDATION_UNCERTAIN
    assert predict.recommendation_for("fracture", "Mild") \
        == predict.recommendation_dict["Mild"]
    assert predict.recommendation_for("fracture", "Unknown") \
        == predict.RECOMMENDATION_TYPE_UNCLEAR
    assert predict.recommendation_for("rejected", None) is None
