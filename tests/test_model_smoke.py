"""Model smoke test: the REAL pipeline on two fixture X-rays, end to end."""

import os
import shutil

import pytest
from conftest import FRACTURE_XRAY, NORMAL_XRAY, latest_scan

import db
from model import predict
from model.report import generate_pdf

pytestmark = pytest.mark.model

VERDICTS = {"fracture", "no_fracture", "uncertain"}


def test_pipeline_on_fracture_fixture(tmp_path):
    img = tmp_path / "fracture.jpg"
    shutil.copyfile(FRACTURE_XRAY, img)

    stages = []
    result = predict.predict_fracture(str(img), status_cb=stages.append)

    assert not result["rejected"]
    assert result["verdict"] in VERDICTS
    assert 0.0 <= result["fracture_prob"] <= 100.0
    assert 0.0 <= result["fracture_prob_raw"] <= 100.0
    assert result["tier"] and result["tier_text"] == predict.tier_text_for(result["tier"])
    assert stages[:3] == ["preprocessing", "ood_check", "stage1"]

    # this fixture was chosen because Stage 1 scores it confidently positive
    assert result["verdict"] == "fracture"
    assert result["top3"] and len(result["top3"]) == 3
    for t in result["top3"]:
        assert 0.0 <= t["prob"] <= 100.0
    # Grad-CAM written in BOTH forms: merged JPEG + transparent RGBA PNG
    assert result["gradcam"] and os.path.exists(result["gradcam"])
    assert result["cam_overlay"] and os.path.exists(result["cam_overlay"])
    assert result["cam_overlay"].endswith("_cam.png")

    pdf = tmp_path / "report.pdf"
    generate_pdf(
        report_path=str(pdf), original_img_path=str(img),
        gradcam_img_path=result["gradcam"], result=result["result"],
        confidence=result["confidence"], severity=result["severity"],
        recommendation=result["recommendation"],
        fracture_prob=result["fracture_prob"], top3=result["top3"],
        tier_text=result["tier_text"], threshold=result["threshold"],
        abstain_low=result["abstain_low"], abstain_high=result["abstain_high"],
    )
    assert pdf.exists() and pdf.stat().st_size > 1000


def test_pipeline_on_normal_fixture(tmp_path):
    img = tmp_path / "normal.jpg"
    shutil.copyfile(NORMAL_XRAY, img)
    result = predict.predict_fracture(str(img))

    assert not result["rejected"]
    assert result["verdict"] == "no_fracture"
    assert result["fracture_prob"] < result["abstain_low"]
    assert result["top3"] == []
    assert result["gradcam"] is None


def test_full_stack_upload_with_real_models(patient_client):
    """Upload through the actual route: scan + job + PDF, no stubs."""
    with open(FRACTURE_XRAY, "rb") as f:
        resp = patient_client.post(
            "/predict", data={"file": (f, "fixture.jpg")},
            content_type="multipart/form-data")
    assert resp.status_code == 302
    job_id = resp.headers["Location"].rsplit("/", 1)[-1]

    job = patient_client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job

    scan = latest_scan()
    assert scan["verdict"] in VERDICTS
    assert scan["tier"]
    assert scan["pdf_path"]
    assert os.path.exists(os.path.join(db.media_dir(), scan["pdf_path"]))
    assert patient_client.get(f"/results/{scan['id']}").status_code == 200
