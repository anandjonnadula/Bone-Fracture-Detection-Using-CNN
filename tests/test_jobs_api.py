"""Async job API: happy path, authorization, rejection and failure states."""

from conftest import job_id_from, latest_scan, upload_xray

import db


def test_job_happy_path(patient_client, stub_pipeline):
    job_id = job_id_from(upload_xray(patient_client))

    resp = patient_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    job = resp.get_json()
    assert job["status"] == "done"  # SYNC_JOBS=1 -> already finished
    assert job["scan_id"] == latest_scan()["id"]
    assert job["redirect"].endswith(f"/results/{job['scan_id']}")

    # /processing redirects straight to results once done
    resp = patient_client.get(f"/processing/{job_id}")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith(f"/results/{job['scan_id']}")


def test_job_authorization(patient_client, patient_client_b, doctor_client,
                           client, stub_pipeline):
    job_id = job_id_from(upload_xray(patient_client))

    assert patient_client.get(f"/api/jobs/{job_id}").status_code == 200
    assert doctor_client.get(f"/api/jobs/{job_id}").status_code == 200
    assert patient_client_b.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}").status_code == 302  # anonymous
    assert patient_client.get("/api/jobs/no-such-job").status_code == 404


def test_ood_rejection_flow(patient_client, stub_pipeline):
    stub_pipeline("rejected")
    job_id = job_id_from(upload_xray(patient_client))

    job = patient_client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "rejected"
    assert "bone X-ray" in job["message"]

    scan = latest_scan()
    assert scan["verdict"] == "rejected"
    assert scan["reject_reason"] == "color_image"
    assert scan["fracture_prob"] is None  # no probability was ever computed


def test_pipeline_failure_recorded(patient_client, stub_pipeline):
    stub_pipeline("error")
    job_id = job_id_from(upload_xray(patient_client))

    job = patient_client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "failed"
    assert "synthetic pipeline failure" in job["error"]


def test_job_rows_written(patient_client, stub_pipeline):
    job_id = job_id_from(upload_xray(patient_client))
    with db.get_db_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "done"
    assert row["updated_at"]
