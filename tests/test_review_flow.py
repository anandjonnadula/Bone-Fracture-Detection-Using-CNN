"""Patient upload → doctor queue → review → audit trail → PDF regeneration.
Also covers private-media authorization, the annotations API, and CSRF."""

import os

from conftest import flask_app, latest_scan, upload_xray

import db


def _review(doctor_client, scan_id, decision="Approved",
            diagnosis="Hairline fracture, distal radius", notes="Confirmed."):
    return doctor_client.post("/doctor", data={
        "id": scan_id, "decision": decision,
        "final_diagnosis": diagnosis, "notes": notes,
    })


def test_full_review_flow(patient_client, doctor_client, stub_pipeline):
    # Patient uploads; pipeline (stubbed) completes synchronously.
    upload_xray(patient_client)
    scan = latest_scan()
    assert scan["verdict"] == "fracture"
    assert scan["pdf_path"], "pipeline should have produced a PDF"

    # Patient requests a specialist review.
    patient_client.post(f"/request-review/{scan['id']}")
    with db.get_db_connection() as conn:
        status = conn.execute("SELECT review_status FROM patients WHERE id = ?",
                              (scan["id"],)).fetchone()["review_status"]
    assert status == "PENDING"

    # The case shows up in the doctor queue.
    queue = doctor_client.get("/doctor")
    assert f"Record #{scan['id']}".encode() in queue.data

    # Doctor approves with a diagnosis -> status + audit columns + PDF.
    pdf_abs = os.path.join(db.media_dir(), scan["pdf_path"])
    before_mtime = os.path.getmtime(pdf_abs)
    resp = _review(doctor_client, scan["id"])
    assert resp.status_code == 302

    reviewed = latest_scan()
    assert reviewed["decision"] == "Approved"
    assert reviewed["final_diagnosis"] == "Hairline fracture, distal radius"
    assert reviewed["reviewed_by"] == "doc-grace"
    assert reviewed["reviewed_at"]
    assert reviewed["review_status"] == "REVIEWED"
    assert os.path.getmtime(pdf_abs) >= before_mtime  # regenerated


def test_uncertain_verdict_autoflags_review(patient_client, stub_pipeline):
    stub_pipeline("uncertain")
    upload_xray(patient_client)
    scan = latest_scan()
    assert scan["verdict"] == "uncertain"
    assert scan["review_status"] == "PENDING"  # flagged without patient action


def test_annotations_roundtrip_and_flattening(patient_client, doctor_client,
                                              stub_pipeline):
    upload_xray(patient_client)
    scan = latest_scan()

    doc = {"image_w": 640, "image_h": 480, "shapes": [
        {"type": "ellipse", "cx": 0.42, "cy": 0.31, "rx": 0.06, "ry": 0.04,
         "color": "#ff5252", "width": 3},
        {"type": "arrow", "x1": 0.6, "y1": 0.55, "x2": 0.45, "y2": 0.34,
         "color": "#ff5252", "width": 3},
        {"type": "label", "x": 0.62, "y": 0.57, "text": "hairline, distal radius",
         "color": "#ff5252"},
    ]}
    resp = doctor_client.post(f"/api/scans/{scan['id']}/annotations", json=doc)
    assert resp.status_code == 200 and resp.get_json()["ok"]

    # Patient can't see annotations before review…
    resp = patient_client.get(f"/api/scans/{scan['id']}/annotations")
    assert resp.get_json()["annotations"] == []

    # …the doctor records the review: annotations flatten into a PNG + PDF.
    _review(doctor_client, scan["id"])
    reviewed = latest_scan()
    assert reviewed["annotated_path"], "flattened annotation image expected"
    assert os.path.exists(os.path.join(db.media_dir(), reviewed["annotated_path"]))

    # …and the patient now sees them.
    resp = patient_client.get(f"/api/scans/{scan['id']}/annotations")
    shapes = resp.get_json()["annotations"][0]["shapes"]
    assert len(shapes) == 3


def test_annotations_rejects_patients_and_bad_shapes(patient_client,
                                                     doctor_client,
                                                     stub_pipeline):
    upload_xray(patient_client)
    scan_id = latest_scan()["id"]

    ok_shape = {"type": "ellipse", "cx": 0.5, "cy": 0.5, "rx": 0.1, "ry": 0.1}
    assert patient_client.post(f"/api/scans/{scan_id}/annotations",
                               json={"shapes": [ok_shape]}).status_code == 403

    for bad in (
        {"shapes": "not-a-list"},
        {"shapes": [{"type": "polygon"}]},                      # unknown type
        {"shapes": [{"type": "ellipse", "cx": 9, "cy": 0, "rx": 0.1, "ry": 0.1}]},
        {"shapes": [{"type": "label", "x": 0.5, "y": 0.5, "color": "javascript:"}]},
    ):
        resp = doctor_client.post(f"/api/scans/{scan_id}/annotations", json=bad)
        assert resp.status_code == 400, bad


def test_media_requires_ownership(patient_client, patient_client_b,
                                  doctor_client, client, stub_pipeline):
    upload_xray(patient_client)
    scan = latest_scan()
    url = f"/media/{scan['image_path']}"

    assert patient_client.get(url).status_code == 200       # owner
    assert doctor_client.get(url).status_code == 200        # clinical staff
    assert patient_client_b.get(url).status_code == 404     # other patient
    assert client.get(url).status_code == 302               # anonymous -> login
    assert patient_client.get("/media/no-such-file.png").status_code == 404


def test_results_page_authz(patient_client, patient_client_b, stub_pipeline):
    upload_xray(patient_client)
    scan = latest_scan()
    assert patient_client.get(f"/results/{scan['id']}").status_code == 200
    assert patient_client_b.get(f"/results/{scan['id']}").status_code == 404


def test_post_without_csrf_token_fails(patient_client, stub_pipeline):
    upload_xray(patient_client)
    scan_id = latest_scan()["id"]
    flask_app.config["WTF_CSRF_ENABLED"] = True
    try:
        resp = patient_client.post(f"/request-review/{scan_id}")
        assert resp.status_code == 400
        resp = upload_xray(patient_client)
        assert resp.status_code == 400
    finally:
        flask_app.config["WTF_CSRF_ENABLED"] = False
