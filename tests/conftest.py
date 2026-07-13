"""Shared fixtures: temp DB + media dir, role-based logged-in clients, and a
stubbed pipeline so the fast suite never loads TensorFlow weights.

The environment MUST be configured before `app` is imported — the module
reads DATABASE_PATH / MEDIA_DIR lazily but runs init_db() at import time.
"""

import io
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="bfd-tests-")
os.environ["DATABASE_PATH"] = os.path.join(_TMP, "test.db")
os.environ["MEDIA_DIR"] = os.path.join(_TMP, "media")
os.environ["SYNC_JOBS"] = "1"          # pipeline runs inline -> deterministic tests
os.environ["RATELIMIT_ENABLED"] = "0"  # individual tests re-enable what they need
os.environ.pop("DEMO_MODE", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402

import app as app_module  # noqa: E402
import db  # noqa: E402
from model import predict  # noqa: E402

flask_app = app_module.app

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FRACTURE_XRAY = os.path.join(FIXTURES, "fracture.jpg")
NORMAL_XRAY = os.path.join(FIXTURES, "normal.jpg")


@pytest.fixture(autouse=True)
def clean_state():
    """Fresh tables, empty media dir, CSRF off (tests opt back in)."""
    flask_app.config["WTF_CSRF_ENABLED"] = False
    app_module._failed_logins.clear()
    with db.get_db_connection() as conn:
        for table in ("scan_annotations", "jobs", "patients", "users"):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    for name in os.listdir(db.media_dir()):
        try:
            os.remove(os.path.join(db.media_dir(), name))
        except OSError:
            pass
    yield


@pytest.fixture
def client():
    return flask_app.test_client()


def create_user(username, password, role):
    with db.get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
        return cur.lastrowid


def login(client, username, password):
    return client.post(
        "/login", data={"username": username, "password": password},
        follow_redirects=False,
    )


def _role_client(role, username):
    c = flask_app.test_client()
    create_user(username, "sup3r-secret", role)
    resp = login(c, username, "sup3r-secret")
    assert resp.status_code == 302
    return c


@pytest.fixture
def patient_client():
    return _role_client("patient", "pat-alice")


@pytest.fixture
def patient_client_b():
    return _role_client("patient", "pat-bob")


@pytest.fixture
def doctor_client():
    return _role_client("doctor", "doc-grace")


@pytest.fixture
def admin_client():
    return _role_client("admin", "adm-root")


# ---------------- pipeline stubbing ----------------
def make_png_bytes(size=(64, 64), color=(90, 90, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def stub_result(img_path, verdict="fracture"):
    """Canned pipeline results shaped exactly like predict.predict_fracture."""
    root, _ = os.path.splitext(img_path)
    if verdict == "rejected":
        return {
            "rejected": True,
            "reject_reason": "color_image",
            "reject_message": predict.OOD_REJECT_MESSAGE,
            "ood": {"saturation": 0.9},
        }
    common = {
        "rejected": False,
        "verdict": verdict,
        "fracture_prob": 91.2 if verdict == "fracture" else
                         30.0 if verdict == "uncertain" else 2.1,
        "fracture_prob_raw": 90.0,
        "threshold": 30.4,
        "abstain_low": 20.4,
        "abstain_high": 43.4,
        "tier": {"fracture": "fracture_high", "uncertain": "uncertain",
                 "no_fracture": "clear_high"}[verdict],
        "tier_text": "",
        "ood": {"saturation": 0.01},
        "detections": None,
    }
    common["tier_text"] = predict.tier_text_for(common["tier"])
    if verdict == "no_fracture":
        return {
            **common,
            "result": "No Fracture Detected", "fracture_detected": False,
            "confidence": 97.9, "fracture_type": None, "type_unclear": False,
            "top3": [], "gradcam": None, "cam_overlay": None,
            "severity": "None",
            "recommendation": predict.RECOMMENDATION_NO_FRACTURE,
        }
    # fracture / uncertain need heatmap artifacts on disk (jobs stores basenames)
    gradcam, cam = f"{root}_gradcam.jpg", f"{root}_cam.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(gradcam)
    Image.new("RGBA", (64, 64), (200, 30, 30, 120)).save(cam)
    top3 = [{"label": "Hairline", "prob": 55.0},
            {"label": "Oblique", "prob": 25.0},
            {"label": "Spiral", "prob": 10.0}]
    if verdict == "uncertain":
        return {
            **common,
            "result": "Uncertain — Specialist Review Requested",
            "fracture_detected": False, "confidence": 55.0,
            "fracture_type": None, "type_unclear": True, "top3": top3,
            "gradcam": gradcam, "cam_overlay": cam, "severity": "None",
            "recommendation": predict.RECOMMENDATION_UNCERTAIN,
        }
    return {
        **common,
        "result": "Fracture Detected: Hairline", "fracture_detected": True,
        "confidence": 55.0, "fracture_type": "Hairline", "type_unclear": False,
        "top3": top3, "gradcam": gradcam, "cam_overlay": cam,
        "severity": "Mild",
        "recommendation": predict.recommendation_dict["Mild"],
    }


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the real pipeline with a canned result. Returns a setter so a
    test can pick the verdict ('fracture' default, 'no_fracture',
    'uncertain', 'rejected', or 'error' to simulate a crash)."""
    mode = {"verdict": "fracture"}

    def fake_predict(img_path, status_cb=None, run_detector=True):
        if status_cb:
            for stage in ("preprocessing", "ood_check", "stage1"):
                status_cb(stage)
        if mode["verdict"] == "error":
            raise RuntimeError("synthetic pipeline failure")
        return stub_result(img_path, mode["verdict"])

    monkeypatch.setattr(predict, "predict_fracture", fake_predict)

    def set_verdict(v):
        mode["verdict"] = v
    return set_verdict


def upload_xray(client, filename="scan.png", data=None):
    """POST an upload; returns the response (302 -> /processing/<job_id>)."""
    payload = {"file": (data or make_png_bytes(), filename)}
    return client.post("/predict", data=payload,
                       content_type="multipart/form-data")


def job_id_from(resp):
    assert resp.status_code == 302, resp.data
    location = resp.headers["Location"]
    assert "/processing/" in location
    return location.rsplit("/", 1)[-1]


def latest_scan():
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM patients ORDER BY id DESC LIMIT 1").fetchone()
