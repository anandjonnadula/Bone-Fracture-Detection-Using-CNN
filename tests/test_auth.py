"""Register/login/logout, clinical keys, throttling, role gates."""

from conftest import create_user, login

import app as app_module
import db


def _user_role(username):
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)).fetchone()
    return row["role"] if row else None


def test_register_then_login(client):
    resp = client.post("/register",
                       data={"username": "newpatient", "password": "longenough1"})
    assert resp.status_code == 302
    assert _user_role("newpatient") == "patient"
    assert login(client, "newpatient", "longenough1").status_code == 302


def test_register_rejects_short_password(client):
    client.post("/register", data={"username": "shortpw", "password": "short"})
    assert _user_role("shortpw") is None


def test_clinical_key_required(client):
    client.post("/register-clinical", data={
        "username": "fakedoc", "password": "longenough1", "security_key": "WRONG"})
    assert _user_role("fakedoc") is None


def test_clinical_key_grants_doctor(client):
    client.post("/register-clinical", data={
        "username": "realdoc", "password": "longenough1",
        "security_key": app_module.CLINICAL_SECURITY_KEY})
    assert _user_role("realdoc") == "doctor"


def test_admin_key_grants_admin(client):
    client.post("/register-clinical", data={
        "username": "realadmin", "password": "longenough1",
        "security_key": app_module.ADMIN_SECURITY_KEY})
    assert _user_role("realadmin") == "admin"


def test_login_throttle_after_5_attempts(client):
    create_user("throttled", "correct-horse1", "patient")
    for _ in range(5):
        login(client, "throttled", "wrong-password")
    # even the CORRECT password is now locked out
    resp = login(client, "throttled", "correct-horse1")
    assert resp.status_code == 200
    assert b"Too many failed attempts" in resp.data


def test_wrong_password_rejected(client):
    create_user("someone", "correct-horse1", "patient")
    resp = login(client, "someone", "not-the-password")
    assert resp.status_code == 200
    assert b"Invalid username or password" in resp.data


def test_unauthenticated_redirects_to_login(client):
    for path in ("/", "/patient", "/doctor", "/admin", "/history", "/model-info"):
        resp = client.get(path)
        assert resp.status_code == 302, path
        assert "/login" in resp.headers["Location"], path


def test_patient_cannot_access_clinical_routes(patient_client):
    for path in ("/doctor", "/history", "/admin", "/model-info"):
        resp = patient_client.get(path)
        assert resp.status_code == 302, path  # redirected away, not rendered


def test_doctor_cannot_access_admin_routes(doctor_client):
    for path in ("/admin", "/model-info"):
        assert doctor_client.get(path).status_code == 302, path


def test_doctor_and_admin_access(doctor_client, admin_client):
    assert doctor_client.get("/doctor").status_code == 200
    assert doctor_client.get("/history").status_code == 200
    assert admin_client.get("/admin").status_code == 200
    assert admin_client.get("/model-info").status_code == 200


def test_logout_clears_session(patient_client):
    assert patient_client.get("/patient").status_code == 200
    patient_client.get("/logout")
    assert patient_client.get("/patient").status_code == 302
