"""Bone Fracture Detection Using CNN — Flask application.

Hardened rewrite:
  * Secrets come from environment variables (SECRET_KEY, CLINICAL_KEY,
    ADMIN_KEY) with development fallbacks.
  * Uploads are validated (extension whitelist + real-image check + 15 MB
    cap) and stored under collision-proof UUID filenames — previously two
    uploads named "xray.jpg" silently overwrote each other and old records
    pointed at the wrong patient's image.
  * SQLite runs with foreign keys ON, proper indexes, and idempotent
    column migrations; rows are accessed by name, not position.
  * The orphaned 'admin' role now has a real /admin analytics dashboard;
    /learn and /model-info no longer 500 (their templates exist).
  * Login attempts are throttled; passwords require a minimum length.
  * Doctor reviews are audited (reviewed_by / reviewed_at) and the PDF
    report is regenerated to include the doctor's diagnosis.
"""

import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from model.predict import generate_pdf, get_model_performance, predict_fracture

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fracture-app")

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB — matches the UI promise
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Registration keys for privileged roles (override via environment)
CLINICAL_SECURITY_KEY = os.environ.get("CLINICAL_KEY", "DOC-VVIT-2026")
ADMIN_SECURITY_KEY = os.environ.get("ADMIN_KEY", "ADM-VVIT-2026")

DB_PATH = os.path.join(BASE_DIR, "database.db")
# No branding on PDF reports; generate_pdf renders a text-only header when None.
LOGO_PATH = None

MIN_PASSWORD_LENGTH = 8

# Simple in-memory login throttle: username -> list of failed-attempt times
_failed_logins = {}
MAX_ATTEMPTS = 5
LOCKOUT_WINDOW = 600  # seconds


# ---------------- DATABASE ---------------- #
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def _existing_columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            result TEXT,
            confidence REAL,
            severity TEXT,
            date TEXT,
            notes TEXT,
            decision TEXT,
            final_diagnosis TEXT,
            user_id INTEGER REFERENCES users(id),
            review_status TEXT DEFAULT 'NONE',
            gradcam_path TEXT,
            pdf_path TEXT,
            fracture_prob REAL,
            reviewed_by TEXT,
            reviewed_at TEXT
        )
        """)

        # Idempotent migration for databases created by older versions.
        wanted = {
            "user_id": "INTEGER",
            "review_status": "TEXT DEFAULT 'NONE'",
            "gradcam_path": "TEXT",
            "pdf_path": "TEXT",
            "fracture_prob": "REAL",
            "reviewed_by": "TEXT",
            "reviewed_at": "TEXT",
        }
        have = _existing_columns(conn, "patients")
        for column, decl in wanted.items():
            if column not in have:
                conn.execute(f"ALTER TABLE patients ADD COLUMN {column} {decl}")
                log.info("DB migration: added patients.%s", column)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_user ON patients(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_review ON patients(review_status)")
        conn.commit()


init_db()


# ---------------- AUTH HELPERS ---------------- #
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if session.get("role") not in roles:
                flash("Unauthorized: this section is restricted.", "error")
                return redirect(url_for("index"))
            return f(*args, **kwargs)
        return wrapper
    return decorator


doctor_required = role_required("doctor", "admin")
admin_required = role_required("admin")


def _throttled(username):
    now = time.time()
    attempts = [t for t in _failed_logins.get(username, []) if now - t < LOCKOUT_WINDOW]
    _failed_logins[username] = attempts
    return len(attempts) >= MAX_ATTEMPTS


def _register_user(username, password, role):
    """Shared registration logic. Returns an error message or None."""
    if not username or not password:
        return "Username and password are required."
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    with get_db_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            return "Username already exists."
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
    log.info("Registered new %s account: %s", role, username)
    return None


# ---------------- AUTH ROUTES ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if _throttled(username):
            flash("Too many failed attempts. Try again in a few minutes.", "error")
            return render_template("login.html")

        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            _failed_logins.pop(username, None)
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("index"))

        _failed_logins.setdefault(username, []).append(time.time())
        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Public registration — patients only."""
    if request.method == "POST":
        error = _register_user(
            (request.form.get("username") or "").strip(),
            request.form.get("password") or "",
            "patient",
        )
        if error:
            flash(error, "error")
        else:
            flash("Patient registration successful. Please login.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/register-clinical", methods=["GET", "POST"])
def register_clinical():
    """Hidden registration for clinical staff (doctor / admin via key)."""
    if request.method == "POST":
        key = request.form.get("security_key") or ""
        if key == ADMIN_SECURITY_KEY:
            role = "admin"
        elif key == CLINICAL_SECURITY_KEY:
            role = "doctor"
        else:
            flash("Invalid Clinical Security Key. Access denied.", "error")
            return render_template("register_clinical.html")

        error = _register_user(
            (request.form.get("username") or "").strip(),
            request.form.get("password") or "",
            role,
        )
        if error:
            flash(error, "error")
        else:
            flash(f"{role.capitalize()} registration successful. Please login.", "success")
            return redirect(url_for("login"))
    return render_template("register_clinical.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- HOME ---------------- #
@app.route("/")
@login_required
def index():
    return render_template("index.html")


# ---------------- PREDICT ---------------- #
def _validate_upload(file):
    """Returns (safe_filename, error_message). Exactly one is None."""
    if file is None or not file.filename:
        return None, "No file selected."

    name = secure_filename(file.filename)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return None, "Unsupported file type. Please upload a JPG or PNG X-ray image."

    # Confirm it is actually a decodable image, not just a renamed file.
    try:
        Image.open(file.stream).verify()
    except Exception:
        return None, "The uploaded file is not a valid image."
    file.stream.seek(0)

    unique = f"{uuid.uuid4().hex[:12]}_{name}"
    return unique, None


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files.get("file")
    file_name, error = _validate_upload(file)
    if error:
        flash(error, "error")
        return redirect(url_for("index"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)

    try:
        result_data = predict_fracture(file_path)
    except Exception as e:
        log.exception("Prediction failed for %s", file_name)
        flash(f"Error during diagnostic scanning: {e}", "error")
        return redirect(url_for("index"))

    rel_image = f"static/uploads/{file_name}"
    gradcam_abs = result_data.get("gradcam")
    rel_gradcam = None
    if gradcam_abs and os.path.exists(gradcam_abs):
        rel_gradcam = f"static/uploads/{os.path.basename(gradcam_abs)}"

    # Save the scan first so the PDF can reference its record number.
    with get_db_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO patients
                (image_path, result, confidence, severity, date, user_id,
                 gradcam_path, fracture_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel_image,
                result_data.get("result"),
                float(result_data.get("confidence") or 0),
                result_data.get("severity"),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                session.get("user_id"),
                rel_gradcam,
                float(result_data.get("fracture_prob") or 0),
            ),
        )
        record_id = cur.lastrowid
        conn.commit()

    # PDF report
    report_filename = None
    try:
        report_filename = f"{os.path.splitext(file_name)[0]}_report.pdf"
        report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)
        generate_pdf(
            report_path=report_path,
            original_img_path=file_path,
            gradcam_img_path=gradcam_abs,
            result=result_data.get("result"),
            confidence=result_data.get("confidence"),
            severity=result_data.get("severity"),
            recommendation=result_data.get("recommendation"),
            fracture_prob=result_data.get("fracture_prob"),
            top3=result_data.get("top3"),
            logo_path=LOGO_PATH,
            record_id=record_id,
        )
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE patients SET pdf_path = ? WHERE id = ?",
                (f"static/uploads/{report_filename}", record_id),
            )
            conn.commit()
    except Exception:
        log.exception("PDF generation failed for record %s", record_id)
        report_filename = None

    log.info(
        "Prediction #%s by user %s: %s (conf %.2f%%)",
        record_id, session.get("username"),
        result_data.get("result"), result_data.get("confidence") or 0,
    )

    return render_template(
        "result.html",
        record_id=record_id,
        result=result_data.get("result"),
        fracture_detected=result_data.get("fracture_detected"),
        confidence=result_data.get("confidence"),
        fracture_prob=result_data.get("fracture_prob"),
        threshold=result_data.get("threshold"),
        top3=result_data.get("top3"),
        image_path=rel_image,
        gradcam_path=rel_gradcam,
        severity=result_data.get("severity"),
        recommendation=result_data.get("recommendation"),
        pdf_report=report_filename,
        performance=get_model_performance(),
    )


# ---------------- PATIENT DASHBOARD ---------------- #
@app.route("/patient")
@login_required
def patient_dashboard():
    with get_db_connection() as conn:
        records = conn.execute(
            "SELECT * FROM patients WHERE user_id = ? ORDER BY id DESC",
            (session.get("user_id"),),
        ).fetchall()
    return render_template(
        "patient_dashboard.html", records=records, no_data=(len(records) == 0)
    )


# ---------------- REQUEST REVIEW ---------------- #
@app.route("/request-review/<int:record_id>", methods=["POST"])
@login_required
def request_review(record_id):
    with get_db_connection() as conn:
        record = conn.execute(
            "SELECT user_id FROM patients WHERE id = ?", (record_id,)
        ).fetchone()
        if record and record["user_id"] == session.get("user_id"):
            conn.execute(
                "UPDATE patients SET review_status = 'PENDING' WHERE id = ?",
                (record_id,),
            )
            conn.commit()
            flash("Specialist review requested successfully.", "success")
        else:
            flash("Unauthorized request.", "error")
    return redirect(url_for("patient_dashboard"))


# ---------------- HISTORY (doctor) ---------------- #
@app.route("/history")
@doctor_required
def history():
    with get_db_connection() as conn:
        data = conn.execute(
            """
            SELECT p.*, u.username AS patient_name
            FROM patients p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC
            """
        ).fetchall()
    return render_template("patient_history.html", data=data)


# ---------------- DOCTOR DASHBOARD ---------------- #
@app.route("/doctor", methods=["GET", "POST"])
@doctor_required
def doctor_dashboard():
    if request.method == "POST":
        patient_id = request.form.get("id")
        notes = request.form.get("notes")
        decision = request.form.get("decision")
        final_diagnosis = request.form.get("final_diagnosis")

        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE patients
                SET notes = ?, decision = ?, final_diagnosis = ?,
                    reviewed_by = ?, reviewed_at = ?,
                    review_status = CASE WHEN review_status = 'PENDING'
                                         THEN 'REVIEWED' ELSE review_status END
                WHERE id = ?
                """,
                (
                    notes, decision, final_diagnosis,
                    session.get("username"),
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    patient_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()

        # Regenerate the PDF so it carries the doctor's review.
        if row and row["pdf_path"]:
            try:
                generate_pdf(
                    report_path=os.path.join(BASE_DIR, row["pdf_path"]),
                    original_img_path=os.path.join(BASE_DIR, row["image_path"] or ""),
                    gradcam_img_path=(
                        os.path.join(BASE_DIR, row["gradcam_path"])
                        if row["gradcam_path"] else None
                    ),
                    result=row["result"],
                    confidence=row["confidence"],
                    severity=row["severity"],
                    fracture_prob=row["fracture_prob"],
                    logo_path=LOGO_PATH,
                    record_id=row["id"],
                    doctor_diagnosis=final_diagnosis,
                    doctor_notes=notes,
                )
            except Exception:
                log.exception("PDF regeneration failed for record %s", patient_id)

        flash(f"Review saved for record #{patient_id}.", "success")
        return redirect(url_for("doctor_dashboard"))

    with get_db_connection() as conn:
        data = conn.execute(
            """
            SELECT p.*, u.username AS patient_name
            FROM patients p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY (p.review_status = 'PENDING') DESC, p.id DESC
            """
        ).fetchall()
    return render_template("doctor_dashboard.html", cases=data)


# ---------------- ADMIN ANALYTICS ---------------- #
@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db_connection() as conn:
        total_scans = conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]
        fractures = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE result LIKE 'Fracture%'"
        ).fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE review_status = 'PENDING'"
        ).fetchone()["c"]
        reviewed = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE reviewed_by IS NOT NULL"
        ).fetchone()["c"]
        users_by_role = conn.execute(
            "SELECT role, COUNT(*) c FROM users GROUP BY role"
        ).fetchall()
        type_rows = conn.execute(
            """
            SELECT TRIM(SUBSTR(result, INSTR(result, ':') + 1)) AS ftype, COUNT(*) c
            FROM patients
            WHERE result LIKE 'Fracture Detected:%'
            GROUP BY ftype ORDER BY c DESC
            """
        ).fetchall()
        by_day = conn.execute(
            """
            SELECT SUBSTR(date, 1, 10) AS day, COUNT(*) c
            FROM patients
            WHERE date IS NOT NULL
            GROUP BY day ORDER BY day DESC LIMIT 14
            """
        ).fetchall()
        recent = conn.execute(
            """
            SELECT p.id, p.result, p.severity, p.date, u.username AS patient_name
            FROM patients p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC LIMIT 8
            """
        ).fetchall()

    return render_template(
        "admin_dashboard.html",
        total_scans=total_scans,
        fractures=fractures,
        no_fractures=total_scans - fractures,
        pending=pending,
        reviewed=reviewed,
        users_by_role=users_by_role,
        type_rows=type_rows,
        by_day=list(reversed(by_day)),
        recent=recent,
    )


# ---------------- STATIC-ISH PAGES ---------------- #
@app.route("/hospitals")
@login_required
def hospitals():
    if session.get("role") == "doctor":
        flash("Facility Locator is intended for patient use.", "error")
        return redirect(url_for("index"))
    return render_template("hospitals.html")


@app.route("/learn")
@login_required
def learn():
    return render_template("learn.html")


@app.route("/model-info")
@admin_required
def model_info():
    return render_template("model_info.html", performance=get_model_performance())


# ---------------- ERROR HANDLERS ---------------- #
@app.errorhandler(413)
def too_large(_):
    flash("File too large — the limit is 15 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404,
                           message="The page you requested does not exist."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500,
                           message="Something went wrong on our side."), 500


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug)
