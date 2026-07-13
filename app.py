"""Bone Fracture Detection Using CNN — Flask application.

Production-hardened build:
  * CSRF protection on every POST (Flask-WTF), rate limiting (Flask-Limiter),
    strict security headers with a nonce'd CSP, secure session cookies.
  * Patient images/reports live in a PRIVATE media dir served only through
    the authorizing /media route (owner or doctor/admin) — never /static.
  * Inference is asynchronous: uploads insert a scan + job row, a
    ThreadPoolExecutor runs the pipeline, the browser polls /api/jobs/<id>
    and shows a live progress stepper.
  * Every scan gets a calibrated verdict (fracture | no_fracture | uncertain);
    uncertain scans are auto-flagged into the doctor queue. An OOD gate
    rejects non-radiographs before any probability is computed.
  * Real DICOM (.dcm) support with a strict no-PHI policy.
  * Doctors annotate scans (vector JSON) — annotations are flattened
    server-side into the regenerated PDF and shown to the patient.
  * DEMO_MODE=1 enables seeded demo accounts, a sample gallery, aggressive
    limits and a periodic data wipe for public deployments.

Configuration is environment-driven: SECRET_KEY, CLINICAL_KEY, ADMIN_KEY,
DATABASE_PATH, MEDIA_DIR, DEMO_MODE, PORT, SECURE_COOKIES, PRELOAD_MODELS.
"""

import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import jobs
from db import get_db_connection, init_db, media_dir
from model import dicom_utils, ood_gate, predict
from model.predict import get_model_performance, tier_text_for
from model.report import flatten_annotations, generate_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fracture-app")

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me-in-production")
app.config["DEMO_MODE"] = os.environ.get("DEMO_MODE", "0") == "1"
# Global request cap sized for DICOM; per-type caps are enforced in the route.
app.config["MAX_CONTENT_LENGTH"] = dicom_utils.DICOM_MAX_BYTES + 2 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SECURE_COOKIES", "0") == "1"
app.config["RATELIMIT_ENABLED"] = os.environ.get("RATELIMIT_ENABLED", "1") == "1"

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
DICOM_EXTENSIONS = {"dcm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DICOM_EXTENSIONS

IMAGE_MAX_BYTES = 15 * 1024 * 1024
DEMO_MAX_BYTES = 5 * 1024 * 1024

# Registration keys for privileged roles (override via environment)
CLINICAL_SECURITY_KEY = os.environ.get("CLINICAL_KEY", "DOC-VVIT-2026")
ADMIN_SECURITY_KEY = os.environ.get("ADMIN_KEY", "ADM-VVIT-2026")

# No branding on PDF reports; generate_pdf renders a text-only header when None.
LOGO_PATH = None

MIN_PASSWORD_LENGTH = 8

# Simple in-memory login throttle: username -> list of failed-attempt times
_failed_logins = {}
MAX_ATTEMPTS = 5
LOCKOUT_WINDOW = 600  # seconds

SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")

csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

init_db()


# ---------------- SECURITY HEADERS ---------------- #
@app.before_request
def _new_csp_nonce():
    g.csp_nonce = secrets.token_urlsafe(16)


@app.context_processor
def _inject_globals():
    return {
        "csp_nonce": g.get("csp_nonce", ""),
        "demo_mode": app.config["DEMO_MODE"],
        "demo_accounts": app.config.get("DEMO_ACCOUNTS", []),
        "tier_text_for": tier_text_for,
    }


@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    # Scripts/styles/fonts are all served from 'self' (Chart.js + Inter are
    # vendored into static/ — no CDNs). Inline <script> blocks carry the
    # per-request nonce; style attributes need 'unsafe-inline' (styles only).
    # The only remote service is OpenStreetMap — the Facility Locator embeds
    # the OSM map (frame-src) and queries Nominatim for the directory
    # (connect-src). Google's keyless map embed is deprecated/unframable, so
    # the app uses OSM, which is free and needs no API key.
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{g.get('csp_nonce', '')}'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self' https://nominatim.openstreetmap.org; "
        "frame-src https://www.openstreetmap.org; "
        "object-src 'none'; base-uri 'self'; form-action 'self'; "
        "frame-ancestors 'none'"
    )
    return resp


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


def is_clinical():
    return session.get("role") in ("doctor", "admin")


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
@limiter.limit("20 per minute", methods=["POST"])
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
@limiter.limit("5 per minute", methods=["POST"])
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
@limiter.limit("5 per minute", methods=["POST"])
def register_clinical():
    """Hidden registration for clinical staff (doctor / admin via key)."""
    if app.config["DEMO_MODE"]:
        # Nobody mints real roles on the public demo; seeded accounts exist.
        flash("Clinical registration is disabled in the public demo. "
              "Use the demo doctor/admin accounts on the login page.", "error")
        return redirect(url_for("login"))

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
def _sample_gallery():
    """Bundled sample X-rays for the demo ('try a sample' strip)."""
    if not app.config["DEMO_MODE"] or not os.path.isdir(SAMPLES_DIR):
        return []
    samples = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png"):
            label = os.path.splitext(name)[0].replace("_", " ").replace("-", " ").title()
            samples.append({"file": name, "label": label})
    return samples


@app.route("/")
@login_required
def index():
    return render_template("index.html", samples=_sample_gallery())


# ---------------- PRIVATE MEDIA ---------------- #
def _scan_for_file(filename):
    """Find the scan record referencing a stored media filename."""
    with get_db_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM patients
            WHERE image_path = ? OR gradcam_path = ? OR pdf_path = ?
               OR cam_path = ? OR annotated_path = ?
            """,
            (filename,) * 5,
        ).fetchone()


def _can_view_scan(scan):
    return is_clinical() or scan["user_id"] == session.get("user_id")


@app.route("/media/<path:filename>")
@login_required
def media(filename):
    filename = os.path.basename(filename)  # defeat traversal
    scan = _scan_for_file(filename)
    if scan is None or not _can_view_scan(scan):
        abort(404)
    return send_from_directory(media_dir(), filename)


# ---------------- UPLOAD + ASYNC PIPELINE ---------------- #
def _file_size(file):
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    return size


def _validate_upload(file):
    """Returns (kind, safe_filename, error). kind is 'image' or 'dicom'."""
    if file is None or not file.filename:
        return None, None, "No file selected."

    name = secure_filename(file.filename)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return None, None, ("Unsupported file type. Please upload a JPG/PNG "
                            "X-ray image or a DICOM (.dcm) file.")

    size = _file_size(file)
    if app.config["DEMO_MODE"] and size > DEMO_MAX_BYTES:
        return None, None, "Demo uploads are limited to 5 MB."
    cap = dicom_utils.DICOM_MAX_BYTES if ext in DICOM_EXTENSIONS else IMAGE_MAX_BYTES
    if size > cap:
        limit_mb = cap // (1024 * 1024)
        return None, None, f"File too large — the limit is {limit_mb} MB for this format."

    kind = "dicom" if ext in DICOM_EXTENSIONS else "image"
    if kind == "image":
        # Confirm it is actually a decodable image, not just a renamed file.
        try:
            Image.open(file.stream).verify()
        except Exception:
            return None, None, "The uploaded file is not a valid image."
        file.stream.seek(0)

    unique = f"{uuid.uuid4().hex[:12]}_{name}"
    return kind, unique, None


def _store_upload(file, kind, file_name):
    """Persist the upload into MEDIA_DIR; DICOM converts to PNG (no PHI kept).

    Returns (stored_filename, dicom_meta or None, error or None).
    """
    if kind == "image":
        file.save(os.path.join(media_dir(), file_name))
        return file_name, None, None

    # DICOM: validate by parsing, convert, keep whitelisted tags, DELETE the
    # original .dcm — the patient-identifying module never touches disk twice.
    tmp_path = os.path.join(media_dir(), file_name)
    file.save(tmp_path)
    try:
        ds = dicom_utils.read_and_validate(tmp_path)
        meta = dicom_utils.safe_metadata(ds)
        png_name = os.path.splitext(file_name)[0] + ".png"
        dicom_utils.to_png(ds, os.path.join(media_dir(), png_name))
        return png_name, meta, None
    except dicom_utils.DicomValidationError as e:
        return None, None, str(e)
    except Exception:
        log.exception("DICOM conversion failed for %s", file_name)
        return None, None, "Could not process the DICOM file."
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _create_scan_and_job(stored_name, source_format, dicom_meta=None):
    body_part = (dicom_meta or {}).get("body_part")
    view_position = (dicom_meta or {}).get("view_position")
    with get_db_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO patients (image_path, date, user_id, source_format,
                                  body_part, view_position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                stored_name,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                session.get("user_id"),
                source_format,
                body_part,
                view_position,
            ),
        )
        scan_id = cur.lastrowid
        conn.commit()
    job_id = jobs.create_job(scan_id, session.get("user_id"))
    jobs.submit(job_id)
    return scan_id, job_id


def _upload_limit():
    return "3 per minute" if app.config["DEMO_MODE"] else "10 per minute"


@app.route("/predict", methods=["POST"])
@login_required
@limiter.limit(_upload_limit)
def predict_route():
    kind, file_name, error = _validate_upload(request.files.get("file"))
    if error:
        flash(error, "error")
        return redirect(url_for("index"))

    stored_name, dicom_meta, error = _store_upload(request.files["file"], kind, file_name)
    if error:
        flash(error, "error")
        return redirect(url_for("index"))

    _scan_id, job_id = _create_scan_and_job(
        stored_name, "dicom" if kind == "dicom" else "image", dicom_meta
    )
    log.info("Upload by user %s -> job %s (%s)", session.get("username"), job_id, kind)
    return redirect(url_for("processing", job_id=job_id))


@app.route("/predict-sample", methods=["POST"])
@login_required
@limiter.limit(_upload_limit)
def predict_sample():
    """Run the pipeline on a bundled sample image (demo gallery)."""
    samples = {s["file"] for s in _sample_gallery()}
    name = request.form.get("sample") or ""
    if name not in samples:
        abort(404)
    stored = f"{uuid.uuid4().hex[:12]}_{secure_filename(name)}"
    shutil.copyfile(os.path.join(SAMPLES_DIR, name), os.path.join(media_dir(), stored))
    _scan_id, job_id = _create_scan_and_job(stored, "sample")
    return redirect(url_for("processing", job_id=job_id))


def _job_or_404(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        abort(404)
    if not (is_clinical() or job["user_id"] == session.get("user_id")):
        abort(404)  # don't reveal existence
    return job


@app.route("/processing/<job_id>")
@login_required
def processing(job_id):
    job = _job_or_404(job_id)
    if job["status"] == "done":
        return redirect(url_for("results", scan_id=job["scan_id"]))
    return render_template("processing.html", job=job)


@app.route("/api/jobs/<job_id>")
@login_required
def job_status(job_id):
    job = _job_or_404(job_id)
    payload = {
        "id": job["id"],
        "status": job["status"],
        "error": job["error"],
        "scan_id": job["scan_id"],
    }
    if job["status"] == "done":
        payload["redirect"] = url_for("results", scan_id=job["scan_id"])
    elif job["status"] == "rejected":
        payload["message"] = predict.OOD_REJECT_MESSAGE
    return jsonify(payload)


# ---------------- RESULTS ---------------- #
def _get_scan_or_404(scan_id, clinical_ok=True):
    with get_db_connection() as conn:
        scan = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (scan_id,)
        ).fetchone()
    if scan is None:
        abort(404)
    allowed = scan["user_id"] == session.get("user_id") or (clinical_ok and is_clinical())
    if not allowed:
        abort(404)
    return scan


def _annotations_for(scan_id):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_annotations WHERE scan_id = ? ORDER BY id",
            (scan_id,),
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


@app.route("/results/<int:scan_id>")
@login_required
def results(scan_id):
    scan = _get_scan_or_404(scan_id)
    performance = get_model_performance()
    stage1_meta = performance.get("stage1") or {}
    top3 = json.loads(scan["top3"]) if scan["top3"] else []
    detections = json.loads(scan["detections"]) if scan["detections"] else None
    annotations = _annotations_for(scan_id)
    # Patients see doctor annotations only once the review is recorded.
    show_annotations = is_clinical() or bool(scan["reviewed_by"])
    return render_template(
        "result.html",
        scan=scan,
        top3=top3,
        detections=detections,
        annotations=annotations if show_annotations else [],
        recommendation=predict.recommendation_for(scan["verdict"], scan["severity"]),
        threshold=round(float(stage1_meta.get("threshold_calibrated",
                                              stage1_meta.get("threshold", 0.5))) * 100, 2),
        abstain_low=round(float(stage1_meta.get("abstain_low", 0.2)) * 100, 2),
        abstain_high=round(float(stage1_meta.get("abstain_high", 0.45)) * 100, 2),
        performance=performance,
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


# ---------------- ANNOTATIONS API ---------------- #
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_MAX_SHAPES = 200

_SHAPE_FIELDS = {
    "ellipse": ("cx", "cy", "rx", "ry"),
    "arrow": ("x1", "y1", "x2", "y2"),
    "path": (),
    "label": ("x", "y"),
}


def _valid_shape(shape):
    kind = shape.get("type")
    if kind not in _SHAPE_FIELDS:
        return False
    for field in _SHAPE_FIELDS[kind]:
        v = shape.get(field)
        if not isinstance(v, (int, float)) or not (-0.5 <= v <= 1.5):
            return False
    if kind == "path":
        pts = shape.get("points")
        if not isinstance(pts, list) or not (2 <= len(pts) <= 2000):
            return False
        if not all(isinstance(p, list) and len(p) == 2
                   and all(isinstance(c, (int, float)) and -0.5 <= c <= 1.5 for c in p)
                   for p in pts):
            return False
    if kind == "label" and not isinstance(shape.get("text", ""), str):
        return False
    color = shape.get("color", "#ff5252")
    if not isinstance(color, str) or not _HEX_COLOR.match(color):
        return False
    width = shape.get("width", 3)
    if not isinstance(width, (int, float)) or not (0.5 <= width <= 20):
        return False
    return True


@app.route("/api/scans/<int:scan_id>/annotations", methods=["GET", "POST"])
@login_required
def scan_annotations(scan_id):
    scan = _get_scan_or_404(scan_id)

    if request.method == "GET":
        annotations = _annotations_for(scan_id)
        if not is_clinical() and not scan["reviewed_by"]:
            annotations = []  # patient sees markup only after review
        return jsonify({"annotations": annotations})

    # POST — doctors/admins only
    if not is_clinical():
        abort(403)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Expected a JSON annotation document.")
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or len(shapes) > _MAX_SHAPES:
        abort(400, description="Invalid shapes list.")
    if not all(isinstance(s, dict) and _valid_shape(s) for s in shapes):
        abort(400, description="Invalid shape data.")

    doc = {
        "image_w": int(data.get("image_w") or 0),
        "image_h": int(data.get("image_h") or 0),
        "shapes": [
            {**s, "text": str(s.get("text", ""))[:120]} if s.get("type") == "label" else s
            for s in shapes
        ],
    }
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO scan_annotations (scan_id, doctor_id, data) VALUES (?, ?, ?)",
            (scan_id, session["user_id"], json.dumps(doc)),
        )
        conn.commit()
        annotation_id = cur.lastrowid
    log.info("Annotation %s saved on scan %s by %s",
             annotation_id, scan_id, session.get("username"))
    return jsonify({"ok": True, "id": annotation_id})


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
def _regenerate_pdf_for(row, final_diagnosis, notes):
    """Re-issue the PDF with the doctor's review + flattened annotations."""
    original_abs = os.path.join(media_dir(), row["image_path"] or "")
    annotated_name = None
    annotations = _annotations_for(row["id"])
    if annotations and os.path.exists(original_abs):
        annotated_name = f"{os.path.splitext(row['image_path'])[0]}_annotated.png"
        try:
            if not flatten_annotations(
                original_abs, annotations, os.path.join(media_dir(), annotated_name)
            ):
                annotated_name = None
        except Exception:
            log.exception("Annotation flattening failed for scan %s", row["id"])
            annotated_name = None

    pdf_name = row["pdf_path"] or f"{os.path.splitext(row['image_path'])[0]}_report.pdf"
    detections = json.loads(row["detections"]) if row["detections"] else None
    generate_pdf(
        report_path=os.path.join(media_dir(), pdf_name),
        original_img_path=original_abs,
        gradcam_img_path=(os.path.join(media_dir(), row["gradcam_path"])
                          if row["gradcam_path"] else None),
        result=row["result"],
        confidence=row["confidence"],
        severity=row["severity"],
        fracture_prob=row["fracture_prob"],
        top3=json.loads(row["top3"]) if row["top3"] else None,
        logo_path=LOGO_PATH,
        record_id=row["id"],
        doctor_diagnosis=final_diagnosis,
        doctor_notes=notes,
        verdict=row["verdict"],
        tier_text=tier_text_for(row["tier"]) if row["tier"] else None,
        annotated_img_path=(os.path.join(media_dir(), annotated_name)
                            if annotated_name else None),
        detections=detections,
        body_part=row["body_part"],
        view_position=row["view_position"],
    )
    return annotated_name, pdf_name


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

        if row and row["image_path"]:
            try:
                annotated_name, pdf_name = _regenerate_pdf_for(row, final_diagnosis, notes)
                with get_db_connection() as conn:
                    conn.execute(
                        "UPDATE patients SET annotated_path = ?, pdf_path = ? WHERE id = ?",
                        (annotated_name, pdf_name, row["id"]),
                    )
                    conn.commit()
            except Exception:
                log.exception("PDF regeneration failed for record %s", patient_id)

        flash(f"Review saved for record #{patient_id}.", "success")
        return redirect(url_for("doctor_dashboard"))

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.*, u.username AS patient_name
            FROM patients p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY (p.review_status = 'PENDING') DESC, p.id DESC
            """
        ).fetchall()
        ann_rows = conn.execute(
            "SELECT scan_id, data FROM scan_annotations ORDER BY id"
        ).fetchall()

    ann_map = {}
    for r in ann_rows:
        ann_map.setdefault(r["scan_id"], []).append(json.loads(r["data"]))
    cases = []
    for row in rows:
        case = dict(row)
        case["annotations_json"] = (
            json.dumps(ann_map[row["id"]]) if row["id"] in ann_map else None
        )
        cases.append(case)
    return render_template("doctor_dashboard.html", cases=cases)


# ---------------- ADMIN ANALYTICS ---------------- #
@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db_connection() as conn:
        total_scans = conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]
        fractures = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE result LIKE 'Fracture%'"
        ).fetchone()["c"]
        uncertain = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE verdict = 'uncertain'"
        ).fetchone()["c"]
        rejected = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE verdict = 'rejected'"
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
            SELECT p.id, p.result, p.severity, p.date, p.verdict, p.tier,
                   u.username AS patient_name
            FROM patients p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC LIMIT 8
            """
        ).fetchall()

    return render_template(
        "admin_dashboard.html",
        total_scans=total_scans,
        fractures=fractures,
        no_fractures=total_scans - fractures - uncertain - rejected,
        uncertain=uncertain,
        rejected=rejected,
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
    stats = ood_gate.load_stats()
    ood_info = None
    if stats:
        ood_info = {
            "threshold": round(stats["threshold"], 4),
            "k": stats["k"],
            "percentile": stats["percentile"],
            "refs": len(stats["refs"]),
        }
    return render_template(
        "model_info.html", performance=get_model_performance(), ood_info=ood_info
    )


# ---------------- ERROR HANDLERS ---------------- #
@app.errorhandler(413)
def too_large(_):
    flash("File too large — images up to 15 MB, DICOM up to 50 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(400)
def bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": getattr(e, "description", "Bad request")}), 400
    return render_template("error.html", code=400,
                           message=getattr(e, "description", "Bad request.")), 400


@app.errorhandler(403)
def forbidden(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Forbidden"}), 403
    return render_template("error.html", code=403,
                           message="You do not have access to this resource."), 403


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("error.html", code=404,
                           message="The page you requested does not exist."), 404


@app.errorhandler(429)
def rate_limited(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Too many requests — slow down."}), 429
    flash("Too many requests — please wait a moment and try again.", "error")
    return redirect(url_for("index"))


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500,
                           message="Something went wrong on our side."), 500


# ---------------- DEMO MODE ---------------- #
def _setup_demo_mode():
    """Seeded accounts + periodic wipe for the public demo."""
    from demo.seed_demo import seed_demo_accounts

    app.config["DEMO_ACCOUNTS"] = seed_demo_accounts()
    log.info("DEMO MODE: seeded %d demo accounts", len(app.config["DEMO_ACCOUNTS"]))

    def wipe_loop():
        while True:
            time.sleep(24 * 3600)
            try:
                wipe_demo_data()
            except Exception:
                log.exception("Demo wipe failed")

    threading.Thread(target=wipe_loop, daemon=True, name="demo-wipe").start()


def wipe_demo_data():
    """Delete all scans, jobs, annotations and media files (demo hygiene)."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM scan_annotations")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM patients")
        conn.commit()
    removed = 0
    for name in os.listdir(media_dir()):
        try:
            os.remove(os.path.join(media_dir(), name))
            removed += 1
        except OSError:
            pass
    log.info("Demo wipe: cleared scan tables and %d media files", removed)


if app.config["DEMO_MODE"]:
    _setup_demo_mode()


# ---------------- MODEL WARMUP ---------------- #
# gunicorn runs 1 worker with N threads: load models once, up front, off the
# request path (PRELOAD_MODELS=1 is set in the Docker image).
if os.environ.get("PRELOAD_MODELS", "0") == "1":
    threading.Thread(target=predict.warm_models, daemon=True, name="model-warmup").start()


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=debug, port=port)
