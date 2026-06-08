from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import json
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from contextlib import contextmanager

from model.predict import predict_fracture, generate_pdf, get_model_performance

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
app.secret_key = "super_secret_final_project_key_for_authentication"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- SECURITY CONFIG ---------------- #
# This key is required for Doctor registration via the hidden URL
CLINICAL_SECURITY_KEY = "DOC-VVIT-2026"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

LOGO_PATH = os.path.join("static", "images", "project_logo.png")


# ---------------- DATABASE INIT ---------------- #
@contextmanager
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
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
            user_id INTEGER,
            review_status TEXT DEFAULT 'NONE'
        )
        """)

        # Safe migration: add columns to existing databases
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN review_status TEXT DEFAULT 'NONE'")
        except Exception:
            pass

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """)

        conn.commit()

init_db()


# ---------------- AUTHENTICATION WRAPPER ---------------- #
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'doctor':
            flash("Unauthorized Access: This section is restricted to medical personnel.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- AUTH ROUTES ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Public Registration - Patients Only"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Role is forced to patient for public registration
        role = 'patient'
        
        with get_db_connection() as conn:
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            
            if existing:
                flash("Username already exists.", "error")
            else:
                conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                             (username, generate_password_hash(password), role))
                conn.commit()
                flash("Patient registration successful. Please login.", "success")
                return redirect(url_for('login'))
        
    return render_template("register.html")

@app.route("/register-clinical", methods=["GET", "POST"])
def register_clinical():
    """Hidden Registration - Clinical Staff Only"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        security_key = request.form.get("security_key")
        
        # Verify Clinical Security Key
        if security_key != CLINICAL_SECURITY_KEY:
            flash("Invalid Clinical Security Key. Access Denied.", "error")
            return render_template("register_clinical.html")
            
        role = 'doctor'
        
        with get_db_connection() as conn:
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            
            if existing:
                flash("Username already exists.", "error")
            else:
                conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                             (username, generate_password_hash(password), role))
                conn.commit()
                flash("Clinical registration successful. Please login.", "success")
                return redirect(url_for('login'))
        
    return render_template("register_clinical.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------- HOME ---------------- #
@app.route("/")
@login_required
def index():
    return render_template("index.html")


# ---------------- PREDICT ---------------- #
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    file_name = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)

    # 🔥 Prediction
    try:
        result_data = predict_fracture(file_path)
    except Exception as e:
        flash(f"Error during diagnostic scanning: {str(e)}", "error")
        return redirect(url_for('index'))

    # Grad-CAM fix
    gradcam_path = result_data.get("gradcam")
    if gradcam_path and os.path.exists(gradcam_path):
        gradcam_path = gradcam_path.replace("\\", "/")
    else:
        gradcam_path = None

    # PDF
    try:
        report_filename = file_name.rsplit(".", 1)[0] + "_report.pdf"
        report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)

        generate_pdf(
            report_path=report_path,
            original_img_path=file_path,
            gradcam_img_path=gradcam_path,
            result=result_data.get("result"),
            confidence=result_data.get("confidence"),
            severity=result_data.get("severity"),
            logo_path=LOGO_PATH
        )
    except Exception:
        report_filename = None

    # 💾 SAVE TO DB
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO patients (image_path, result, confidence, severity, date, user_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file_path.replace("\\", "/"),
            result_data.get("result"),
            float(result_data.get("confidence", 0)),
            result_data.get("severity"),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            session.get("user_id")
        ))

        conn.commit()

    # Performance
    performance = get_model_performance()

    return render_template(
        "result.html",
        result=result_data.get("result"),
        confidence=result_data.get("confidence"),
        image_path=file_path.replace("\\", "/"),
        gradcam_path=gradcam_path,
        severity=result_data.get("severity"),
        recommendation=result_data.get("recommendation"),
        pdf_report=report_filename,
        epochs=performance["epochs"],
        accuracy=performance["accuracy"],
        val_accuracy=performance["val_accuracy"]
    )


# ---------------- PATIENT DASHBOARD ---------------- #
@app.route("/patient")
@login_required
def patient_dashboard():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Fetch ALL scans that belong to the currently logged-in patient
        cursor.execute(
            "SELECT * FROM patients WHERE user_id = ? ORDER BY id DESC",
            (session.get("user_id"),)
        )
        records = cursor.fetchall()

    if records:
        return render_template("patient_dashboard.html", records=records)
    else:
        return render_template("patient_dashboard.html", no_data=True)


# ---------------- REQUEST REVIEW ---------------- #
@app.route("/request-review/<int:record_id>", methods=["POST"])
@login_required
def request_review(record_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Verify the scan belongs to the user
        cursor.execute("SELECT user_id FROM patients WHERE id = ?", (record_id,))
        record = cursor.fetchone()
        
        if record and record['user_id'] == session.get('user_id'):
            cursor.execute("UPDATE patients SET review_status = 'PENDING' WHERE id = ?", (record_id,))
            conn.commit()
            flash("Specialist review requested successfully.", "success")
        else:
            flash("Unauthorized request.", "error")
            
    return redirect(url_for('patient_dashboard'))


# ---------------- HISTORY (DB BASED) ---------------- #
@app.route("/history")
@doctor_required
def history():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients ORDER BY id DESC")
        data = cursor.fetchall()

    return render_template("patient_history.html", data=data)


# ---------------- DOCTOR DASHBOARD ---------------- #
@app.route("/doctor", methods=["GET", "POST"])
@doctor_required
def doctor_dashboard():
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if request.method == "POST":
            patient_id = request.form.get("id")
            notes = request.form.get("notes")
            decision = request.form.get("decision")
            final_diagnosis = request.form.get("final_diagnosis")

            cursor.execute("""
            UPDATE patients
            SET notes=?, decision=?, final_diagnosis=?
            WHERE id=?
            """, (notes, decision, final_diagnosis, patient_id))

            conn.commit()

        cursor.execute("SELECT * FROM patients ORDER BY id DESC")
        data = cursor.fetchall()

    return render_template("doctor_dashboard.html", cases=data)


# ---------------- HOSPITALS ---------------- #
@app.route("/hospitals")
@login_required
def hospitals():
    if session.get('role') == 'doctor':
        flash("Facility Locator is primarily for Patient use. Access restricted for clinical roles.", "error")
        return redirect(url_for('index'))
    return render_template("hospitals.html")


# ---------------- LEARN PAGE ---------------- #
@app.route("/learn")
@login_required
def learn():
    return render_template("learn.html")


# ---------------- MODEL INFO ---------------- #
@app.route("/model-info")
@login_required
def model_info():
    return render_template("model_info.html")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(debug=True)