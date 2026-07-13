"""SQLite access + idempotent schema migrations.

Every schema change in this project goes through init_db() below — columns
are added with ALTER TABLE only when missing, tables use CREATE TABLE IF
NOT EXISTS, and data migrations (like the static/uploads → media move) are
written to be safe to run repeatedly.

Paths are resolved from environment variables on every call so the same
code runs bare-metal, in Docker (named volumes), and in tests (temp dirs):
  DATABASE_PATH — SQLite file (default: ./database.db)
  MEDIA_DIR     — private upload/report storage (default: ./media)
"""

import logging
import os
import shutil
import sqlite3
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger("fracture-app.db")


def db_path():
    return os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "database.db"))


def media_dir():
    return os.environ.get("MEDIA_DIR", os.path.join(BASE_DIR, "media"))


@contextmanager
def get_db_connection():
    os.makedirs(os.path.dirname(os.path.abspath(db_path())), exist_ok=True)
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def _existing_columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    os.makedirs(media_dir(), exist_ok=True)
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

        # Async job tracking (UPGRADE_PLAN 2.1)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id         TEXT PRIMARY KEY,
            scan_id    INTEGER REFERENCES patients(id),
            user_id    INTEGER REFERENCES users(id),
            status     TEXT NOT NULL DEFAULT 'queued',
            error      TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT
        )
        """)

        # Doctor annotations, vector-first (UPGRADE_PLAN 2.4)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_annotations (
            id         INTEGER PRIMARY KEY,
            scan_id    INTEGER NOT NULL REFERENCES patients(id),
            doctor_id  INTEGER NOT NULL REFERENCES users(id),
            data       TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Idempotent column migrations for databases created by older versions.
        wanted = {
            "user_id": "INTEGER",
            "review_status": "TEXT DEFAULT 'NONE'",
            "gradcam_path": "TEXT",
            "pdf_path": "TEXT",
            "fracture_prob": "REAL",
            "reviewed_by": "TEXT",
            "reviewed_at": "TEXT",
            # Phase 1: calibrated verdicts + tiers
            "verdict": "TEXT",
            "tier": "TEXT",
            "reject_reason": "TEXT",
            # Phase 2: interactive viewer + annotations
            "cam_path": "TEXT",
            "annotated_path": "TEXT",
            # Phase 3: DICOM metadata (non-identifying only) + localization
            "body_part": "TEXT",
            "view_position": "TEXT",
            "source_format": "TEXT",
            "detections": "TEXT",
            "top3": "TEXT",
        }
        have = _existing_columns(conn, "patients")
        for column, decl in wanted.items():
            if column not in have:
                conn.execute(f"ALTER TABLE patients ADD COLUMN {column} {decl}")
                log.info("DB migration: added patients.%s", column)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_user ON patients(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_review ON patients(review_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_scan ON jobs(scan_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_scan ON scan_annotations(scan_id)")
        conn.commit()

        _migrate_uploads_to_media(conn)


def _migrate_uploads_to_media(conn):
    """One-time move of files out of the publicly served static/uploads/.

    Old rows stored paths like 'static/uploads/<file>'; those files were
    reachable by anyone with the URL. Move every referenced file into the
    private MEDIA_DIR and rewrite the DB paths to bare filenames (served
    through the authorizing /media/<filename> route). Safe to re-run: rows
    already holding bare filenames are ignored.
    """
    prefix = "static/uploads/"
    columns = ("image_path", "gradcam_path", "pdf_path", "cam_path", "annotated_path")
    rows = conn.execute(
        "SELECT id, {} FROM patients".format(", ".join(columns))
    ).fetchall()

    moved = 0
    for row in rows:
        updates = {}
        for col in columns:
            value = row[col]
            if not value or not value.startswith(prefix):
                continue
            filename = os.path.basename(value)
            src = os.path.join(BASE_DIR, value.replace("/", os.sep))
            dst = os.path.join(media_dir(), filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
            updates[col] = filename
        if updates:
            sets = ", ".join(f"{c} = ?" for c in updates)
            conn.execute(
                f"UPDATE patients SET {sets} WHERE id = ?",
                (*updates.values(), row["id"]),
            )
    if moved:
        conn.commit()
        log.info("Media migration: moved %d files from static/uploads to %s",
                 moved, media_dir())
    elif rows and any(
        row[c] and row[c].startswith(prefix) for row in rows for c in columns
    ):
        conn.commit()  # paths rewritten even if files were already moved/missing
