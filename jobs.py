"""Async inference: a ThreadPoolExecutor + jobs table.

Right-sized for a single-instance SQLite app (no Celery/Redis): the upload
route inserts a scan + job row and submits run_pipeline(); the browser polls
GET /api/jobs/<id> and follows the status through the pipeline stages.
Each worker thread opens its OWN SQLite connection — connections are never
shared across threads. Works under gunicorn's `-w 1 --threads N` model.
"""

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from db import get_db_connection, media_dir
from model import predict
from model.report import generate_pdf

log = logging.getLogger("fracture-app.jobs")

# queued | preprocessing | ood_check | stage1 | stage2 | localizing |
# explaining | reporting | done | failed | rejected
TERMINAL_STATUSES = {"done", "failed", "rejected"}

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")


def create_job(scan_id, user_id):
    job_id = uuid.uuid4().hex
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (id, scan_id, user_id, status) VALUES (?, ?, ?, 'queued')",
            (job_id, scan_id, user_id),
        )
        conn.commit()
    return job_id


def get_job(job_id):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def update_job(job_id, status, error=None):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error,
             datetime.now(UTC).isoformat(timespec="seconds"), job_id),
        )
        conn.commit()


def submit(job_id):
    """Queue the pipeline. SYNC_JOBS=1 (tests/CI) runs it inline instead."""
    if os.environ.get("SYNC_JOBS") == "1":
        run_pipeline(job_id)
    else:
        _executor.submit(run_pipeline, job_id)


def _basename(path):
    return os.path.basename(path) if path else None


def run_pipeline(job_id):
    """Full pipeline for one scan; runs on a worker thread."""
    try:
        with get_db_connection() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            scan = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (job["scan_id"],)
            ).fetchone() if job else None
        if job is None or scan is None:
            log.error("Job %s: job or scan row missing", job_id)
            if job is not None:
                update_job(job_id, "failed", "Scan record missing.")
            return

        image_path = os.path.join(media_dir(), scan["image_path"])

        def status_cb(stage):
            update_job(job_id, stage)

        result = predict.predict_fracture(image_path, status_cb=status_cb)

        if result.get("rejected"):
            with get_db_connection() as conn:
                conn.execute(
                    """
                    UPDATE patients SET result = ?, verdict = 'rejected',
                        reject_reason = ?, severity = NULL,
                        confidence = 0, fracture_prob = NULL
                    WHERE id = ?
                    """,
                    ("Rejected — not an X-ray", result.get("reject_reason"), scan["id"]),
                )
                conn.commit()
            log.info("Job %s: OOD-rejected (%s, details=%s)",
                     job_id, result.get("reject_reason"), result.get("ood"))
            update_job(job_id, "rejected")
            return

        detections = result.get("detections")
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE patients SET
                    result = ?, confidence = ?, severity = ?, fracture_prob = ?,
                    verdict = ?, tier = ?, gradcam_path = ?, cam_path = ?,
                    detections = ?, top3 = ?,
                    review_status = CASE WHEN ? = 'uncertain' AND review_status = 'NONE'
                                         THEN 'PENDING' ELSE review_status END
                WHERE id = ?
                """,
                (
                    result.get("result"),
                    float(result.get("confidence") or 0),
                    result.get("severity"),
                    float(result.get("fracture_prob") or 0),
                    result.get("verdict"),
                    result.get("tier"),
                    _basename(result.get("gradcam")),
                    _basename(result.get("cam_overlay")),
                    json.dumps(detections) if detections else None,
                    json.dumps(result.get("top3")) if result.get("top3") else None,
                    result.get("verdict"),
                    scan["id"],
                ),
            )
            conn.commit()

        # PDF report
        update_job(job_id, "reporting")
        report_filename = f"{os.path.splitext(scan['image_path'])[0]}_report.pdf"
        report_path = os.path.join(media_dir(), report_filename)
        try:
            generate_pdf(
                report_path=report_path,
                original_img_path=image_path,
                gradcam_img_path=result.get("gradcam"),
                result=result.get("result"),
                confidence=result.get("confidence"),
                severity=result.get("severity"),
                recommendation=result.get("recommendation"),
                fracture_prob=result.get("fracture_prob"),
                top3=result.get("top3"),
                record_id=scan["id"],
                verdict=result.get("verdict"),
                tier_text=result.get("tier_text"),
                threshold=result.get("threshold"),
                abstain_low=result.get("abstain_low"),
                abstain_high=result.get("abstain_high"),
                type_unclear=result.get("type_unclear", False),
                detections=detections,
                body_part=scan["body_part"],
                view_position=scan["view_position"],
            )
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE patients SET pdf_path = ? WHERE id = ?",
                    (report_filename, scan["id"]),
                )
                conn.commit()
        except Exception:
            log.exception("PDF generation failed for scan %s", scan["id"])

        log.info("Job %s: scan %s done — %s (verdict %s)",
                 job_id, scan["id"], result.get("result"), result.get("verdict"))
        update_job(job_id, "done")
    except Exception as e:
        log.exception("Job %s failed", job_id)
        try:
            update_job(job_id, "failed", str(e)[:500])
        except Exception:
            log.exception("Job %s: could not record failure", job_id)
