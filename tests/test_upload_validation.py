"""Upload validation: extension/content/size checks, UUID storage names."""

import io
import os
import re

from conftest import job_id_from, latest_scan, make_png_bytes, upload_xray

import db


def _flash_after(client, resp):
    assert resp.status_code == 302
    page = client.get(resp.headers["Location"])
    return page.data.decode()


def test_rejects_missing_file(patient_client):
    resp = patient_client.post("/predict", data={},
                               content_type="multipart/form-data")
    assert "No file selected" in _flash_after(patient_client, resp)


def test_rejects_disallowed_extension(patient_client):
    resp = upload_xray(patient_client, filename="scan.gif")
    assert "Unsupported file type" in _flash_after(patient_client, resp)


def test_rejects_nonimage_bytes_with_image_extension(patient_client):
    junk = io.BytesIO(b"this is definitely not a PNG image, just text bytes")
    resp = upload_xray(patient_client, filename="renamed.png", data=junk)
    assert "not a valid image" in _flash_after(patient_client, resp)


def test_rejects_oversize_image(patient_client):
    big = io.BytesIO(b"\x89PNG\r\n" + b"\0" * (16 * 1024 * 1024))
    resp = upload_xray(patient_client, filename="huge.jpg", data=big)
    assert "too large" in _flash_after(patient_client, resp).lower()


def test_accepts_valid_png_with_uuid_name(patient_client, stub_pipeline):
    resp = upload_xray(patient_client, filename="my xray.png")
    job_id_from(resp)
    scan = latest_scan()
    # UUID-prefixed, sanitized storage name
    assert re.match(r"^[0-9a-f]{12}_my_xray\.png$", scan["image_path"])
    assert os.path.exists(os.path.join(db.media_dir(), scan["image_path"]))


def test_second_upload_never_overwrites_first(patient_client, stub_pipeline):
    upload_xray(patient_client, filename="xray.png",
                data=make_png_bytes(color=(10, 10, 10)))
    first = latest_scan()["image_path"]
    upload_xray(patient_client, filename="xray.png",
                data=make_png_bytes(color=(240, 240, 240)))
    second = latest_scan()["image_path"]
    assert first != second
    assert os.path.exists(os.path.join(db.media_dir(), first))
    assert os.path.exists(os.path.join(db.media_dir(), second))


def test_upload_requires_login(client):
    resp = upload_xray(client)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
