"""DICOM: parse-based validation, correct 8-bit conversion (VOI LUT +
MONOCHROME1), modality whitelist, and the strict no-PHI upload flow."""

import io
import os

import numpy as np
import pydicom
import pytest
from conftest import latest_scan
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import db
from model import dicom_utils


def make_dicom(path, modality="DX", photometric="MONOCHROME2",
               body_part="WRIST", view="PA", with_patient=True):
    """Synthesize a small radiograph-like DICOM with a horizontal gradient."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = 64, 64
    ds.SamplesPerPixel = 1
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    ds.BodyPartExamined = body_part
    ds.ViewPosition = view
    if with_patient:  # PHI that must never be persisted
        ds.PatientName = "Doe^Jane"
        ds.PatientID = "SECRET-123"

    # gradient: dark on the left, bright on the right (values 0..4095)
    arr = np.tile(np.linspace(0, 4095, 64, dtype=np.uint16), (64, 1))
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


# ---------------- unit level ----------------
def test_valid_dx_accepted(tmp_path):
    p = make_dicom(tmp_path / "ok.dcm")
    ds = dicom_utils.read_and_validate(str(p))
    assert ds.Modality == "DX"


def test_non_radiograph_modality_rejected(tmp_path):
    p = make_dicom(tmp_path / "ct.dcm", modality="CT")
    with pytest.raises(dicom_utils.DicomValidationError, match="CR/DX"):
        dicom_utils.read_and_validate(str(p))


def test_non_dicom_bytes_rejected(tmp_path):
    p = tmp_path / "junk.dcm"
    p.write_bytes(b"not dicom at all" * 100)
    with pytest.raises(dicom_utils.DicomValidationError, match="not a valid DICOM"):
        dicom_utils.read_and_validate(str(p))


def test_monochrome2_conversion_orientation(tmp_path):
    ds = dicom_utils.read_and_validate(str(make_dicom(tmp_path / "m2.dcm")))
    arr = dicom_utils.to_8bit_array(ds)
    assert arr.dtype == np.uint8
    assert arr[0, 0] < arr[0, -1]  # gradient preserved: bright on the right
    assert arr.max() == 255 and arr.min() == 0


def test_monochrome1_is_inverted(tmp_path):
    p = make_dicom(tmp_path / "m1.dcm", photometric="MONOCHROME1")
    ds = dicom_utils.read_and_validate(str(p))
    arr = dicom_utils.to_8bit_array(ds)
    assert arr[0, 0] > arr[0, -1]  # MONOCHROME1: raw bright means dark -> inverted


def test_safe_metadata_is_whitelist_only(tmp_path):
    ds = dicom_utils.read_and_validate(str(make_dicom(tmp_path / "m.dcm")))
    meta = dicom_utils.safe_metadata(ds)
    assert meta == {"body_part": "WRIST", "view_position": "PA",
                    "modality": "DX", "bits_stored": 12}
    assert "SECRET" not in str(meta) and "Doe" not in str(meta)


# ---------------- upload route integration (stubbed pipeline) ----------------
def _upload_dcm(client, path, name="scan.dcm"):
    with open(path, "rb") as f:
        data = io.BytesIO(f.read())
    return client.post("/predict", data={"file": (data, name)},
                       content_type="multipart/form-data")


def test_dicom_upload_end_to_end_no_phi(tmp_path, patient_client, stub_pipeline):
    p = make_dicom(tmp_path / "upload.dcm")
    resp = _upload_dcm(patient_client, p)
    assert resp.status_code == 302 and "/processing/" in resp.headers["Location"]

    scan = latest_scan()
    assert scan["source_format"] == "dicom"
    assert scan["body_part"] == "WRIST"
    assert scan["view_position"] == "PA"
    assert scan["image_path"].endswith(".png")

    media_files = os.listdir(db.media_dir())
    assert not any(f.endswith(".dcm") for f in media_files), \
        "original DICOM (with PHI) must be deleted after conversion"
    # the converted PNG carries no metadata at all (it's raw pixels)
    assert scan["image_path"] in media_files


def test_ct_dicom_upload_rejected_with_message(tmp_path, patient_client,
                                               stub_pipeline):
    p = make_dicom(tmp_path / "ct.dcm", modality="CT")
    resp = _upload_dcm(patient_client, p)
    assert resp.status_code == 302
    page = patient_client.get(resp.headers["Location"])
    assert b"Only plain radiographs" in page.data
    assert latest_scan() is None  # no scan row was created
    assert not any(f.endswith(".dcm") for f in os.listdir(db.media_dir()))
