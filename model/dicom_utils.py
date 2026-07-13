"""DICOM ingestion: validation, correct 8-bit conversion, and a strict no-PHI policy.

X-rays leave machines as DICOM. This module:
  * verifies a file really is DICOM by parsing it (never by extension);
  * accepts only plain radiographs (Modality CR / DX / DR);
  * converts pixel data correctly — the two classic pitfalls are the VOI LUT
    (windowing) and MONOCHROME1 (inverted: air bright, bone dark);
  * extracts ONLY whitelisted, non-identifying tags (body part, view,
    modality, bit depth). The caller must delete the original .dcm right
    after conversion — no PHI is ever persisted (documented in the README).
"""

import numpy as np

try:
    import pydicom
    from pydicom.pixels import apply_voi_lut
except ImportError:  # pragma: no cover - pydicom < 3 fallback
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

ALLOWED_MODALITIES = {"CR", "DX", "DR"}

# 50 MB — DICOMs are larger than JPEGs; enforced by the upload route.
DICOM_MAX_BYTES = 50 * 1024 * 1024


class DicomValidationError(ValueError):
    """User-facing rejection reason (message is safe to show)."""


def read_and_validate(path_or_file):
    """Parse a DICOM file and enforce the radiograph-only policy.

    Returns the parsed dataset. Raises DicomValidationError with a
    user-friendly message on any problem.
    """
    try:
        ds = pydicom.dcmread(path_or_file)
    except Exception:
        raise DicomValidationError(
            "The uploaded file is not a valid DICOM file."
        ) from None

    modality = str(getattr(ds, "Modality", "") or "").upper()
    if modality not in ALLOWED_MODALITIES:
        raise DicomValidationError(
            "Only plain radiographs (CR/DX) are supported — this DICOM's "
            f"modality is '{modality or 'unknown'}'."
        )

    if not hasattr(ds, "PixelData"):
        raise DicomValidationError("The DICOM file contains no image data.")
    return ds


def to_8bit_array(ds):
    """Correctly convert DICOM pixel data to an 8-bit grayscale array.

    Applies the VOI LUT (the windowing radiologists actually view with) and
    inverts MONOCHROME1 so bone is always bright.
    """
    try:
        arr = apply_voi_lut(ds.pixel_array, ds).astype("float32")
    except Exception as e:
        raise DicomValidationError(
            "Could not decode the DICOM pixel data. If the file uses an "
            "unusual compression, try exporting it as JPG/PNG instead."
        ) from e

    if arr.ndim == 3:  # some CR files carry an extra frame/channel dim
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr  # invert: bone should be bright

    arr -= arr.min()
    arr /= max(float(np.ptp(arr)), 1e-6)
    return (arr * 255).astype("uint8")


def to_png(ds, out_path):
    """Convert a validated dataset to PNG at out_path. Returns out_path."""
    from PIL import Image

    arr = to_8bit_array(ds)
    Image.fromarray(arr, mode="L").save(out_path)
    return out_path


def safe_metadata(ds):
    """The ONLY DICOM tags this application persists — all non-identifying.

    Everything else (patient module, dates, institution, device serials)
    is discarded along with the original file.
    """
    def _clean(value):
        s = str(value or "").strip()
        return s[:64] if s else None

    bits = getattr(ds, "BitsStored", None)
    return {
        "body_part": _clean(getattr(ds, "BodyPartExamined", None)),
        "view_position": _clean(getattr(ds, "ViewPosition", None)),
        "modality": _clean(getattr(ds, "Modality", None)),
        "bits_stored": int(bits) if bits else None,
    }
