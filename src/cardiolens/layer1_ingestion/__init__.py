"""Layer 1 — ingestion and normalisation of input into the unified signal (TZ §4)."""

from cardiolens.layer1_ingestion.digitize import (
    DigitizedPage,
    digitize_image,
    digitize_page,
    render_ecg_image,
)
from cardiolens.layer1_ingestion.parsers import (
    load,
    load_csv,
    load_dicom,
    load_edf,
    load_from_image,
    load_wfdb,
)
from cardiolens.layer1_ingestion.ptbxl import load_ptbxl_record
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg

__all__ = [
    "DigitizedPage",
    "digitize_image",
    "digitize_page",
    "generate_synthetic_ecg",
    "load",
    "load_csv",
    "load_dicom",
    "load_edf",
    "load_from_image",
    "load_ptbxl_record",
    "load_wfdb",
    "render_ecg_image",
]
