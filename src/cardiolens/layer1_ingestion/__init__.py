"""Layer 1 — ingestion and normalisation of input into the unified signal (TZ §4)."""

from cardiolens.layer1_ingestion.digitize import (
    degrade_image,
    digitize_image,
    digitize_page,
    digitize_page_to_signal,
    page_to_signal,
    render_ecg_image,
    render_ecg_page,
)
from cardiolens.layer1_ingestion.parsers import (
    load,
    load_csv,
    load_edf,
    load_from_image,
    load_wfdb,
)
from cardiolens.layer1_ingestion.ptbxl import load_ptbxl_record
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg

__all__ = [
    "degrade_image",
    "digitize_image",
    "digitize_page",
    "digitize_page_to_signal",
    "generate_synthetic_ecg",
    "load",
    "load_csv",
    "load_edf",
    "load_from_image",
    "load_ptbxl_record",
    "load_wfdb",
    "page_to_signal",
    "render_ecg_image",
    "render_ecg_page",
]
