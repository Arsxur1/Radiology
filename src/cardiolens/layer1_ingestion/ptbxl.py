"""PTB-XL loader hook (TZ §15).

PTB-XL is the primary clinical benchmark (12-lead, SCP-ECG, ages 0-95). We do
NOT bundle it (it is large and license-gated on PhysioNet); instead this hook
loads a record from a local PTB-XL checkout. See README for how to obtain it.
The synthetic generator covers offline/deterministic testing.
"""

from __future__ import annotations

from pathlib import Path

from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer1_ingestion.parsers import load_wfdb

# PTB-XL records are WFDB; lead order is the standard 12-lead set.
PTBXL_DOWNLOAD_URL = "https://physionet.org/content/ptb-xl/"


def load_ptbxl_record(record_path: str | Path) -> SignalRepresentation:
    """Load one PTB-XL WFDB record from a local path.

    ``record_path`` points at a record name without extension, e.g.
    ``<PTBXL_ROOT>/records500/00000/00001_hr``.
    """
    record_path = Path(record_path)
    if not (record_path.with_suffix(".hea")).exists():
        raise FileNotFoundError(
            f"PTB-XL record not found: {record_path}.hea\n"
            f"Download PTB-XL from {PTBXL_DOWNLOAD_URL} and point at a record "
            f"(see README). For offline runs use the synthetic generator."
        )
    sig = load_wfdb(record_path)
    sig.source = "wfdb"
    return sig
