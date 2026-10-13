"""QRS axis from the six limb leads (D49) and the label plumbing it depends on."""

import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline
from cardiolens.validation.harness import (
    PTBXL_SCP_MAP,
    parse_dx_from_header,
    parse_patient_from_header,
    ptbxl_axis_labels,
)

CHALLENGE_HEADER = """A0001 12 500 7500
A0001.mat 16x1+24 1000.0(0)/mV 16 0 28 -1716 0 I
# Age: 74
# Sex: Male
# Dx: 59118001,164931005
# Rx: Unknown
"""


def test_challenge_2020_header_comments_are_parsed():
    # Challenge-2020 (CPSC-2018) headers write "# Dx:" with a space; ecg-arrhythmia "#Dx:".
    assert parse_dx_from_header(CHALLENGE_HEADER) == ["59118001", "164931005"]
    assert parse_patient_from_header(CHALLENGE_HEADER) == (74.0, "male")


def test_ptbxl_axis_categories_map_to_deviations_only_when_abnormal():
    # PTB-XL's "LAD" is the mild leftward axis (12SL median -16 degrees): not a deviation.
    assert ptbxl_axis_labels("LAD") == set()
    assert ptbxl_axis_labels("MID") == set() and ptbxl_axis_labels("") == set()
    assert {PTBXL_SCP_MAP[c] for c in ptbxl_axis_labels("ALAD")} == {"left_axis_deviation"}
    assert {PTBXL_SCP_MAP[c] for c in ptbxl_axis_labels("RAD")} == {"right_axis_deviation"}
    assert {PTBXL_SCP_MAP[c] for c in ptbxl_axis_labels("ARAD")} == {"right_axis_deviation"}


LIMB = {"I": 0.0, "II": 60.0, "III": 120.0, "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}


def _dipole_ecg(theta_deg: float):
    """Limb leads rebuilt as projections of one frontal dipole with the lead-II
    waveform of the generator: the true axis is theta."""
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    w = sig.lead("II").copy()
    for lead, ang in LIMB.items():
        sig.signal[sig.lead_names.index(lead)] = w * np.cos(np.radians(theta_deg - ang))
    return sig


def test_axis_follows_the_limb_lead_dipole():
    for theta in (-45.0, 20.0, 60.0, 110.0):
        axis = run_pipeline(_dipole_ecg(theta), mode="A").measurements.axes.qrs_deg
        assert axis is not None, theta
        assert abs((axis - theta + 180) % 360 - 180) <= 15, (theta, axis)


def test_axis_survives_a_missing_limb_lead():
    sig = _dipole_ecg(40.0)
    sig.signal[sig.lead_names.index("aVF")] = 0.0      # lead-off: the other five carry the axis
    axis = run_pipeline(sig, mode="A").measurements.axes.qrs_deg
    assert axis is not None and abs(axis - 40.0) <= 20
