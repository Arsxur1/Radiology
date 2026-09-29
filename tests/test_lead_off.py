"""Leads that carry no signal are not measured (D45).

The ecg-arrhythmia paediatric cohort stores limb-only recordings as 12 leads
with V1-V6 all zero (673 of 971 records aged 4-8 y). Before D45 those zero
channels were read as flat T waves, 0-µV R/S voltages and 0-µV ST — 70 % of
the children carried ``flat_t_waves``.
"""

import numpy as np

from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer2_preprocess.quality import lead_off_leads
from cardiolens.pipeline import run_pipeline

PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]


def _limb_only():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    for lead in PRECORDIAL:
        sig.signal[sig.lead_names.index(lead)] = 0.0
    return sig


def test_zero_leads_are_detected_as_lead_off():
    assert lead_off_leads(_limb_only()) == set(PRECORDIAL)
    assert lead_off_leads(generate_synthetic_ecg(heart_rate_bpm=72)) == set()


def test_zero_precordials_are_not_measured():
    res = run_pipeline(_limb_only(), mode="A")
    m = res.measurements
    measured = {st.lead for st in m.morphology.st}
    assert measured.isdisjoint(PRECORDIAL) and {"I", "II", "aVF"} <= measured
    assert not set(m.morphology.flat_t_leads) & set(PRECORDIAL)
    assert "flat_t_waves" not in res.findings.codes()
    assert m.voltages.r_v6_uv is None and m.voltages.s_v1_uv is None
    assert m.sokolow_lyon_uv is None
    assert m.qrs_v1_terminal_positive is None and m.qrs_v6_terminal_positive is None
    assert any(f.startswith("lead_off:") for f in res.signal.quality.flags)
    # Limb-lead measurements survive.
    assert m.heart_rate_bpm is not None and abs(m.heart_rate_bpm - 72) < 3
    assert m.axes.qrs_deg is not None and m.qrs_amplitude_limb_uv is not None


def test_full_record_unchanged_by_the_guard():
    ref = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").measurements
    assert len(ref.morphology.st) == 12
    assert ref.voltages.r_v6_uv is not None and ref.sokolow_lyon_uv is not None
    assert np.isfinite(ref.voltages.r_v6_uv)
