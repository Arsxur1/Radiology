"""DICOM-ECG waveform parser (TZ §4, D93) on pydicom's bundled 12-lead ECG sample."""

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.data import get_testdata_file

from cardiolens.layer1_ingestion import load, load_dicom
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.pipeline import run_pipeline

SAMPLE = get_testdata_file("waveform_ecg.dcm")


def test_reads_rhythm_group_leads_units_and_demographics():
    s = load_dicom(SAMPLE)
    assert s.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert s.sampling_rate_hz == 1000.0 and s.duration_s == 10.0     # RHYTHM, not the 1.2 s median
    assert s.source == "dicom"
    assert (s.patient.age_years, s.patient.sex) == (42.0, "female")  # "042Y", "F"
    assert 500 < float(np.abs(s.signal).max()) < 5000                 # µV scale (1.25 µV per unit)
    assert load(SAMPLE).lead_names == s.lead_names                    # dispatch by .dcm


def test_measurements_agree_with_the_devices_own_annotations():
    # The file carries the acquiring device's measurements: RR 982 ms, PR 161, QRS 75 ms,
    # QRS axis 52°. Our QT is calibrated to GE 12SL (+30 ms, D30), so it is not compared.
    r = run_pipeline(load_dicom(SAMPLE), mode="A", provider=StubProvider())
    m = r.measurements
    assert abs(m.heart_rate_bpm - 60000 / 982) < 2
    assert abs(m.intervals.qrs_ms - 75) <= 10
    assert abs(m.intervals.pr_ms - 161) <= 20
    assert abs(m.axes.qrs_deg - 52) <= 15
    assert "sinus_rhythm" in r.findings.codes()


def test_millivolt_channels_are_scaled_to_microvolts(tmp_path):
    ds = pydicom.dcmread(SAMPLE)
    for ch in ds.WaveformSequence[0].ChannelDefinitionSequence:
        ch.ChannelSensitivityUnitsSequence[0].CodeValue = "mV"
    p = tmp_path / "mv.dcm"
    ds.save_as(p)
    a, b = load_dicom(SAMPLE).signal, load_dicom(p).signal
    assert np.allclose(b, a * 1000.0, rtol=1e-5)


def test_age_in_months_sets_age_days(tmp_path):
    ds = pydicom.dcmread(SAMPLE)
    ds.PatientAge = "006M"
    p = tmp_path / "infant.dcm"
    ds.save_as(p)
    pt = load_dicom(p).patient
    assert pt.age_days == pytest.approx(182.6, abs=0.5) and pt.age_years == pytest.approx(0.5, abs=0.01)


def test_non_waveform_dicom_is_rejected():
    with pytest.raises(ValueError, match="WaveformSequence"):
        load_dicom(get_testdata_file("CT_small.dcm"))


def test_dicom_upload_through_the_api():
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    with open(SAMPLE, "rb") as fh:
        r = TestClient(app).post("/api/analyze", data={"mode": "A"},
                                 files=[("files", ("ecg.dcm", fh.read(), "application/dicom"))])
    assert r.status_code == 200
    assert r.json()["patient"]["sex"] == "female"
