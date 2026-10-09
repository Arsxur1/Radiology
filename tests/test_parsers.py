"""Layer-1 file parsers: EDF round trip, CSV variants, extension dispatch, PTB-XL hook."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import load, load_csv, load_edf
from cardiolens.layer1_ingestion.parsers import patient_from_comments
from cardiolens.layer1_ingestion.ptbxl import load_ptbxl_record

FS = 500


def _wave(n_leads=3, seconds=2.0):
    t = np.arange(int(FS * seconds)) / FS
    return np.vstack([np.sin(2 * np.pi * (1 + k) * t) * (0.5 + 0.1 * k) for k in range(n_leads)])  # mV


def test_edf_round_trip_in_microvolts(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    x = _wave()
    p = tmp_path / "rec.edf"
    with pyedflib.EdfWriter(str(p), 3, file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.setSignalHeaders([{"label": name, "dimension": "mV", "sample_frequency": FS,
                             "physical_min": -2.0, "physical_max": 2.0,
                             "digital_min": -32768, "digital_max": 32767} for name in ("I", "avl", "V1")])
        w.writeSamples(list(x))
    s = load(p)
    assert s.source == "edf" and s.sampling_rate_hz == FS
    assert s.lead_names == ["I", "aVL", "V1"]          # canonical casing
    assert s.duration_s == pytest.approx(2.0)
    assert np.allclose(s.signal, x * 1000.0, atol=1.0)  # mV -> µV, 16-bit quantisation
    assert load_edf(p).signal.shape == (3, 2 * FS)


def test_csv_without_header_in_millivolts(tmp_path):
    x = _wave(2)
    p = tmp_path / "rec.csv"
    np.savetxt(p, x.T, delimiter=",")
    s = load_csv(p, sampling_rate_hz=FS, has_header=False, unit="mv")
    assert s.lead_names == ["ch0", "ch1"]
    assert np.allclose(s.signal, x * 1000.0, atol=1e-2)


def test_single_column_csv_and_explicit_lead_names(tmp_path):
    p = tmp_path / "one.txt"
    np.savetxt(p, np.arange(100.0))
    s = load(p, sampling_rate_hz=FS, has_header=False, lead_names=["ii"])
    assert s.signal.shape == (1, 100) and s.lead_names == ["II"]


def test_unsupported_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported extension"):
        load(tmp_path / "ecg.xml")


def test_ptbxl_hook_reads_a_local_record_and_explains_a_missing_one(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    with pytest.raises(FileNotFoundError, match="physionet.org/content/ptb-xl"):
        load_ptbxl_record(tmp_path / "00001_hr")
    x = _wave(2)
    wfdb.wrsamp("00001_hr", fs=FS, units=["mV", "mV"], sig_name=["I", "II"],
                p_signal=x.T, fmt=["16", "16"], write_dir=str(tmp_path),
                comments=["Age: 7", "Sex: F"])
    s = load_ptbxl_record(tmp_path / "00001_hr")
    assert s.source == "wfdb" and s.lead_names == ["I", "II"]
    assert (s.patient.age_years, s.patient.sex) == (7.0, "female")
    assert np.allclose(s.signal, x * 1000.0, atol=5.0)


@pytest.mark.parametrize("comments, expected", [
    (["Age: 150", "Sex: Unknown"], (None, None)),   # implausible age, unknown sex
    (["<age>: 65 <sex>: M"], (65.0, "male")),       # INCART style
    (None, (None, None)),
])
def test_patient_from_comments_edge_cases(comments, expected):
    p = patient_from_comments(comments)
    assert (p.age_years, p.sex) == expected


def test_dicom_scp_code_fallback_and_unknown_unit(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.data import get_testdata_file

    from cardiolens.layer1_ingestion import load_dicom

    sample = get_testdata_file("waveform_ecg.dcm")
    ds = pydicom.dcmread(sample)
    rhythm = next(g for g in ds.WaveformSequence if "RHYTHM" in str(g.MultiplexGroupLabel).upper())
    for ch in rhythm.ChannelDefinitionSequence:
        ch.ChannelSourceSequence[0].CodeMeaning = "unspecified"   # forces the SCP code path
    p = tmp_path / "scp.dcm"
    ds.save_as(p)
    assert load_dicom(p).lead_names == load_dicom(sample).lead_names

    rhythm.ChannelDefinitionSequence[0].ChannelSensitivityUnitsSequence[0].CodeValue = "mmHg"
    q = tmp_path / "unit.dcm"
    ds.save_as(q)
    with pytest.raises(ValueError, match="unsupported channel unit"):
        load_dicom(q)
