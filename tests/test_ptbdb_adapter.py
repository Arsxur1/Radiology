"""PTB Diagnostic DB adapter (open item 1): header metadata, day binning, territories."""

import json

import numpy as np

from cardiolens.validation.fetch_ptbdb import days_between, parse_header_comments, truncate_header
from cardiolens.validation.harness import (
    PTBDB_TERRITORY,
    build_ptbdb_dataset,
    territories_of_leads,
)

_HEADER = """s0010_re 15 1000 38400
s0010_re.dat 16 2000 16 0 -489 -8337 0 i
s0010_re.dat 16 2000 16 0 -458 -16369 0 ii
s0010_re.dat 16 2000 16 0 31 6829 0 iii
s0010_re.dat 16 2000 16 0 474 4582 0 avr
s0010_re.dat 16 2000 16 0 -260 11687 0 avl
s0010_re.dat 16 2000 16 0 -214 -16657 0 avf
s0010_re.dat 16 2000 16 0 -88 -12469 0 v1
s0010_re.dat 16 2000 16 0 -241 5636 0 v2
s0010_re.dat 16 2000 16 0 -112 -14299 0 v3
s0010_re.dat 16 2000 16 0 212 -17916 0 v4
s0010_re.dat 16 2000 16 0 393 -6668 0 v5
s0010_re.dat 16 2000 16 0 390 -17545 0 v6
s0010_re.xyz 16 2000 16 0 -3 -13009 0 vx
s0010_re.xyz 16 2000 16 0 120 7109 0 vy
s0010_re.xyz 16 2000 16 0 -18 -1992 0 vz

# age: 81
# sex: female
# ECG date: 01/10/1990

# Diagnose:
# Reason for admission: Myocardial infarction
# Acute infarction (localization): infero-latera
# Former infarction (localization): no
# Infarction date (acute): 29-Sep-90
"""


def test_header_metadata_and_days():
    meta = parse_header_comments(_HEADER)
    assert meta["reason"] == "Myocardial infarction"
    assert meta["acute_localization"] == "infero-latera"
    assert meta["days_since_infarction"] == 2
    assert meta["age"] == 81.0 and meta["sex"] == "female"
    assert days_between("n/a", "01/10/1990") is None


def test_truncate_header_rewrites_sample_count():
    t = truncate_header(_HEADER, 10000)
    assert t.splitlines()[0] == "s0010_re 15 1000 10000"
    assert "# Reason for admission" in t


def test_territories():
    assert PTBDB_TERRITORY["infero-latera"] == {"inferior", "lateral"}
    assert PTBDB_TERRITORY["posterior"] == set()
    assert territories_of_leads(["II", "III", "aVF"]) == {"inferior"}
    assert territories_of_leads(["V2", "V3", "aVR"]) == {"anterior"}


def _write_record(root, rec, header, n_sig=15, fs=1000, seconds=2.0):
    p = root / rec
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fs)
    p.with_suffix(".hea").write_text(truncate_header(header, n), encoding="utf-8")
    rng = np.random.default_rng(0)
    data = (rng.normal(0, 50, (n, n_sig))).astype("<i2")
    p.with_suffix(".dat").write_bytes(data.tobytes())


def test_build_dataset_gold_by_days(tmp_path):
    recs = []
    for name, days_line in (("patient001/s0010_re", "29-Sep-90"),      # 2 days -> STEMI gold
                            ("patient002/s0020_re", "01-Sep-90")):     # 30 days -> negative
        header = _HEADER.replace("29-Sep-90", days_line).replace("s0010_re", name.split("/")[1])
        _write_record(tmp_path, name, header)
        meta = parse_header_comments(header)
        meta.update(record=name, patient=name.split("/")[0])
        recs.append(meta)
    ctrl = _HEADER.replace("Myocardial infarction", "Healthy control").replace("s0010_re", "s0030_re")
    _write_record(tmp_path, "patient003/s0030_re", ctrl)
    m = parse_header_comments(ctrl); m.update(record="patient003/s0030_re", patient="patient003"); recs.append(m)
    (tmp_path / "manifest.json").write_text(json.dumps({"records": recs}), encoding="utf-8")

    cases = build_ptbdb_dataset(tmp_path, max_days=3, seconds=2.0)
    assert [c.gold_findings for c in cases] == [{"st_elevation"}, set(), set()]
    assert cases[0].meta["territory"] == {"inferior", "lateral"}
    assert cases[0].meta["days"] == 2 and cases[2].meta["mi"] is False
    sig = cases[0].signal
    assert sig.lead_names[:3] == ["I", "II", "III"] and sig.signal.shape == (12, 2000)
    assert sig.patient.sex == "female" and sig.patient.age_years == 81.0
