"""ST logic hardened on real data (TZ §7.2): thresholds, contiguity, concordance.

Rule-level tests build Measurements directly; end-to-end tests run synthetic
stimuli through the full pipeline.
"""

from cardiolens.config import norms_for_patient
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import STANDARD_12_LEADS
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline
from cardiolens.safety import scan_critical


def _st(dev: dict[str, float], *, j60=None, qrs_net=None) -> Morphology:
    """ST list where `dev` gives the J-point deviation per lead (0 elsewhere)."""
    out = []
    for lead in STANDARD_12_LEADS:
        d = dev.get(lead, 0.0)
        out.append(STMeasurement(
            lead=lead, j_point_uv=d,
            j60_uv=(j60 or {}).get(lead, d), j80_uv=(j60 or {}).get(lead, d),
            qrs_net_uv=(qrs_net or {}).get(lead)))
    return Morphology(st=out)


def _codes(m: Measurements, *, age=45.0, sex="male", extra=None):
    sig = generate_synthetic_ecg(age_years=age, sex=sex)
    norms = norms_for_patient(age, None)
    fs = detect_findings(sig, m, norms=norms)
    return fs.codes()


def _base(**kw) -> Measurements:
    return Measurements(heart_rate_bpm=72, rhythm_regular=True, intervals=Intervals(qrs_ms=90), **kw)


def test_v2_v3_use_sex_and_age_specific_thresholds():
    # 150 µV in V2/V3: STEMI-level for a woman, not for a man >= 40 (2 mm), nor <40 (2.5 mm).
    m = _base(morphology=_st({"V2": 150, "V3": 150}))
    assert "st_elevation" in _codes(m, sex="female")
    assert "st_elevation" not in _codes(m, sex="male", age=45)
    assert "st_elevation" not in _codes(m, sex="male", age=30)
    m2 = _base(morphology=_st({"V2": 220, "V3": 220}))
    assert "st_elevation" in _codes(m2, sex="male", age=45)
    assert "st_elevation" not in _codes(m2, sex="male", age=30)


def test_elevation_must_persist_into_st_segment():
    # J-point 180 but J+60 back at baseline: an S-wave tail, not a STEMI.
    m = _base(morphology=_st({"II": 180, "III": 180, "aVF": 180}, j60={"II": 0, "III": 0, "aVF": 0}))
    assert "st_elevation" not in _codes(m)
    m2 = _base(morphology=_st({"II": 180, "III": 180, "aVF": 180}))
    assert "st_elevation" in _codes(m2)


def test_non_contiguous_elevation_is_nonspecific_and_not_critical():
    m = _base(morphology=_st({"II": 180, "V2": 260}))
    codes = _codes(m)
    assert "st_elevation" not in codes and "st_elevation_nonspecific" in codes
    sig = generate_synthetic_ecg(age_years=45)
    fs = detect_findings(sig, m, norms=norms_for_patient(45, None))
    assert not any(a.code == "st_elevation" for a in scan_critical(fs, m).alerts)


def test_upsloping_depression_is_not_called():
    m = _base(morphology=_st({"V5": -120, "V6": -120}))
    m.morphology.st = [STMeasurement(lead=s.lead, j_point_uv=s.j_point_uv,
                                     j60_uv=s.j60_uv, j80_uv=(-40 if s.lead in ("V5", "V6") else 0))
                       for s in m.morphology.st]
    assert "st_depression" not in _codes(m)
    assert "st_depression" in _codes(_base(morphology=_st({"V5": -120, "V6": -120})))


def test_discordant_st_in_lvh_is_flagged_secondary():
    lvh = dict(sokolow_lyon_uv=4000)
    # Tall R in V5/V6 with ST depression = discordant: still reported as depression
    # (strain pattern) but flagged secondary with lower confidence.
    m = _base(morphology=_st({"V5": -150, "V6": -150}, qrs_net={"V5": 2000, "V6": 1800}), **lvh)
    sig = generate_synthetic_ecg(age_years=45)
    fs = detect_findings(sig, m, norms=norms_for_patient(45, None))
    codes = fs.codes()
    assert "lvh" in codes and "secondary_st_t_change" in codes and "st_depression" in codes
    assert fs.by_code("st_depression").confidence < 0.7
    # Discordant ELEVATION in that context is never a STEMI.
    m2 = _base(morphology=_st({"V1": 200, "V2": 300, "V3": 300}, qrs_net={"V1": -1500, "V2": -1800, "V3": -1200}), **lvh)
    codes2 = _codes(m2)
    assert "st_elevation" not in codes2 and "secondary_st_t_change" in codes2


def test_inferior_stemi_end_to_end_is_still_critical():
    sig = generate_synthetic_ecg(heart_rate_bpm=70, st_elevation_uv=250,
                                 st_leads=["II", "III", "aVF"], seed=50)
    r = run_pipeline(sig, mode="A", provider=StubProvider())
    assert "st_elevation" in r.findings.codes()
    assert any(a.code == "st_elevation" for a in r.safety.alerts)


def test_bundle_branch_block_raises_no_primary_st_codes():
    for bbb in ("rbbb", "lbbb"):
        r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, bbb=bbb, seed=6),
                         mode="A", provider=StubProvider())
        codes = r.findings.codes()
        assert bbb in codes
        assert "st_elevation" not in codes
        assert not any(a.code == "st_elevation" for a in r.safety.alerts)


def test_st_elevation_any_is_descriptive_and_never_critical():
    # Discordant anterior elevation under LVH: descriptor present, no STEMI alert.
    m = _base(morphology=_st({"V1": 200, "V2": 300, "V3": 300},
                             qrs_net={"V1": -1500, "V2": -1800, "V3": -1200}),
              sokolow_lyon_uv=4000)
    sig = generate_synthetic_ecg(age_years=45)
    fs = detect_findings(sig, m, norms=norms_for_patient(45, None))
    codes = fs.codes()
    assert "st_elevation_any" in codes and "st_elevation" not in codes
    assert not any(a.code == "st_elevation" for a in scan_critical(fs, m).alerts)
    # A single non-contiguous lead: descriptor only.
    m2 = _base(morphology=_st({"V4": 150}))
    assert _codes(m2) & {"st_elevation_any"} and "st_elevation" not in _codes(m2)
    # Inferior STEMI carries both the alert and the descriptor.
    m3 = _base(morphology=_st({"II": 180, "III": 180, "aVF": 180}))
    assert {"st_elevation", "st_elevation_any"} <= _codes(m3)


def test_normal_has_no_st_codes():
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=75, seed=9), mode="A", provider=StubProvider())
    assert not r.findings.codes() & {"st_elevation", "st_elevation_nonspecific", "st_depression",
                                     "secondary_st_t_change", "pericarditis", "st_elevation_any"}
