"""Hypertrophy criteria (TZ §7.4, open item 2): Cornell, Romhilt-Estes, pediatric bands."""

from cardiolens.config import norms_for_patient
from cardiolens.contracts.features import (
    Intervals,
    Measurements,
    Morphology,
    STMeasurement,
    Voltages,
)
from cardiolens.contracts.signal import Patient
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer4_findings.hypertrophy import romhilt_estes_points
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline


def _fs(m: Measurements, *, age=50.0, sex="male", age_days=None):
    sig = generate_synthetic_ecg(age_years=age, sex=sex)
    sig.patient = Patient(age_years=age, age_days=age_days, sex=sex)
    return detect_findings(sig, m, norms=norms_for_patient(age, age_days))


def _m(**kw) -> Measurements:
    base = dict(heart_rate_bpm=70, rhythm_regular=True, intervals=Intervals(pr_ms=150, qrs_ms=88))
    base.update(kw)
    return Measurements(**base)


def test_cornell_fires_without_sokolow():
    m = _m(sokolow_lyon_uv=2500, voltages=Voltages(r_avl_uv=1500, s_v3_uv=1600))
    fs = _fs(m, sex="male")
    lvh = fs.by_code("lvh")
    assert lvh is not None and "Cornell" in lvh.label_ru
    # Same voltages are LVH in a woman too (lower threshold) and not in a man at 2.7 mV.
    m2 = _m(sokolow_lyon_uv=2500, voltages=Voltages(r_avl_uv=1200, s_v3_uv=1500))
    assert "lvh" not in _fs(m2, sex="male").codes()
    assert "lvh" in _fs(m2, sex="female").codes()


def test_romhilt_estes_points_give_probable_only():
    # Voltage (3) + strain (3) = 6 points with Sokolow-Lyon and Cornell negative:
    # `lvh_probable`, never the main code on the point score alone (D39).
    st = [STMeasurement(lead=l, j_point_uv=-80, j60_uv=-120, j80_uv=-120, qrs_net_uv=1800) for l in ("V5", "V6")]
    m = _m(sokolow_lyon_uv=3000, voltages=Voltages(r_max_limb_uv=2100, r_avl_uv=900, s_v3_uv=800),
           morphology=Morphology(st=st, inverted_t_leads=["V5", "V6"]))
    fs = _fs(m)
    pts, why = romhilt_estes_points(m, norms_for_patient(50, None), fs)
    assert pts >= 6 and {"voltage", "strain"} <= set(why)
    assert "lvh_probable" in fs.codes() and "lvh" not in fs.codes()
    assert not fs.by_code("lvh_probable").critical
    # Strain needs BOTH T inversion and >= 1 mm depression: inversion alone is 0 points.
    m2 = _m(sokolow_lyon_uv=3000, voltages=Voltages(r_max_limb_uv=2100, r_avl_uv=900, s_v3_uv=800),
            morphology=Morphology(inverted_t_leads=["V5", "V6"]))
    pts2, why2 = romhilt_estes_points(m2, norms_for_patient(50, None), _fs(m2))
    assert "strain" not in why2 and pts2 == 3
    # A voltage criterion plus the score -> `lvh` with the score in the label.
    m3 = _m(sokolow_lyon_uv=3800, voltages=Voltages(r_max_limb_uv=2100, r_avl_uv=900, s_v3_uv=800),
            morphology=Morphology(st=st, inverted_t_leads=["V5", "V6"]))
    fs3 = _fs(m3)
    assert "lvh" in fs3.codes() and "Romhilt-Estes" in fs3.by_code("lvh").label_ru


def test_voltage_criteria_are_silent_under_bundle_branch_block():
    # Cornell-level voltages with a wide QRS typed as LBBB: no LVH call (D39).
    m = _m(intervals=Intervals(pr_ms=150, qrs_ms=150), qrs_v1_terminal_positive=False,
           sokolow_lyon_uv=3800, voltages=Voltages(r_avl_uv=1500, s_v3_uv=1600))
    fs = _fs(m)
    assert "lbbb" in fs.codes()
    assert not fs.codes() & {"lvh", "lvh_probable"}


def test_pediatric_tall_r_alone_is_not_lvh():
    # A 6-year-old with RV6 2.4 mV and SV1 2.1 mV: adult Sokolow-Lyon (4.5 mV) would
    # scream LVH; both are under the 3-8 y 98th percentiles -> nothing.
    m = _m(heart_rate_bpm=95, sokolow_lyon_uv=4500,
           voltages=Voltages(r_v6_uv=2400, s_v1_uv=2100, r_v1_uv=1200, s_v6_uv=300))
    assert not _fs(m, age=6.0).codes() & {"lvh", "rvh", "lvh_probable"}


def test_pediatric_lvh_and_rvh_against_band():
    # RV6 above the 3-8 y band plus strain -> LVH.
    st = [STMeasurement(lead="V6", j_point_uv=-60, j60_uv=-110, j80_uv=-110, qrs_net_uv=2900)]
    m = _m(heart_rate_bpm=95, voltages=Voltages(r_v6_uv=2900, s_v1_uv=1500, r_v1_uv=800, s_v6_uv=200),
           morphology=Morphology(st=st, inverted_t_leads=["V6"]))
    fs = _fs(m, age=6.0)
    assert "lvh" in fs.codes() and "детские нормы" in fs.by_code("lvh").label_ru
    # RV1 and SV6 both above the infant band -> RVH.
    m2 = _m(heart_rate_bpm=130, voltages=Voltages(r_v1_uv=2300, s_v6_uv=1100, r_v6_uv=1500, s_v1_uv=900))
    assert "rvh" in _fs(m2, age=0.5).codes()


def test_synthetic_normals_have_no_hypertrophy_codes():
    for kw in (dict(age_years=40.0), dict(age_years=8.0), dict(age_years=0.3, heart_rate_bpm=130)):
        r = run_pipeline(generate_synthetic_ecg(seed=11, **kw), mode="A", provider=StubProvider())
        assert not r.findings.codes() & {"lvh", "rvh", "lvh_probable"}, (kw, r.findings.codes())
    big = run_pipeline(generate_synthetic_ecg(amplitude_scale=3.0, seed=12), mode="A", provider=StubProvider())
    assert "lvh" in big.findings.codes()
    v = big.measurements.voltages
    assert v.r_v6_uv and v.s_v1_uv and v.r_avl_uv is not None and v.intrinsicoid_v5_v6_ms is not None
