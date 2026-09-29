"""End-to-end pipeline + outputs smoke test (TZ §18 п.4, §11)."""

import json
import os

from cardiolens.config import load_rule_base
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.outputs import result_to_dict, write_all_outputs
from cardiolens.pipeline import run_pipeline


def test_rule_base_loads_and_builds():
    rb = load_rule_base()
    assert len(rb.rules) >= 4
    ids = {r.id for r in rb.rules}
    assert {"hyperkalemia_severity_chain", "electrode_reversal_la_ra"} <= ids


def test_end_to_end_normal():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    result = run_pipeline(sig, mode="A")
    d = result_to_dict(result)
    assert d["mode"] == "A"
    assert d["measurements"]["heart_rate_bpm"] > 0
    assert d["versions"]["rules_base"]
    assert isinstance(d["audit_log"], list) and d["audit_log"]
    # JSON is serialisable (TZ §11.1).
    assert json.loads(json.dumps(d, ensure_ascii=False))


def test_all_outputs_written(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=750, potassium=6.6)
    result = run_pipeline(sig, mode="B")
    paths = write_all_outputs(result, str(tmp_path), stem="t")
    for key in ("json", "waveform_svg", "causal_graph_png", "pdf", "conclusion_txt"):
        assert os.path.exists(paths[key]), key
        assert os.path.getsize(paths[key]) > 0, key


def test_audit_log_traceability():
    # Traceability: audit log records each layer (TZ §11, §12).
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    joined = " ".join(result.audit_log)
    for tag in ("[L2]", "[L3]", "[L4]", "[L5]", "[L6]"):
        assert tag in joined


def _fs(*codes):
    from cardiolens.contracts.findings import Finding, FindingSet
    return FindingSet(findings=[Finding(code=c, group="rhythm", label_ru=c, label_uz=c,
                                        confidence=0.7) for c in codes])


def test_af_hierarchy_withholds_contradictory_rhythm_labels():
    from cardiolens.contracts.features import Measurements
    from cardiolens.pipeline import _apply_rhythm_hierarchy

    fs = _fs("atrial_fibrillation", "sinus_rhythm", "sinus_tachycardia",
             "first_degree_av_block", "irregular_rhythm", "rbbb")
    audit = []
    _apply_rhythm_hierarchy(fs, Measurements(heart_rate_bpm=128), audit)
    assert fs.codes() == {"atrial_fibrillation", "rbbb"}
    assert "тахисистолическая" in fs.by_code("atrial_fibrillation").label_ru
    assert any("rhythm hierarchy" in a for a in audit)


def test_af_hierarchy_keeps_sinus_bradycardia_and_noop_without_af():
    from cardiolens.contracts.features import Measurements
    from cardiolens.pipeline import _apply_rhythm_hierarchy

    fs = _fs("atrial_fibrillation", "sinus_bradycardia")
    _apply_rhythm_hierarchy(fs, Measurements(heart_rate_bpm=52), [])
    assert fs.codes() == {"atrial_fibrillation", "sinus_bradycardia"}  # dev data: not net-positive
    assert "брадисистолическая" in fs.by_code("atrial_fibrillation").label_ru

    fs = _fs("sinus_rhythm", "first_degree_av_block")
    _apply_rhythm_hierarchy(fs, Measurements(heart_rate_bpm=70), [])
    assert fs.codes() == {"sinus_rhythm", "first_degree_av_block"}


def test_paced_hierarchy_withholds_invalid_criteria_but_keeps_st_elevation():
    from cardiolens.contracts.features import Measurements
    from cardiolens.pipeline import _apply_rhythm_hierarchy

    fs = _fs("paced_rhythm", "lbbb", "left_axis_deviation", "t_inversion", "sinus_rhythm",
             "first_degree_av_block", "st_elevation", "pathological_q_waves")
    audit = []
    _apply_rhythm_hierarchy(fs, Measurements(heart_rate_bpm=70), audit)
    # STE is a §10 must-not-miss finding: never silenced by the hierarchy.
    assert fs.codes() == {"paced_rhythm", "st_elevation"}
    assert any("paced rhythm, withheld" in a for a in audit)


def test_normal_synthetic_ecg_has_q_measures_but_no_pathological_q():
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10), mode="A")
    assert len(result.measurements.morphology.q_waves) == 12
    assert "pathological_q_waves" not in result.findings.codes()


def test_long_recording_is_analysed_on_leading_window():
    sig = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=75)
    result = run_pipeline(sig, mode="A")
    assert any("analysed the first 60 s" in line for line in result.audit_log)
    assert result.measurements.heart_rate_bpm is not None
