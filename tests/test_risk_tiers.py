"""Error analysis by clinical risk tier (TZ §16; blocker 1 of the application review)."""

from cardiolens.config import load_risk_tiers
from cardiolens.validation import SCORED_LABELS, risk_report_lines, score_risk_tiers

TIERS = {"tiers": {"critical": ["st_elevation", "long_qt"], "minor": ["sinus_rhythm"]},
         "targets": {"critical": {"case_sensitivity": 0.9, "case_false_alert_rate": 0.05}}}


def test_case_level_pools_the_tier_and_label_level_counts_each_code():
    preds = [{"long_qt"}, set(), {"st_elevation"}, {"sinus_rhythm"}]
    gold = [{"st_elevation"}, {"st_elevation"}, {"st_elevation", "long_qt"}, {"sinus_rhythm"}]
    r = score_risk_tiers(preds, gold, tiers=TIERS)
    c = r["critical"]
    # case 1: STEMI reported as long QT — caught at the case level (escalation), missed at the label level
    assert c.case_positive == 3 and c.case_caught == 2 and c.case_sensitivity == 0.667
    assert c.pooled.fn == 3 and c.missed_by_code == {"st_elevation": 2, "long_qt": 1}
    assert c.pooled.fp == 1 and c.false_by_code == {"long_qt": 1}
    assert c.case_negative == 1 and c.case_false_alert == 0 and c.case_false_alert_rate == 0.0
    assert c.meets() == {"case_sensitivity": False, "case_false_alert_rate": True}
    assert r["minor"].case_sensitivity == 1.0 and r["minor"].targets == {}


def test_labels_restrict_the_codes_and_empty_support_gives_none():
    r = score_risk_tiers([set()], [set()], labels=["sinus_rhythm"], tiers=TIERS)
    assert r["critical"].codes == [] and r["critical"].case_sensitivity is None
    assert r["critical"].meets() == {"case_sensitivity": None, "case_false_alert_rate": None}


def test_yaml_tiers_cover_every_scored_label_exactly_once():
    cfg = load_risk_tiers()
    codes = [c for tier in cfg["tiers"].values() for c in tier]
    assert sorted(codes) == sorted(SCORED_LABELS)
    assert set(cfg["targets"]) == set(cfg["tiers"])


def test_report_lines_name_the_misses():
    r = score_risk_tiers([set()], [{"st_elevation"}], tiers=TIERS)
    text = "\n".join(risk_report_lines(r))
    assert "critical" in text and "missed: st_elevation 1" in text and "not claimed" in text
