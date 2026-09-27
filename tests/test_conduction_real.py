"""Measurement robustness lessons from real PhysioNet data (TZ §6, §16).

Envelope-based QRS bounds (BBB no longer reads as narrow), amplitude-refined P
onsets (PR tracks the stimulus within a few ms) and the resulting findings, all
through the full pipeline on synthetic stimuli.
"""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline
from cardiolens.validation import build_synthetic_dataset, run_validation


def _run(**kw):
    return run_pipeline(generate_synthetic_ecg(**kw), mode="A", provider=StubProvider())


@pytest.mark.parametrize("bbb,expect,other", [("rbbb", "rbbb", "lbbb"), ("lbbb", "lbbb", "rbbb")])
def test_bundle_branch_block_end_to_end(bbb, expect, other):
    r = _run(heart_rate_bpm=70, bbb=bbb, seed=6)
    assert r.measurements.intervals.qrs_ms >= 120
    codes = r.findings.codes()
    assert expect in codes and other not in codes
    assert "wide_qrs_nonspecific" not in codes


def test_normal_qrs_is_narrow_and_not_a_block():
    r = _run(heart_rate_bpm=75, seed=9)
    assert 60 <= r.measurements.intervals.qrs_ms <= 110
    assert not r.findings.codes() & {"rbbb", "lbbb", "wide_qrs_nonspecific", "wpw_preexcitation"}


@pytest.mark.parametrize("pr", [150, 240, 280, 300])
def test_pr_tracks_stimulus(pr):
    r = _run(heart_rate_bpm=70, pr_ms=pr, seed=21)
    assert r.measurements.intervals.pr_ms == pytest.approx(pr, abs=12)
    assert ("first_degree_av_block" in r.findings.codes()) == (pr > 200)


def test_synthetic_interval_mae_is_small():
    rep = run_validation(build_synthetic_dataset())
    assert rep.pr_mae_ms is not None and rep.pr_mae_ms <= 15
    assert rep.qt_mae_ms is not None and rep.qt_mae_ms <= 15
    assert rep.macro_sensitivity == 1.0


def test_slow_rate_does_not_double_count_t_waves():
    r = _run(heart_rate_bpm=40, pr_ms=160, seed=20)
    assert r.measurements.heart_rate_bpm == pytest.approx(40, abs=2)
    assert "sinus_bradycardia" in r.findings.codes()
