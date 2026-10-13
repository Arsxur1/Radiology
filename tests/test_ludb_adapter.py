"""LUDB annotation triplets -> reference PR / QRS / QT per record (TZ §16 interval MAE)."""

import numpy as np
import wfdb

from cardiolens.validation.fetch_ludb import parse_header_comments
from cardiolens.validation.harness import ludb_intervals_from_annotations


def _write_lead(tmp_path, name, lead, fs, beats):
    """beats: list of (p_on, p_off, q_on, q_off, t_on, t_off) in samples; None p = no P wave."""
    syms, samples = [], []
    for p_on, p_off, q_on, q_off, t_on, t_off in beats:
        if p_on is not None:
            syms += ["(", "p", ")"]; samples += [p_on, (p_on + p_off) // 2, p_off]
        syms += ["(", "N", ")", "(", "t", ")"]
        samples += [q_on, (q_on + q_off) // 2, q_off, t_on, (t_on + t_off) // 2, t_off]
    wfdb.wrann(name, lead, np.asarray(samples), symbol=syms, fs=fs, write_dir=str(tmp_path))


def test_intervals_are_medians_over_beats_and_leads(tmp_path):
    fs = 500
    beat = lambda k: (1000 * k + 0, 1000 * k + 50, 1000 * k + 80, 1000 * k + 125, 1000 * k + 200, 1000 * k + 280)
    _write_lead(tmp_path, "7", "ii", fs, [beat(k) for k in range(4)])          # PR 160, QRS 90, QT 400 ms
    _write_lead(tmp_path, "7", "avf", fs, [beat(k)[:2] + (1000 * k + 80, 1000 * k + 130, 1000 * k + 200, 1000 * k + 290) for k in range(4)])  # QRS 100, QT 420
    ivs = ludb_intervals_from_annotations(tmp_path, "7", fs)
    assert ivs["pr"] == 160.0
    assert ivs["qrs"] == 95.0 and ivs["qt"] == 410.0                           # median of the two leads


def test_beats_without_p_give_no_pr(tmp_path):
    fs = 500
    _write_lead(tmp_path, "8", "ii", fs, [(None, None, 1000 * k + 80, 1000 * k + 125, 1000 * k + 200, 1000 * k + 280) for k in range(3)])
    ivs = ludb_intervals_from_annotations(tmp_path, "8", fs)
    assert ivs["pr"] is None and ivs["qrs"] == 90.0


def test_header_comments():
    meta = parse_header_comments("#<age>: 51\n#<sex>: F\n#<diagnoses>:\n#Rhythm: Sinus bradycardia.\n"
                                 "#Electric axis of the heart: left axis deviation. \n#Left ventricular hypertrophy. \n")
    assert meta["age"] == 51.0 and meta["sex"] == "female"
    assert meta["rhythm"] == "Sinus bradycardia" and meta["axis"] == "left axis deviation"
    assert meta["diagnoses"] == ["Left ventricular hypertrophy"]
