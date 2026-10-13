"""Photograph + record pairing and the image report (blocker 3 of the application review)."""

import numpy as np

from cardiolens.validation.ecg_images import ImageCase, ImageReport, pair_images_with_records


def test_pairs_by_basename_and_by_the_challenge_suffix(tmp_path):
    for name in ("rec1", "rec2", "lonely"):
        (tmp_path / f"{name}.hea").write_text("x")
    (tmp_path / "rec1.dat").write_bytes(b"\0")
    (tmp_path / "rec2.dat").write_bytes(b"\0")
    (tmp_path / "rec1.png").write_bytes(b"\0")            # same basename
    (tmp_path / "rec2-0001.jpg").write_bytes(b"\0")       # Challenge-2024 layout
    (tmp_path / "rec2-0002.jpg").write_bytes(b"\0")       # a second photo of the same record
    (tmp_path / "noise.png").write_bytes(b"\0")           # no record
    (tmp_path / "lonely.png").write_bytes(b"\0")          # header without signal file
    pairs = pair_images_with_records(tmp_path)
    assert [(i.name, r.name) for i, r in pairs] == [("rec1.png", "rec1"), ("rec2-0001.jpg", "rec2"), ("rec2-0002.jpg", "rec2")]


def test_report_summary_counts_failures_and_averages_the_rest():
    rep = ImageReport(cases=[
        ImageCase("a.png", r_mean=0.9, hr_delta=1.0, axis_delta=5.0, qrs_delta=8.0, jaccard=1.0, identical=True),
        ImageCase("b.png", r_mean=0.5, hr_delta=7.0, axis_delta=20.0, qrs_delta=30.0, jaccard=0.5, identical=False),
        ImageCase("c.png", error="ValueError: no grid"),
    ])
    s = rep.summary()
    assert s["n"] == 3 and s["digitised"] == 2 and s["failed"] == 1
    assert s["r_mean"] == 0.7 and s["hr_within3"] == 0.5 and s["axis_within15"] == 0.5 and s["codes_identical"] == 0.5
    assert np.isclose(s["jaccard"], 0.75)
    text = "\n".join(rep.lines())
    assert "c.png: FAILED" in text and "digitised 2" in text
