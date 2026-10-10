"""FastAPI service tests (TZ §14)."""

import os

import pytest
from fastapi.testclient import TestClient

from cardiolens.api.app import app

client = TestClient(app)
_SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_cases_listed():
    r = client.get("/api/cases")
    assert r.status_code == 200
    assert {"normal", "hyperk", "stemi"} <= set(r.json()["cases"])


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "CardioLens" in r.text


def test_analyze_synthetic_hyperk_mode_b_escalates():
    r = client.post("/api/analyze", data={"case": "hyperk", "mode": "B"})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "escalate"
    assert d["escalation_reason"]
    assert any(a["code"] == "peaked_t_waves" for a in d["critical_alerts"])
    assert d["conclusion_ru"] and d["conclusion_uz"]
    assert "analysis_id" in d


def test_analyze_stemi_critical():
    r = client.post("/api/analyze", data={"case": "stemi", "mode": "A"})
    d = r.json()
    assert d["mode"] == "A"
    assert any(a["code"] == "st_elevation" for a in d["critical_alerts"])


def test_analyze_requires_input():
    r = client.post("/api/analyze", data={"mode": "A"})
    assert r.status_code == 400


def test_analyze_csv_upload_and_artifacts():
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"mode": "A", "sampling_rate_hz": "500", "unit": "uv"},
            files={"files": ("sample_ecg.csv", f, "text/csv")},
        )
    assert r.status_code == 200
    d = r.json()
    assert d["measurements"]["heart_rate_bpm"] == pytest.approx(75, abs=4)
    aid = d["analysis_id"]
    # All downstream artifacts are retrievable.
    assert client.get(f"/api/analysis/{aid}/report.pdf").status_code == 200
    assert client.get(f"/api/analysis/{aid}/waveform.svg").status_code == 200
    assert client.get(f"/api/analysis/{aid}/graph.png").status_code == 200
    assert client.get(f"/api/analysis/{aid}/conclusion?lang=uz").status_code == 200


def test_csv_upload_without_rate_rejected():
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A"},
                        files={"files": ("x.csv", f, "text/csv")})
    assert r.status_code == 400


def test_missing_analysis_404():
    assert client.get("/api/analysis/deadbeef/report.pdf").status_code == 404


def test_feedback_and_calibration_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    r = client.post("/api/analyze", data={"case": "stemi", "mode": "A"})
    aid = r.json()["analysis_id"]
    fb = {
        "analysis_id": aid,
        "reviewer": "dr. Y",
        "findings": [
            {"code": "st_elevation", "verdict": "confirmed", "confidence": 0.8},
            {"code": "short_qt", "verdict": "rejected", "confidence": 0.7},
        ],
        "overall_agreement": True,
    }
    rf = client.post(f"/api/analysis/{aid}/feedback", json=fb)
    assert rf.status_code == 200 and rf.json()["status"] == "recorded"
    # Calibration endpoint runs (too few samples -> stays identity).
    rc = client.post("/api/calibration/fit")
    assert rc.status_code == 200
    assert "temperature" in rc.json() and "ece_before" in rc.json()


def test_feedback_uses_stored_confidence_and_summarises(tmp_path, monkeypatch):
    # D87: the server takes confidences from the stored analysis, not the client.
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("CARDIOLENS_CALIBRATOR_PATH", str(tmp_path / "cal.json"))
    r = client.post("/api/analyze", data={"case": "stemi", "mode": "A"}).json()
    aid = r["analysis_id"]
    f0 = r["findings"][0]
    fb = {"analysis_id": aid, "findings": [
        {"code": f0["code"], "verdict": "rejected", "confidence": 0.99},
        {"code": "atrial_flutter", "verdict": "missed"},
    ]}
    assert client.post(f"/api/analysis/{aid}/feedback", json=fb).status_code == 200
    s = client.get("/api/feedback/summary").json()["findings"]
    assert s[f0["code"]]["rejected"] == 1 and s[f0["code"]]["precision"] == 0.0
    assert s[f0["code"]]["mean_confidence_shown"] == f0["confidence"]  # not the client's 0.99
    assert s["atrial_flutter"]["missed"] == 1 and s["atrial_flutter"]["precision"] is None
    # Too few samples: nothing is fitted or saved.
    rc = client.post("/api/calibration/fit", params={"save": True}).json()
    assert rc["saved"] is False and not (tmp_path / "cal.json").exists()


def test_calibration_fit_saves_when_it_helps(tmp_path, monkeypatch):
    from cardiolens.feedback import Feedback, FeedbackStore, FindingFeedback

    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("CARDIOLENS_CALIBRATOR_PATH", str(tmp_path / "cal.json"))
    store = FeedbackStore.at(tmp_path / "fb.jsonl")
    # Overconfident: 0.9 findings right only 1 time in 3.
    for i in range(30):
        store.append(Feedback(analysis_id=str(i), findings=[FindingFeedback(
            code="x", verdict="confirmed" if i % 3 == 0 else "rejected", confidence=0.9)]))
    rc = client.post("/api/calibration/fit", params={"save": True}).json()
    assert rc["fitted"] and rc["saved"] and rc["ece_after"] < rc["ece_before"]
    assert (tmp_path / "cal.json").exists()


def test_wfdb_header_cannot_point_at_files_that_were_not_uploaded():
    # D104: a header naming a data file that was not uploaded is rejected before wfdb reads
    # anything, so no other file on the server can be pulled in as a signal.
    hea = b"x 1 250 2000\nother 16 200/mV 16 0 0 0 0 I\n"
    r = client.post("/api/analyze", data={"mode": "A"}, files=[("files", ("x.hea", hea, "text/plain"))])
    assert r.status_code == 400 and "not uploaded" in r.json()["detail"]


def test_malformed_upload_is_a_400_not_a_server_error():
    r = client.post("/api/analyze", data={"mode": "A"},
                    files=[("files", ("ecg.edf", b"not an edf file", "application/octet-stream"))])
    assert r.status_code == 400 and "could not parse" in r.json()["detail"]


def test_oversized_upload_is_rejected_while_reading(monkeypatch):
    from cardiolens.api import service
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/analyze", data={"mode": "A", "sampling_rate_hz": "500"},
                    files=[("files", ("ecg.csv", b"1\n" * 2000, "text/csv"))])
    assert r.status_code == 400 and "too large" in r.json()["detail"]


def test_wfdb_upload_with_its_data_file_is_accepted():
    files = []
    for ext in ("hea", "dat"):
        with open(os.path.join(_SAMPLES, f"sample_ecg.{ext}"), "rb") as fh:
            files.append(("files", (f"sample_ecg.{ext}", fh.read(), "application/octet-stream")))
    r = client.post("/api/analyze", data={"mode": "A"}, files=files)
    assert r.status_code == 200 and r.json()["measurements"]["heart_rate_bpm"] > 0
