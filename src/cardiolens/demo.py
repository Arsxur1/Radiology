"""End-to-end demo runner (TZ §18 п.4 — pipeline on one example).

Usage:
    python -m cardiolens.demo                     # synthetic normal ECG
    python -m cardiolens.demo --case hyperk       # hyperkalemia pattern
    python -m cardiolens.demo --wfdb path/to/rec  # a real WFDB/PTB-XL record
    python -m cardiolens.demo --mode B            # request autonomous mode

Writes all five output formats (TZ §11) to ``outputs/``.
"""

from __future__ import annotations

import argparse

from cardiolens.layer1_ingestion import generate_synthetic_ecg, load_wfdb
from cardiolens.outputs import format_conclusion, write_all_outputs
from cardiolens.pipeline import run_pipeline

_CASES = {
    "normal": dict(heart_rate_bpm=72),
    "brady": dict(heart_rate_bpm=44),
    "tachy": dict(heart_rate_bpm=125),
    "hyperk": dict(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6),
    "stemi": dict(heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]),
    "child": dict(heart_rate_bpm=95, age_years=6),
}


def _run_stream_demo(mode: str | None) -> None:
    """Replay a recording that develops a STEMI mid-stream (TZ §4)."""
    import numpy as np

    from cardiolens.contracts.signal import SignalRepresentation
    from cardiolens.streaming import StreamProcessor

    normal = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=20)
    stemi = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=20, seed=99,
                                   st_elevation_uv=250, st_leads=["II", "III", "aVF"])
    rec = SignalRepresentation(
        signal=np.concatenate([normal.signal, stemi.signal], axis=1),
        sampling_rate_hz=500, lead_names=normal.lead_names, duration_s=40,
        source="synthetic",
    )
    sp = StreamProcessor(rec.lead_names, 500, window_s=10, hop_s=5, mode=mode or "B")
    print("Потоковый режим (§4): 10-с окна, шаг 5 с; STEMI развивается на 20-й секунде\n")
    for wr in sp.process_signal(rec, chunk_s=2.0):
        flag = f"  ⚠ НОВАЯ критическая: {wr.new_critical}" if wr.new_critical else ""
        print(f"[{wr.t_start_s:>5}-{wr.t_end_s:>5}s] режим={wr.result.report.mode:9} "
              f"находки={wr.finding_codes}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CardioLens end-to-end demo")
    ap.add_argument("--case", choices=sorted(_CASES), default="normal")
    ap.add_argument("--wfdb", help="path to a WFDB record (no extension)")
    ap.add_argument("--mode", choices=["A", "B"], default=None)
    ap.add_argument("--stream", action="store_true",
                    help="streaming demo: STEMI develops mid-recording (TZ §4)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.stream:
        _run_stream_demo(args.mode)
        return

    if args.wfdb:
        sig = load_wfdb(args.wfdb)
        stem = "record"
    else:
        sig = generate_synthetic_ecg(**_CASES[args.case])
        stem = args.case

    result = run_pipeline(sig, mode=args.mode)
    paths = write_all_outputs(result, args.out, stem=stem)

    print(format_conclusion(result, "ru"))
    print("\nВыходные файлы (TZ §11):")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
