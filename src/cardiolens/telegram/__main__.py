"""CLI of the Telegram collector.

    python -m cardiolens.telegram chats      # log in (first time) and list your groups with their ids
    python -m cardiolens.telegram run        # listen, send cards, record verdicts
    python -m cardiolens.telegram backfill   # one pass over recent group history, then exit
    python -m cardiolens.telegram stats      # counters and reviewer agreement, offline
    python -m cardiolens.telegram export     # harness report on the verified records, offline
    python -m cardiolens.telegram analyse photo.jpg   # one photo through the same path, offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from cardiolens.telegram.config import load_telegram_config
from cardiolens.telegram.store import DatasetStore


def main() -> None:
    ap = argparse.ArgumentParser(description="CardioLens Telegram collector")
    ap.add_argument("command", choices=["chats", "run", "backfill", "stats", "export", "analyse"])
    ap.add_argument("path", nargs="?", help="photo for `analyse`")
    ap.add_argument("--config", help="alternative telegram.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_telegram_config(args.config)

    if args.command == "chats":
        from cardiolens.telegram.agent import list_chats

        lines = asyncio.run(list_chats(cfg))
        print(f"{'id':>16}  kind     title  @username   -> paste ids or @usernames into configs/telegram.yaml `groups`")
        print("\n".join(lines) if lines else "(no groups or channels on this account)")
        return

    if args.command in ("run", "backfill"):
        from cardiolens.telegram.agent import run_agent

        asyncio.run(run_agent(cfg, backfill_only=args.command == "backfill"))
        return

    if args.command == "analyse":
        from cardiolens.telegram.analyze import analyse_photo, format_card, is_ecg_photo
        from cardiolens.telegram.store import Record

        ok, pitch = is_ecg_photo(args.path, cfg.min_grid_pitch_px)
        print(f"grid pitch: {pitch} px/mm -> {'ECG' if ok else 'no ECG grid found'}")
        result, error = analyse_photo(args.path, mode=cfg.mode)
        rec = Record(id="local", chat_id=0, msg_id=0, chat_title="local file", photo_path=args.path)
        print(format_card(rec, result, error))
        return

    store = DatasetStore(cfg.store_dir)
    from cardiolens.telegram.export import build_telegram_dataset, summary_lines

    print(f"=== CardioLens Telegram collector — {cfg.store_dir} ===")
    for line in summary_lines(store):
        print(line)
    if args.command == "export":
        from cardiolens.validation import risk_report_lines, run_validation

        cases = build_telegram_dataset(cfg.store_dir)
        if not cases:
            print("\nno verified records with a digitised signal yet")
            return
        report = run_validation(cases)
        print(f"\nharness on {report.n_cases} verified photo(s): macro sensitivity {report.macro_sensitivity}")
        for lbl, m in report.per_class.items():
            if m.support:
                print(f"  {lbl:24s} sens={m.sensitivity} spec={m.specificity} n={m.support}")
        print("risk tiers:")
        for line in risk_report_lines(report.risk):
            print("  " + line)
        print("\nJSON:")
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
