"""Fetch a label-balanced subset of the PhysioNet ecg-arrhythmia dataset (TZ §15, §16).

    python -m cardiolens.validation.fetch_physionet --out data/ecg-arrhythmia-balanced \\
        --folders 60 --per-class 80 --seed 0

The dataset (Chapman-Shaoxing + Ningbo, ~45k 12-lead records) is licensed on
PhysioNet and never committed. ``RECORDS`` lists 452 folders; each folder's index
page lists its ``JSxxxxx.hea``/``.mat`` pairs. Headers are ~1 KB, so the script
downloads *headers* for a spread of folders, maps their ``#Dx:`` SNOMED labels to
CardioLens finding codes, picks a per-class-capped selection (rarest classes
first) and only then downloads the ``.mat`` files it needs. A ``manifest.json``
records what was selected and why. Needs outbound access to ``physionet.org``.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cardiolens.validation.harness import parse_dx_from_header, snomed_dx_to_codes

BASE_URL = "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
UNLABELED = "_unlabeled"   # pseudo-class: records with no mapped label (negatives)
_HEA_RE = re.compile(r'href="(JS\d+)\.hea"')


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if not cafile and Path("/root/.ccr/ca-bundle.crt").exists():
        cafile = "/root/.ccr/ca-bundle.crt"
    return ssl.create_default_context(cafile=cafile)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _KeepAlive:
    """One persistent HTTPS connection per thread (through the CONNECT proxy when set).

    Opening a fresh TLS tunnel per request costs ~1 s each through the egress
    proxy; keeping the connection alive makes thousands of small header requests
    tractable. Reconnects on any error, with backoff.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _connect(self) -> http.client.HTTPSConnection:
        host = urllib.parse.urlparse(BASE_URL).hostname or "physionet.org"
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        ctx = _ssl_context()
        if proxy:
            u = urllib.parse.urlparse(proxy)
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, context=ctx, timeout=90)
            headers = {}
            if u.username:
                token = base64.b64encode(f"{u.username}:{u.password or ''}".encode()).decode()
                headers["Proxy-Authorization"] = f"Basic {token}"
            conn.set_tunnel(host, 443, headers=headers)
        else:
            conn = http.client.HTTPSConnection(host, 443, context=ctx, timeout=90)
        return conn

    def get(self, url: str, attempts: int = 8, max_bytes: int | None = None) -> bytes:
        """GET the body; with ``max_bytes`` only the first bytes (HTTP Range)."""
        path = urllib.parse.urlparse(url).path
        delay = 1.0
        headers = {"Connection": "keep-alive"}
        if max_bytes is not None:
            headers["Range"] = f"bytes=0-{max_bytes - 1}"
        for i in range(attempts):
            conn = getattr(self._local, "conn", None)
            try:
                if conn is None:
                    conn = self._local.conn = self._connect()
                conn.request("GET", path, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status not in (200, 206):
                    raise http.client.HTTPException(f"HTTP {resp.status} for {url}")
                return data[:max_bytes] if max_bytes is not None else data
            except (OSError, http.client.HTTPException) as e:
                self._local.conn = None
                if i == attempts - 1:
                    raise
                _log(f"[fetch] retry {i + 1}/{attempts - 1} after {type(e).__name__}: {url}")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)   # the egress proxy resets in bursts; wait them out
        raise RuntimeError("unreachable")


_CLIENT = _KeepAlive()


def fetch_bytes(url: str, max_bytes: int | None = None) -> bytes:
    return _CLIENT.get(url, max_bytes=max_bytes)


def parse_age_from_header(text: str) -> float | None:
    """``#Age: 7`` -> 7.0 (None when absent or NaN)."""
    m = re.search(r"^#\s*Age:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_records(text: str) -> list[str]:
    """Folder paths from the dataset's RECORDS file (e.g. ``WFDBRecords/01/010/``)."""
    return [ln.strip().rstrip("/") + "/" for ln in text.splitlines() if ln.strip()]


def parse_index(html: str) -> list[str]:
    """Record names (``JS00001``) listed on a folder's index page."""
    return sorted(set(_HEA_RE.findall(html)))


def choose_folders(folders: list[str], n: int, seed: int = 0) -> list[str]:
    """Spread ``n`` folders evenly across the top-level directories (seeded)."""
    by_top: dict[str, list[str]] = defaultdict(list)
    for f in folders:
        by_top[f.split("/")[1]].append(f)
    rng = random.Random(seed)
    for lst in by_top.values():
        rng.shuffle(lst)
    tops = sorted(by_top)
    out: list[str] = []
    while len(out) < n and any(by_top.values()):
        for t in tops:
            if by_top[t] and len(out) < n:
                out.append(by_top[t].pop())
    return out


def select_balanced(records: dict[str, set[str]], per_class: int, seed: int = 0) -> list[str]:
    """Pick records so every class has at most ``per_class`` members, rarest first.

    A record counts toward every class it carries, so common classes fill up as a
    side effect of collecting rare ones. Records without a mapped label form the
    pseudo-class ``_unlabeled`` (useful negatives) with the same cap.
    """
    labels = {r: (labs or {UNLABELED}) for r, labs in records.items()}
    counts = Counter(c for labs in labels.values() for c in labs)
    rng = random.Random(seed)
    selected: set[str] = set()
    tally: Counter = Counter()
    for cls, _ in sorted(counts.items(), key=lambda kv: (kv[1], kv[0])):
        cands = [r for r, labs in labels.items() if cls in labs and r not in selected]
        rng.shuffle(cands)
        for r in cands:
            if tally[cls] >= per_class:
                break
            selected.add(r)
            tally.update(labels[r])
    return sorted(selected)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch a balanced ecg-arrhythmia subset")
    ap.add_argument("--out", required=True, help="destination directory (git-ignored)")
    ap.add_argument("--folders", type=int, default=60)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--age-max", type=float, default=None,
                    help="pediatric mode: take EVERY record whose #Age is <= this (no per-class cap)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_folders = parse_records(fetch_bytes(BASE_URL + "RECORDS").decode())
    folders = all_folders if args.folders <= 0 else choose_folders(all_folders, args.folders, args.seed)
    _log(f"[fetch] {len(folders)} folders chosen")

    # 1. Headers for every record in the chosen folders.
    headers: dict[str, tuple[str, str]] = {}  # record -> (folder, header text)

    # Headers are cached on disk so an interrupted run resumes instead of refetching.
    cache = out / ".headers"

    def fetch_folder(folder: str) -> list[tuple[str, str, str]]:
        cdir = cache / folder
        done = cdir / ".complete"
        if done.exists():
            return [(f.stem, folder, f.read_text(encoding="utf-8")) for f in sorted(cdir.glob("*.hea"))]
        cdir.mkdir(parents=True, exist_ok=True)
        names = parse_index(fetch_bytes(BASE_URL + folder).decode())
        rows = []
        for nm in names:
            f = cdir / f"{nm}.hea"
            if not f.exists():
                f.write_text(fetch_bytes(BASE_URL + folder + nm + ".hea").decode(), encoding="utf-8")
            rows.append((nm, folder, f.read_text(encoding="utf-8")))
        done.touch()
        return rows

    with ThreadPoolExecutor(args.workers) as pool:
        for i, batch in enumerate(pool.map(fetch_folder, folders), 1):
            for nm, folder, text in batch:
                headers[nm] = (folder, text)
            _log(f"[fetch] headers {i}/{len(folders)} folders, {len(headers)} records")

    # 2. Balanced selection by mapped label — or every child in pediatric mode.
    labels = {nm: snomed_dx_to_codes(parse_dx_from_header(text)) for nm, (_, text) in headers.items()}
    if args.age_max is not None:
        ages = {nm: parse_age_from_header(text) for nm, (_, text) in headers.items()}
        chosen = sorted(nm for nm, age in ages.items() if age is not None and age <= args.age_max)
        _log(f"[fetch] pediatric mode: {len(chosen)} records with age <= {args.age_max}")
    else:
        chosen = select_balanced(labels, args.per_class, args.seed)
    support = Counter(c for nm in chosen for c in (labels[nm] or {UNLABELED}))
    _log(f"[fetch] selected {len(chosen)} of {len(headers)} records; support: {dict(support)}")

    # 3. Signals for the selection only.
    def fetch_record(nm: str) -> str:
        folder, text = headers[nm]
        dest = out / folder
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{nm}.hea").write_text(text, encoding="utf-8")
        mat = dest / f"{nm}.mat"
        if not mat.exists():
            mat.write_bytes(fetch_bytes(BASE_URL + folder + nm + ".mat"))
        return nm

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(fetch_record, chosen), 1):
            if i % 50 == 0 or i == len(chosen):
                _log(f"[fetch] signals {i}/{len(chosen)}")

    manifest = {
        "source": BASE_URL, "seed": args.seed, "folders": folders, "per_class": args.per_class,
        "records": [{"record": nm, "folder": headers[nm][0], "labels": sorted(labels[nm]),
                     "age": parse_age_from_header(headers[nm][1])}
                    for nm in chosen],
        "support": dict(support),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"[fetch] done -> {out}")


if __name__ == "__main__":
    main()
