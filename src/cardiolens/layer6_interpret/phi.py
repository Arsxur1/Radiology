"""De-identification of the payload sent to a cloud LLM (TZ §12).

The core (layers 1-5) runs locally; only Layer 6 may call a cloud model. What leaves
the machine is built from an allow-list of derived fields (findings, causal edges,
four measurements) — never the patient, the clinical context (medications, symptoms,
prior ECG id) or the source file. As defence in depth every string in the payload is
also scrubbed for identifier-shaped text (e-mail, phone, dates, long numeric ids,
passport-style ids), in case a future rule or an upstream parser lets one into a
finding label or evidence value (D91).
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Keys allowed at the top level of the cloud payload.
ALLOWED_KEYS = frozenset({
    "mode", "grounding_tokens", "findings", "causal_edges", "critical_flags",
    "implausible_flags", "confidence", "measurements",
})

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Dates: 12.03.1961, 12/03/61, 1961-03-12.
_DATE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2})\b")
# Passport / national-id style: two letters and 7 digits (e.g. AA1234567).
_PASSPORT = re.compile(r"\b[A-Z]{2}\d{7}\b")
# Phone candidates: digits with spaces, dashes or parentheses, optional leading +;
# kept only with >= 9 digits so ECG numbers ("-28.0 -273.0", "74.1") never match.
_PHONE = re.compile(r"\+?\(?\d[\d\s()-]{7,}\d")
# Record / medical-record numbers: 7+ digits in a row.
_LONG_ID = re.compile(r"\b\d{7,}\b")


def _phone_sub(m: re.Match[str]) -> str:
    return REDACTED if sum(c.isdigit() for c in m.group(0)) >= 9 else m.group(0)


def scrub_text(text: str) -> tuple[str, int]:
    """Replace identifier-shaped substrings; return the text and the count replaced."""
    n = 0
    for pat in (_EMAIL, _DATE, _PASSPORT):
        text, k = pat.subn(REDACTED, text)
        n += k
    before = text.count(REDACTED)
    text = _PHONE.sub(_phone_sub, text)
    n += text.count(REDACTED) - before
    text, k = _LONG_ID.subn(REDACTED, text)
    return text, n + k


def _walk(obj: Any, counter: list[int]) -> Any:
    if isinstance(obj, str):
        out, k = scrub_text(obj)
        counter[0] += k
        return out
    if isinstance(obj, dict):
        return {k: _walk(v, counter) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, counter) for v in obj]
    return obj


def deidentify(payload: dict) -> tuple[dict, int]:
    """Drop non-allow-listed top-level keys and scrub every string.

    Returns the cleaned payload and the number of redactions (dropped keys count too),
    so the caller can record in the audit trail that something was removed.
    """
    dropped = [k for k in payload if k not in ALLOWED_KEYS]
    counter = [len(dropped)]
    clean = {k: _walk(v, counter) for k, v in payload.items() if k in ALLOWED_KEYS}
    return clean, counter[0]
