"""Cross-cutting safety layer — critical findings (TZ §10)."""

from cardiolens.safety.critical import CriticalAlert, SafetyResult, scan_critical

__all__ = ["scan_critical", "SafetyResult", "CriticalAlert"]
