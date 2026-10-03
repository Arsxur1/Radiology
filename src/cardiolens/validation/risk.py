"""Error analysis by clinical risk tier (TZ §16; application review, blocker 1).

The per-class table of the harness answers "how often is each finding right".
A clinical review asks a different question: "how often does the system miss
something that matters, and how often does it raise an alarm that costs an
urgent read". Finding codes are therefore grouped into tiers of consequence
(``configs/risk_tiers.yaml``: critical / major / minor) and every miss and
false call is pooled by tier, twice:

* **label level** — confusion counts summed over the tier's codes (a case with
  two critical gold labels contributes two positives);
* **case level** — a case carrying any gold label of the tier counts as caught
  when the system reports any label of the tier. This is the escalation view:
  in mode B any critical finding forces a physician read, so a STEMI reported
  as "long QT" still reaches a physician, and a false critical call costs one
  urgent read whatever its code.

Against the open databases' labels these numbers describe agreement with the
label set, not clinical outcome: the labels are not adjudicated. The targets in
the yaml are the protocol's acceptance thresholds for an adjudicated held-out
set and are reported against, never claimed, on database labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cardiolens.config import load_risk_tiers
from cardiolens.validation.metrics import ClassMetrics


@dataclass
class TierMetrics:
    tier: str
    codes: list[str]
    pooled: ClassMetrics                       # label-level counts over the tier's codes
    case_positive: int = 0                     # cases with any gold label of the tier
    case_caught: int = 0                       # ... that received any label of the tier
    case_negative: int = 0                     # cases with no gold label of the tier
    case_false_alert: int = 0                  # ... that received a label of the tier anyway
    missed_by_code: dict[str, int] = field(default_factory=dict)
    false_by_code: dict[str, int] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)

    @property
    def case_sensitivity(self) -> float | None:
        return round(self.case_caught / self.case_positive, 3) if self.case_positive else None

    @property
    def case_false_alert_rate(self) -> float | None:
        return round(self.case_false_alert / self.case_negative, 3) if self.case_negative else None

    def meets(self) -> dict[str, bool | None]:
        """Target checks; None where the metric has no support."""
        out: dict[str, bool | None] = {}
        for key, limit in self.targets.items():
            value = getattr(self, key, None)
            if value is None:
                out[key] = None
            elif key.endswith("rate"):
                out[key] = value <= limit
            else:
                out[key] = value >= limit
        return out

    def to_dict(self) -> dict:
        return {
            "tier": self.tier, "codes": list(self.codes),
            "pooled": self.pooled.to_dict(),
            "case_positive": self.case_positive, "case_caught": self.case_caught,
            "case_sensitivity": self.case_sensitivity,
            "case_negative": self.case_negative, "case_false_alert": self.case_false_alert,
            "case_false_alert_rate": self.case_false_alert_rate,
            "missed_by_code": dict(self.missed_by_code), "false_by_code": dict(self.false_by_code),
            "targets": dict(self.targets), "meets": self.meets(),
        }


def score_risk_tiers(predictions: list[set[str]], gold: list[set[str]],
                     labels: list[str] | None = None, tiers: dict | None = None) -> dict[str, TierMetrics]:
    """Pool the per-case prediction / gold sets by risk tier.

    ``labels`` restricts the codes counted to those the dataset scores (a tier
    code the dataset never labels contributes nothing); ``tiers`` defaults to
    the yaml."""
    cfg = tiers if tiers is not None else load_risk_tiers()
    targets = cfg.get("targets", {})
    out: dict[str, TierMetrics] = {}
    for tier, codes in cfg["tiers"].items():
        codes = [c for c in codes if labels is None or c in labels]
        tm = TierMetrics(tier=tier, codes=codes, pooled=ClassMetrics(label=tier),
                         targets=dict(targets.get(tier, {})))
        code_set = set(codes)
        if not code_set:                       # the dataset labels none of this tier: no metric
            out[tier] = tm
            continue
        for pred, truth in zip(predictions, gold, strict=True):
            for code in codes:
                p, t = code in pred, code in truth
                if p and t:
                    tm.pooled.tp += 1
                elif p:
                    tm.pooled.fp += 1
                    tm.false_by_code[code] = tm.false_by_code.get(code, 0) + 1
                elif t:
                    tm.pooled.fn += 1
                    tm.missed_by_code[code] = tm.missed_by_code.get(code, 0) + 1
                else:
                    tm.pooled.tn += 1
            if truth & code_set:
                tm.case_positive += 1
                tm.case_caught += bool(pred & code_set)
            else:
                tm.case_negative += 1
                tm.case_false_alert += bool(pred & code_set)
        out[tier] = tm
    return out


def risk_report_lines(risk: dict[str, TierMetrics]) -> list[str]:
    """Human-readable tier block for ``validate``."""
    lines = []
    for tier, tm in risk.items():
        meets = tm.meets()
        mark = {True: "✓", False: "✗", None: "–"}
        s = tm.pooled.sensitivity
        lines.append(
            f"{tier:9s} case sens={tm.case_sensitivity} ({tm.case_caught}/{tm.case_positive}) "
            f"{mark[meets.get('case_sensitivity')]}  case false-alert={tm.case_false_alert_rate} "
            f"({tm.case_false_alert}/{tm.case_negative}) {mark[meets.get('case_false_alert_rate')]}  "
            f"| label sens={s} spec={tm.pooled.specificity}"
        )
        if tm.missed_by_code:
            missed = ", ".join(f"{c} {n}" for c, n in sorted(tm.missed_by_code.items(), key=lambda x: -x[1]))
            lines.append(f"{'':9s}   missed: {missed}")
        if tm.false_by_code:
            false = ", ".join(f"{c} {n}" for c, n in sorted(tm.false_by_code.items(), key=lambda x: -x[1]))
            lines.append(f"{'':9s}   false:  {false}")
    lines.append("targets are the protocol's thresholds for an adjudicated set; on database labels they are reported, not claimed")
    return lines
