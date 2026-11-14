"""When to retrain: a pure function of what the monitors report, easy to test.

Triggers (either one):

* ``drift``        the drift monitor says ``alert`` (PSI at or above 0.25 several checks in a row)
* ``performance``  recall of the HIGH band on reported outcomes fell below the floor, with enough
                   labelled outcomes to trust the number

Guards, in this order (the first one that applies decides):

1. ``cooldown``      the last run was less than ``cooldown_hours`` ago
2. ``already handled`` the same drift episode (same features, same day) was already retrained
3. ``not enough new feedback`` retraining on the same data would give the same model

A manual retraining skips the triggers but still respects the lock.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Signals:
    """What the controller learned this round (None means that monitor did not answer)."""

    drift_status: str | None
    drifted_features: list[str] = field(default_factory=list)
    performance: dict[str, Any] | None = None
    new_feedback_rows: int = 0
    now: float = 0.0


@dataclass(frozen=True)
class Decision:
    """The outcome of ``decide``: a trigger to run, or the reason not to."""

    trigger: str | None
    reason: str
    fingerprint: str | None = None


def fingerprint(features: list[str], now: float) -> str:
    """Identify one drift episode: the drifted features and the day."""
    day = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
    return hashlib.sha1(f"{day}:{','.join(sorted(features))}".encode()).hexdigest()[:12]  # noqa: S324


def decide(signals: Signals, state: dict[str, Any], cfg: Any) -> Decision:
    """Decide whether to retrain now."""
    drift = signals.drift_status == "alert"
    perf = signals.performance or {}
    recall = perf.get("recall")
    weak = bool(
        perf.get("enough_data")
        and perf.get("n", 0) >= int(cfg.perf_min_rows)
        and recall is not None
        and recall < float(cfg.recall_floor)
    )
    if not drift and not weak:
        return Decision(None, "healthy: no drift alert and recall is acceptable")
    trigger = "drift" if drift else "performance"
    reason = (
        f"drift alert on {', '.join(signals.drifted_features[:3]) or 'several features'}"
        if drift
        else f"recall {recall:.2f} is below the floor {float(cfg.recall_floor):.2f}"
    )
    print_id = fingerprint(signals.drifted_features, signals.now) if drift else None
    since = signals.now - float(state.get("last_run_ts", 0))
    if since < float(cfg.cooldown_hours) * 3600:
        return Decision(None, f"cooldown: last run {since / 3600:.1f} h ago ({reason})", print_id)
    if drift and print_id == state.get("last_fingerprint"):
        return Decision(None, f"already handled: same drift episode ({reason})", print_id)
    if signals.new_feedback_rows < int(cfg.min_new_feedback):
        return Decision(
            None,
            f"not enough new feedback: {signals.new_feedback_rows} of "
            f"{int(cfg.min_new_feedback)} needed ({reason})",
            print_id,
        )
    return Decision(trigger, reason, print_id)
