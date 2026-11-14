"""The controller process: a loop that watches the monitors and runs ``retrain`` when needed.

``python -m churnsense.controller.main``                 run forever (the compose service)
``python -m churnsense.controller.main --once --trigger manual``  one run now (``make retrain``)

Safety: only one run at a time (a file lock), a cooldown between runs, and every outcome is written
to ``/srv/churnsense/state/runs/`` and pushed to Prometheus. Old inference logs are deleted after
``log_retention_days``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from churnsense.config import Params, Paths, default_paths, load_params, serving_dir
from churnsense.controller.policy import Decision, Signals, decide
from churnsense.controller.runner import OUTCOME_CODE, RunRecord, retrain
from churnsense.deploy import rollback as rollback_module
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.store import read_recent

log = get_logger(__name__)
STATE_FILE, STATUS_FILE, LOCK_FILE = (
    "controller_state.json",
    "controller_status.json",
    "retrain.lock",
)
REQUEST_POLL_SECONDS = 5  # how often the queue of web-console requests is looked at


class BusyError(RuntimeError):
    """Another retraining is already running."""


@contextmanager
def retrain_lock(state_dir: Path) -> Iterator[None]:
    """Hold the retraining lock (non-blocking). Raises ``BusyError`` if somebody else has it."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / LOCK_FILE).open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BusyError("a retraining is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_state(state_dir: Path) -> dict[str, Any]:
    """The controller's memory between runs (last run time, last drift fingerprint, counts)."""
    path = state_dir / STATE_FILE
    base = {"last_run_ts": 0.0, "last_fingerprint": None, "runs": dict.fromkeys(OUTCOME_CODE, 0)}
    return {**base, **json.loads(path.read_text("utf-8"))} if path.is_file() else base


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prune_logs(log_dir: Path, days: int, now: datetime | None = None) -> list[str]:
    """Delete inference-log and feedback files older than ``days`` (by the date in the name)."""
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    removed = []
    for path in sorted(log_dir.glob("*-????-??-??.jsonl")) if log_dir.is_dir() else []:
        try:
            day = datetime.strptime(path.stem[-10:], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        if day < cutoff:
            path.unlink()
            removed.append(path.name)
    return removed


def count_new_feedback(log_dir: Path, state_dir: Path) -> int:
    """Feedback records not yet put into a batch."""
    cursor_file = state_dir / "feedback_cursor.json"
    after = json.loads(cursor_file.read_text("utf-8"))["after"] if cursor_file.is_file() else ""
    return sum(1 for r in read_recent(log_dir, "feedback", 1_000_000) if r["ts"] > after)


def gather(drift_url: str, backend_url: str, log_dir: Path, state_dir: Path) -> Signals:
    """Ask the monitors; a monitor that does not answer counts as no signal (never as a trigger)."""
    drift_status: str | None = None
    drifted: list[str] = []
    performance = None
    try:
        status = httpx.get(f"{drift_url}/status", timeout=5).json()
        drift_status, drifted = status.get("status"), status.get("drifted_features", [])
    except (httpx.HTTPError, ValueError):
        log.warning("drift monitor did not answer")
    try:
        performance = httpx.get(f"{backend_url}/api/v1/performance", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        log.warning("backend did not answer")
    return Signals(
        drift_status, drifted, performance, count_new_feedback(log_dir, state_dir), time.time()
    )


def push_metrics(record: dict[str, Any], state: dict[str, Any]) -> None:
    """Send the run outcome to the Pushgateway (best effort)."""
    gateway = os.environ.get("PUSHGATEWAY_URL", "").strip()
    if not gateway:
        return
    registry = CollectorRegistry()
    Gauge(
        "churnsense_controller_last_outcome",
        "0 failed, 1 rejected, 2 promoted, 3 skipped",
        registry=registry,
    ).set(OUTCOME_CODE[record["outcome"]])
    Gauge(
        "churnsense_controller_last_run_timestamp_seconds",
        "When the last run finished",
        registry=registry,
    ).set(time.time())
    Gauge(
        "churnsense_controller_last_duration_seconds", "Duration of the last run", registry=registry
    ).set(record.get("duration_seconds", 0))
    runs = Gauge(
        "churnsense_controller_runs_total", "Runs by outcome", ["outcome"], registry=registry
    )
    for outcome, count in state["runs"].items():
        runs.labels(outcome).set(count)
    try:
        pushadd_to_gateway(gateway, job="churnsense_controller", registry=registry)
    except Exception as exc:  # noqa: BLE001 - monitoring must never break the controller
        log.warning("could not push controller metrics", extra={"error": str(exc)})


def record_outcome(state_dir: Path, record: dict[str, Any], fingerprint: str | None = None) -> None:
    """Count the run, remember when it happened, save its record and tell Prometheus."""
    state = read_state(state_dir)
    state["runs"][record["outcome"]] = state["runs"].get(record["outcome"], 0) + 1
    state["last_run_ts"] = time.time()
    if fingerprint:
        state["last_fingerprint"] = fingerprint
    write_json(state_dir / STATE_FILE, state)
    write_json(state_dir / "runs" / f"{record['run_id']}.json", record)
    push_metrics(record, state)
    log.info("run finished", extra={"outcome": record["outcome"], "trigger": record["trigger"]})


def execute(
    paths: Paths, cfg: Params, decision: Decision, log_dir: Path, state_dir: Path
) -> dict[str, Any]:
    """Run one retraining under the lock, then record the outcome."""
    with retrain_lock(state_dir):
        record = retrain(
            paths,
            cfg,
            decision.trigger or "manual",
            decision.reason,
            decision.fingerprint,
            log_dir,
            state_dir,
        )
    record_outcome(state_dir, record, decision.fingerprint)
    return record


def run_rollback(state_dir: Path) -> dict[str, Any]:
    """Roll back to the previous model version (requested from the web console) and record it."""
    run = RunRecord("manual-ui", "rollback requested from the web console", None)
    try:
        with retrain_lock(state_dir):  # never roll back in the middle of a retraining
            summary = rollback_module.rollback(serving_dir())
        run.step("rollback", True, f"v{summary['from']} -> v{summary['to']}: {summary['live']}")
        record = run.finish("rolled_back", summary.get("aliases", ""))
    except BusyError as exc:
        run.step("rollback", False, str(exc))
        record = run.finish("skipped", "a retraining is running")
    except Exception as exc:  # noqa: BLE001 - recorded, shown in the console
        run.step("rollback", False, str(exc))
        record = run.finish("failed", "rollback failed")
    record_outcome(state_dir, record)
    return record


def process_requests(
    paths: Paths, cfg: Params, log_dir: Path, state_dir: Path
) -> list[dict[str, Any]]:
    """Carry out the actions queued by the web console (``state/requests/*.json``), oldest first.

    A request file is deleted BEFORE it is carried out, so a crash can never make the same action
    repeat forever.
    """
    queue = state_dir / "requests"
    done = []
    for file in sorted(queue.glob("*.json")) if queue.is_dir() else []:
        try:
            action = json.loads(file.read_text("utf-8")).get("action")
        except (OSError, ValueError):
            action = None
        file.unlink(missing_ok=True)
        if action == "retrain":
            try:
                done.append(
                    execute(
                        paths,
                        cfg,
                        Decision("manual-ui", "requested from the web console"),
                        log_dir,
                        state_dir,
                    )
                )
            except BusyError:
                run = RunRecord("manual-ui", "retraining requested from the web console", None)
                run.step("lock", False, "another retraining is already running")
                record = run.finish("skipped", "another retraining is running")
                record_outcome(state_dir, record)
                done.append(record)
        elif action == "rollback":
            done.append(run_rollback(state_dir))
        else:
            log.warning("ignored an unknown request", extra={"file": file.name})
    return done


def loop(
    paths: Paths, cfg: Params, drift_url: str, backend_url: str, log_dir: Path, state_dir: Path
) -> None:
    """Watch forever: console requests every few seconds, the retrain policy every minute."""
    next_check = 0.0
    while True:
        try:
            process_requests(paths, cfg, log_dir, state_dir)
            if time.time() >= next_check:
                next_check = time.time() + float(cfg.controller.poll_seconds)
                prune_logs(log_dir, int(cfg.controller.log_retention_days))
                signals = gather(drift_url, backend_url, log_dir, state_dir)
                decision = decide(signals, read_state(state_dir), cfg.controller)
                write_json(
                    state_dir / STATUS_FILE,
                    {
                        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "drift_status": signals.drift_status,
                        "new_feedback_rows": signals.new_feedback_rows,
                        "decision": decision.trigger or "no retraining",
                        "reason": decision.reason,
                    },
                )
                if decision.trigger:
                    execute(paths, cfg, decision, log_dir, state_dir)
            (state_dir / STATUS_FILE).touch(exist_ok=True)  # the heartbeat the health checks read
        except BusyError:
            log.info("a retraining is already running")
        except Exception as exc:  # noqa: BLE001 - the watcher must survive any single failure
            log.error("controller round failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        time.sleep(REQUEST_POLL_SECONDS)


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description="ChurnSense retrain controller")
    parser.add_argument("--once", action="store_true", help="run one retraining now and exit")
    parser.add_argument("--trigger", default="manual")
    args = parser.parse_args()
    setup_logging()
    paths, cfg = default_paths(), load_params()
    log_dir = Path(os.environ.get("INFERENCE_LOG_DIR", "/srv/churnsense/inference-logs"))
    state_dir = Path(os.environ.get("STATE_DIR", "/srv/churnsense/state"))
    if args.once:
        try:
            record = execute(
                paths, cfg, Decision(args.trigger, "requested by the operator"), log_dir, state_dir
            )
        except BusyError as exc:
            print(f"REFUSED: {exc}")
            raise SystemExit(3) from exc
        print(
            json.dumps(
                {k: record[k] for k in ("run_id", "trigger", "outcome", "note", "duration_seconds")}
            )
        )
        raise SystemExit(0 if record["outcome"] != "failed" else 1)
    loop(
        paths,
        cfg,
        os.environ.get("DRIFT_URL", "http://drift-monitor:8002"),
        os.environ.get("BACKEND_URL", "http://backend:8000"),
        log_dir,
        state_dir,
    )


if __name__ == "__main__":
    main()
