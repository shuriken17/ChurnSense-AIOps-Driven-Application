"""Common wrapper for every pipeline stage.

``@tracked_stage("ingest", report="ingest_stats.json")`` adds, around a stage function:

* start/end logging and a stopwatch,
* a JSON report with the stage's numbers, duration and rows per second (rubric: pipeline
  speed and throughput),
* metrics pushed to the Prometheus Pushgateway (if PUSHGATEWAY_URL is set),
* a non-zero exit code when the stage fails, so DVC stops and skips downstream stages.

Pushing metrics must never break the build: any push error is caught and logged as a
warning. A stage that detects bad data raises ``StageError`` (with a report), which is
written to disk so the failure can be diagnosed.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from churnsense.config import ConfigError, Params, Paths, default_paths, load_params
from churnsense.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

PUSH_JOB = "churnsense_pipeline"
PUSH_TIMEOUT_S = 3


@dataclass
class StageReport:
    """Numbers a stage reports about its own work."""

    rows_in: int = 0
    rows_out: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class StageError(RuntimeError):
    """A stage found a problem it must not continue past (bad data, broken guardrail)."""

    def __init__(self, message: str, report: StageReport | None = None) -> None:
        super().__init__(message)
        self.report = report or StageReport()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def push_stage_metrics(stage: str, *, ok: bool, duration: float, rows: int) -> bool:
    """Push the stage metrics to the Pushgateway; return True if the push worked.

    ``pushadd`` (HTTP POST) only replaces metrics with the same names, so after a failed
    run the old ``last_success_timestamp`` stays in place.
    """
    gateway = os.environ.get("PUSHGATEWAY_URL", "").strip()
    if not gateway:
        log.debug("PUSHGATEWAY_URL not set; metrics not pushed")
        return False
    try:
        registry = CollectorRegistry()
        Gauge("churnsense_stage_duration_seconds", "Stage wall time.", registry=registry).set(
            duration
        )
        Gauge(
            "churnsense_stage_last_status",
            "1 if the last run succeeded, 0 if it failed.",
            registry=registry,
        ).set(1 if ok else 0)
        if ok:
            Gauge(
                "churnsense_stage_rows_total", "Rows written by the stage.", registry=registry
            ).set(rows)
            Gauge(
                "churnsense_stage_rows_per_second", "Rows written per second.", registry=registry
            ).set(rows / duration if duration > 0 else 0)
            Gauge(
                "churnsense_stage_last_success_timestamp",
                "Unix time of the last success.",
                registry=registry,
            ).set(time.time())
        pushadd_to_gateway(
            gateway,
            job=PUSH_JOB,
            registry=registry,
            grouping_key={"stage": stage},
            timeout=PUSH_TIMEOUT_S,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - a push problem must never fail the build
        log.warning(
            "could not push metrics to the Pushgateway", extra={"stage": stage, "error": str(exc)}
        )
        return False


def _report_payload(
    stage: str,
    status: str,
    started_iso: str,
    duration: float,
    report: StageReport,
    error: str | None,
) -> dict[str, Any]:
    rows_per_second = report.rows_out / duration if duration > 0 else 0.0
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "started_at": started_iso,
        "duration_seconds": round(duration, 3),
        "rows_in": report.rows_in,
        "rows_out": report.rows_out,
        "rows_per_second": round(rows_per_second, 1),
        **report.extra,
    }
    if error:
        payload["error"] = error
    return payload


StageFunction = Callable[[Params, Paths], StageReport]


def tracked_stage(stage: str, report: str | None = None) -> Callable[[StageFunction], Any]:
    """Wrap a stage function ``f(params, paths) -> StageReport`` as described above.

    The wrapped function keeps its signature and raises on failure (handy for tests).
    Its ``.main()`` attribute is the command-line entry point: it loads params.yaml, runs
    the stage and exits with code 1 on failure.
    """

    def decorator(func: StageFunction) -> Any:
        @functools.wraps(func)
        def wrapper(params: Params, paths: Paths) -> StageReport:
            started = time.perf_counter()
            started_iso = datetime.now(UTC).isoformat(timespec="seconds")
            log.info("stage started", extra={"stage": stage})
            try:
                result = func(params, paths)
            except StageError as failure:
                duration = time.perf_counter() - started
                log.error("stage failed", extra={"stage": stage, "error": str(failure)})
                if report:
                    write_json(
                        paths.report(report),
                        _report_payload(
                            stage, "failed", started_iso, duration, failure.report, str(failure)
                        ),
                    )
                push_stage_metrics(stage, ok=False, duration=duration, rows=0)
                raise
            except Exception as exc:
                duration = time.perf_counter() - started
                log.exception("stage crashed", extra={"stage": stage})
                if report:
                    write_json(
                        paths.report(report),
                        _report_payload(
                            stage,
                            "failed",
                            started_iso,
                            duration,
                            StageReport(),
                            f"{type(exc).__name__}: {exc}",
                        ),
                    )
                push_stage_metrics(stage, ok=False, duration=duration, rows=0)
                raise
            duration = time.perf_counter() - started
            if report:
                write_json(
                    paths.report(report),
                    _report_payload(stage, "ok", started_iso, duration, result, None),
                )
            pushed = push_stage_metrics(stage, ok=True, duration=duration, rows=result.rows_out)
            log.info(
                "stage finished",
                extra={
                    "stage": stage,
                    "rows_out": result.rows_out,
                    "duration_seconds": round(duration, 2),
                    "metrics_pushed": pushed,
                },
            )
            return result

        def main(configure: Callable[[Params], None] | None = None) -> None:
            """Command-line entry point: ``python -m churnsense.pipeline.<stage>``.

            ``configure`` may change the loaded parameters first (used for command-line flags).
            """
            setup_logging()
            try:
                params = load_params()
                if configure:
                    configure(params)
                wrapper(params, default_paths())
            except ConfigError as exc:
                log.error("configuration problem", extra={"error": str(exc)})
                sys.exit(2)
            except Exception:  # noqa: BLE001 - already logged and reported by the wrapper
                sys.exit(1)

        wrapper.main = main  # type: ignore[attr-defined]
        wrapper.stage_name = stage  # type: ignore[attr-defined]
        return wrapper

    return decorator
