"""Drift monitor service: compares live traffic with the training profile, every minute.

It reads the latest predictions from the inference log, computes the PSI of every feature
against ``reference_profile.json`` and exposes the result:

* ``GET /status``   the latest verdict (JSON), read by the retrain controller
* ``POST /check``   run a check now (used by tests and demos; the loop also runs by itself)
* ``GET /metrics``  Prometheus gauges (PSI per feature, status, window size, mean score)
* ``GET /health``

Verdicts: ``insufficient_data`` (too few rows), ``ok``, ``warn`` (some PSI at or above 0.10),
``alert_candidate`` (some PSI at or above 0.25, but not yet for long enough) and ``alert``
(at or above 0.25 for ``consecutive_checks`` checks in a row). Requiring several checks in a row
stops a single odd minute from triggering a retraining.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

from churnsense.config import Params, load_params
from churnsense.drift.psi import DriftReport, compute_drift
from churnsense.features.derive import derive_features
from churnsense.features.profile import load_profile
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.schema import INPUT_COLUMNS
from churnsense.store import read_recent

log = get_logger(__name__)
STATUS_CODE = {"insufficient_data": -1, "ok": 0, "warn": 1, "alert_candidate": 2, "alert": 3}


class DriftMonitor:
    """The checking logic, separate from the web layer so it can be tested directly."""

    def __init__(self, log_dir: Path, profile_path: Path, params: Params) -> None:
        self.log_dir, self.profile_path, self.cfg = log_dir, profile_path, params.drift
        self.streak = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=200)
        self.latest: dict[str, Any] = {
            "status": "insufficient_data",
            "checked_at": None,
            "n_rows": 0,
        }
        self._lock = threading.Lock()

    def check(self) -> dict[str, Any]:
        """Run one drift check and update the status and the streak."""
        profile = load_profile(self.profile_path)
        records = read_recent(self.log_dir, "predictions", int(self.cfg.window_rows))
        frame = pd.DataFrame([r["inputs"] for r in records])
        for column in INPUT_COLUMNS:  # optional fields may be absent from the logged inputs
            if column not in frame:
                frame[column] = np.nan
        features = derive_features(frame) if len(frame) else frame
        report: DriftReport = compute_drift(
            profile, features, min_rows=int(self.cfg.min_rows), psi_alert=float(self.cfg.psi_alert)
        )
        if not report.enough_data:
            self.streak, status = 0, "insufficient_data"
        elif report.max_psi >= float(self.cfg.psi_alert):
            self.streak += 1
            status = (
                "alert" if self.streak >= int(self.cfg.consecutive_checks) else "alert_candidate"
            )
        else:
            self.streak = 0
            status = "warn" if report.max_psi >= float(self.cfg.psi_warn) else "ok"
        result = {
            "status": status,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "streak": self.streak,
            "needed_streak": int(self.cfg.consecutive_checks),
            "mean_probability": float(np.mean([r["probability"] for r in records]))
            if records
            else None,
            "latest_run_id": records[-1]["run_id"] if records else None,
            **report.as_dict(),
        }
        with self._lock:
            self.latest = result
            self.history.append(
                {"checked_at": result["checked_at"], "status": status, "max_psi": result["max_psi"]}
            )
        return result


def create_app(
    log_dir: Path | None = None,
    profile_path: Path | None = None,
    params: Params | None = None,
    start_loop: bool = True,
) -> FastAPI:
    """Build the monitor application (``start_loop=False`` in tests)."""
    monitor = DriftMonitor(
        log_dir or Path(os.environ.get("INFERENCE_LOG_DIR", "/data/inference-logs")),
        profile_path
        or Path(os.environ.get("REFERENCE_PROFILE", "/data/models/reference_profile.json")),
        params or load_params(Path(os.environ.get("PARAMS_FILE", "/app/params.yaml"))),
    )
    registry = CollectorRegistry()
    psi_gauge = Gauge(
        "churnsense_drift_psi", "PSI of a feature against training", ["feature"], registry=registry
    )
    max_gauge = Gauge("churnsense_drift_max_psi", "Largest feature PSI", registry=registry)
    status_gauge = Gauge(
        "churnsense_drift_status",
        "-1 too little data, 0 ok, 1 warn, 2 alert candidate, 3 alert",
        registry=registry,
    )
    rows_gauge = Gauge(
        "churnsense_drift_window_rows", "Predictions in the checked window", registry=registry
    )
    mean_gauge = Gauge(
        "churnsense_drift_mean_probability", "Mean predicted churn probability", registry=registry
    )
    last_gauge = Gauge(
        "churnsense_drift_last_check_timestamp_seconds", "Time of the last check", registry=registry
    )
    stop = threading.Event()

    def run_check() -> dict[str, Any]:
        result = monitor.check()
        for name, value in result["psi_by_feature"].items():
            psi_gauge.labels(name).set(value)
        max_gauge.set(result["max_psi"])
        status_gauge.set(STATUS_CODE[result["status"]])
        rows_gauge.set(result["n_rows"])
        mean_gauge.set(
            result["mean_probability"] if result["mean_probability"] is not None else np.nan
        )
        last_gauge.set(time.time())
        return result

    def loop() -> None:
        while not stop.is_set():
            try:
                result = run_check()
                log.info(
                    "drift check",
                    extra={
                        "status": result["status"],
                        "max_psi": result["max_psi"],
                        "rows": result["n_rows"],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - keep checking; the error is visible in /health
                log.error("drift check failed", extra={"error": str(exc)})
            stop.wait(float(monitor.cfg.interval_seconds))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        if start_loop:
            threading.Thread(target=loop, daemon=True, name="drift-loop").start()
        yield
        stop.set()

    app = FastAPI(title="ChurnSense drift monitor", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "last_check": monitor.latest["checked_at"]}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return {**monitor.latest, "history": list(monitor.history)[-20:]}

    @app.post("/check")
    def check_now() -> dict[str, Any]:
        return run_check()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    app.state.monitor = monitor
    return app
