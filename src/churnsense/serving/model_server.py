"""Model server: loads the champion from the serving folder and serves predictions.

It loads ``<serving>/current`` (a link to an exported model ``v<N>``) once at start. A promotion
does not restart the container: it calls ``POST /admin/reload``, which loads the new model
completely, checks it answers a sample customer, and only then swaps it in. If anything fails
the old model keeps serving. That makes a promotion (and a rollback) zero-downtime.

Endpoints: ``GET /health``, ``GET /model``, ``POST /invocations``, ``POST /admin/reload``
(needs the admin token), ``GET /metrics`` (Prometheus, internal network only) and ``/mlflow/*``,
MLflow's own scoring API (``/mlflow/invocations``, ``/mlflow/ping``) around the same model.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from mlflow.pyfunc import scoring_server
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from churnsense.config import serving_dir
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.model.pyfunc import model_inputs

log = get_logger(__name__)

MAX_ROWS = 1000
SELF_CHECK_ROW: dict[str, Any] = {
    "customer_id": "self-check",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.0,
    "total_charges": 840.0,
}


@dataclass
class Loaded:
    """A model that is loaded and has passed its self-check."""

    model: Any
    path: str
    version: str
    run_id: str
    family: str
    trained_at: str
    t_high: float
    t_medium: float
    loaded_at: float

    def info(self) -> dict[str, Any]:
        """Facts about the model, for ``/health``, ``/model`` and the prediction answers."""
        return {
            "version": self.version,
            "run_id": self.run_id,
            "family": self.family,
            "trained_at": self.trained_at,
            "thresholds": {"t_high": self.t_high, "t_medium": self.t_medium},
            "loaded_at": self.loaded_at,
        }


def to_frame(instances: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn request rows into the typed frame the model signature expects."""
    frame = pd.DataFrame(instances)
    if "customer_id" not in frame:
        frame["customer_id"] = ""
    if "total_charges" not in frame:
        frame["total_charges"] = np.nan
    frame["customer_id"] = frame["customer_id"].fillna("").astype(str)
    return model_inputs(frame)


def load_model(path: Path) -> Loaded:
    """Load an exported model, prove that it answers a sample customer, and describe it."""
    resolved = path.resolve()
    model = mlflow.pyfunc.load_model(str(resolved))
    answer = model.predict(to_frame([SELF_CHECK_ROW]))
    probability = float(answer["churn_probability"].iloc[0])
    if not 0.0 <= probability <= 1.0 or answer["risk_band"].iloc[0] not in (
        "HIGH",
        "MEDIUM",
        "LOW",
    ):
        raise ValueError(f"the model failed its self-check (probability {probability})")
    bundle = model.unwrap_python_model().bundle
    return Loaded(
        model=model,
        path=str(resolved),
        version=resolved.name,
        run_id=bundle.run_id,
        family=bundle.family,
        trained_at=bundle.trained_at,
        t_high=bundle.thresholds.t_high,
        t_medium=bundle.thresholds.t_medium,
        loaded_at=time.time(),
    )


class MlflowScoring:
    """MLflow's own scoring server, mounted at ``/mlflow`` and rebuilt on every model swap.

    MLflow ships a standard model-serving API (``POST /invocations``, ``GET /ping``, ``/health``,
    ``/version``). Mounting it keeps that interface available next to ours, and it always serves the
    model that is currently loaded (a promotion replaces it together with the main model).
    """

    def __init__(self) -> None:
        self.app: Any = None

    def set_model(self, model: Any) -> None:
        """Build MLflow's scoring app around a freshly loaded model."""
        self.app = scoring_server.init(model)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Hand the request to MLflow's app, or answer 503 while no model is loaded."""
        if self.app is None:
            body = b'{"error": "no model is loaded"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def create_app(serving_root: Path | None = None, admin_token: str | None = None) -> FastAPI:
    """Build the model-server application (a factory, so tests can use their own folders)."""
    root = serving_root or serving_dir()
    token = os.environ.get("ADMIN_TOKEN", "") if admin_token is None else admin_token
    lock = threading.Lock()
    holder: dict[str, Loaded | None] = {"current": None}

    registry = CollectorRegistry()
    requests = Counter(
        "churnsense_model_requests_total", "Prediction requests", ["outcome"], registry=registry
    )
    seconds = Histogram(
        "churnsense_model_request_seconds",
        "Prediction request time",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
        registry=registry,
    )
    predictions = Counter(
        "churnsense_model_predictions_total",
        "Predictions by risk band",
        ["band"],
        registry=registry,
    )
    scores = Histogram(
        "churnsense_model_probability",
        "Predicted churn probability",
        buckets=tuple(i / 20 for i in range(1, 20)),
        registry=registry,
    )
    info_gauge = Gauge(
        "churnsense_model_info",
        "The loaded model (value is always 1)",
        ["version", "run_id", "family"],
        registry=registry,
    )
    loaded_at = Gauge(
        "churnsense_model_loaded_timestamp_seconds", "When the model was loaded", registry=registry
    )
    reloads = Counter(
        "churnsense_model_reloads_total", "Model reloads", ["result"], registry=registry
    )
    last_labels: list[tuple[str, str, str]] = []

    mlflow_scoring = MlflowScoring()

    def swap_in(new: Loaded) -> None:
        mlflow_scoring.set_model(new.model)
        with lock:
            holder["current"] = new
        for labels in last_labels:
            info_gauge.remove(*labels)
        last_labels[:] = [(new.version, new.run_id, new.family)]
        info_gauge.labels(*last_labels[0]).set(1)
        loaded_at.set(new.loaded_at)

    def reload_current() -> Loaded:
        link = root / "current"
        if not link.exists():
            raise FileNotFoundError(f"{link} does not exist: no model has been promoted yet")
        new = load_model(link)
        swap_in(new)
        return new

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        try:
            loaded = reload_current()
            reloads.labels("ok").inc()
            log.info("model loaded", extra={"version": loaded.version, "run_id": loaded.run_id})
        except Exception as exc:  # noqa: BLE001 - start anyway; /health reports the problem
            reloads.labels("failed").inc()
            log.error("no model could be loaded at start", extra={"error": str(exc)})
        yield

    app = FastAPI(title="ChurnSense model server", lifespan=lifespan)
    app.mount("/mlflow", mlflow_scoring)

    @app.get("/health")
    def health() -> JSONResponse:
        current = holder["current"]
        if current is None:
            return JSONResponse({"status": "no_model"}, status_code=503)
        return JSONResponse({"status": "ok", "model": current.info()})

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        current = holder["current"]
        if current is None:
            raise HTTPException(503, "no model is loaded")
        return current.info()

    @app.post("/invocations")
    def invocations(body: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        started = time.perf_counter()
        current = holder["current"]
        if current is None:
            requests.labels("error").inc()
            raise HTTPException(503, "no model is loaded")
        instances = body.get("instances")
        if not isinstance(instances, list) or not 0 < len(instances) <= MAX_ROWS:
            requests.labels("invalid").inc()
            raise HTTPException(422, f"'instances' must be a list of 1 to {MAX_ROWS} customers")
        try:
            result = current.model.predict(to_frame(instances))
        except Exception as exc:  # noqa: BLE001 - schema or type problems are the caller's
            requests.labels("invalid").inc()
            raise HTTPException(422, f"the customer data does not fit the model: {exc}") from exc
        rows = result.to_dict(orient="records")
        for row in rows:
            row["top_factors"] = json.loads(row["top_factors"])
            predictions.labels(row["risk_band"]).inc()
            scores.observe(row["churn_probability"])
        requests.labels("ok").inc()
        seconds.observe(time.perf_counter() - started)
        return {
            "predictions": rows,
            "model": {"version": current.version, "run_id": current.run_id},
        }

    @app.post("/admin/reload")
    def reload(x_admin_token: str | None = Header(default=None)) -> JSONResponse:
        if not token:
            raise HTTPException(503, "admin actions are disabled: no admin token is configured")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, token):
            raise HTTPException(401, "invalid admin token")
        previous = holder["current"]
        try:
            new = reload_current()
        except Exception as exc:  # noqa: BLE001 - the old model keeps serving
            reloads.labels("failed").inc()
            log.error("reload failed, keeping the old model", extra={"error": str(exc)})
            return JSONResponse(
                {
                    "status": "failed",
                    "error": str(exc),
                    "serving": None if previous is None else previous.info(),
                },
                status_code=500,
            )
        reloads.labels("ok").inc()
        return JSONResponse(
            {
                "status": "reloaded",
                "model": new.info(),
                "previous": None if previous is None else previous.info(),
            }
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
