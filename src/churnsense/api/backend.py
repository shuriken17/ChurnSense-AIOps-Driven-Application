"""The backend API (version 1): the only service the web page talks to.

Endpoints (all under ``/api/v1``):

* ``GET  /health``        backend and model-server status
* ``GET  /fields``        the 20 input fields, for building the form
* ``GET  /model``         facts about the model that is serving
* ``POST /predict``       one customer in; probability, risk band and top factors out
* ``POST /predict/batch`` up to 1000 customers; each row scored or rejected with its reason
* ``POST /feedback``      the real outcome of a scored customer (churned or stayed)
* ``GET  /performance``   accuracy, precision and recall measured on that feedback
* ``GET  /stats``         prediction counts and band shares of the last 24 hours
* ``GET  /pipeline``      gate decision, training and evaluation summary, controller
* ``GET  /drift``         the drift monitor's latest verdict (PSI per feature, history)
* ``GET  /components``    health of every component (model server, MLflow, nginx, Grafana, ...)
* ``GET  /pipeline/stages`` the DVC stages with status, speed and rows per second, and the graph
* ``GET  /pipeline/runs`` the controller's run history (with failed steps) and waiting requests
* ``POST /admin/pipeline/request`` ask the controller to retrain or roll back (admin token)
* ``POST /admin/reload-model``  ask the model server to load ``current`` (admin token)
* ``GET  /metrics``       Prometheus metrics (internal network only; nginx does not expose it)

Errors use one shape: ``{"error": {"code": ..., "message": ..., "details": [...]}}``.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

from churnsense.api.components import ComponentProbe, targets_from_environment
from churnsense.api.validation import validate_customer
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.schema import field_catalogue
from churnsense.store import LogStore, UnknownPredictionError, read_recent

log = get_logger(__name__)
API = "/api/v1"
MAX_BATCH = 1000
STAGE_REPORTS = (
    ("ingest", "ingest_stats.json"),
    ("validate", "validation_report.json"),
    ("eda", "eda_summary.json"),
    ("split", "split_report.json"),
    ("synthesize", "synthetic_stats.json"),
    ("featurize", "featurize_stats.json"),
    ("train", "train_metrics.json"),
    ("evaluate", "eval_metrics.json"),
    ("gate", "gate_decision.json"),
)
#: The DVC graph (checked against dvc.yaml by a test): who feeds whom.
PIPELINE_EDGES = (
    ("ingest", "validate"),
    ("validate", "eda"),
    ("validate", "split"),
    ("split", "synthesize"),
    ("split", "featurize"),
    ("synthesize", "featurize"),
    ("featurize", "train"),
    ("split", "train"),
    ("train", "evaluate"),
    ("split", "evaluate"),
    ("evaluate", "gate"),
    ("train", "gate"),
)
REQUEST_ACTIONS = ("retrain", "rollback")


def error_body(code: str, message: str, details: list[Any] | None = None) -> dict[str, Any]:
    """The single error format of the API."""
    return {"error": {"code": code, "message": message, "details": details or []}}


def create_app(
    model_client: httpx.Client | None = None,
    admin_token: str | None = None,
    store: LogStore | None = None,
    reports_dir: Path | None = None,
    state_dir: Path | None = None,
    drift_client: httpx.Client | None = None,
    components: ComponentProbe | None = None,
) -> FastAPI:
    """Build the backend application (``model_client``, ``store`` are replaced in tests)."""
    token = os.environ.get("ADMIN_TOKEN", "") if admin_token is None else admin_token
    client = model_client or httpx.Client(
        base_url=os.environ.get("MODEL_SERVER_URL", "http://model-server:8001"),
        timeout=httpx.Timeout(10.0, connect=2.0),
    )
    store = store or LogStore(
        os.environ.get("INFERENCE_LOG_DIR", "/data/inference-logs"),
        os.environ.get("ID_HASH_SALT", "dev-salt"),
    )
    reports = reports_dir or Path(os.environ.get("REPORTS_DIR", "/data/reports"))
    state = state_dir or Path(os.environ.get("STATE_DIR", "/data/state"))
    drift = drift_client or httpx.Client(
        base_url=os.environ.get("DRIFT_URL", "http://drift-monitor:8002"),
        timeout=httpx.Timeout(5.0, connect=2.0),
    )
    probe = components or ComponentProbe(targets_from_environment(), state)
    registry = CollectorRegistry()
    http_total = Counter(
        "churnsense_api_requests_total",
        "API requests",
        ["method", "path", "status"],
        registry=registry,
    )
    http_seconds = Histogram(
        "churnsense_api_request_seconds",
        "API request time",
        ["path"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
        registry=registry,
    )
    invalid = Counter(
        "churnsense_api_validation_errors_total", "Rejected customer records", registry=registry
    )
    upstream = Counter(
        "churnsense_api_model_server_errors_total", "Model server failures", registry=registry
    )

    log_errors = Counter(
        "churnsense_api_log_errors_total", "Inference log write failures", registry=registry
    )
    feedback_total = Counter(
        "churnsense_feedback_total", "Feedback received", ["outcome"], registry=registry
    )
    for name, key, text in (
        ("accuracy", "accuracy", "Accuracy on feedback"),
        ("precision", "precision", "HIGH-band precision on feedback"),
        ("recall", "recall", "HIGH-band recall on feedback"),
    ):
        Gauge(f"churnsense_live_{name}", text, registry=registry).set_function(
            lambda key=key: (
                store.performance()[key] if store.performance()[key] is not None else math.nan
            )
        )
    Gauge(
        "churnsense_live_feedback_rows",
        "Labelled outcomes in the performance window",
        registry=registry,
    ).set_function(lambda: store.performance()["n"])

    dependency = Gauge(
        "churnsense_dependency_up",
        "1 if the component answers its health URL, else 0",
        ["name"],
        registry=registry,
    )
    for component in probe.targets:
        dependency.labels(component).set_function(
            lambda name=component: probe.results.get(name, math.nan)
        )
    Gauge(
        "churnsense_controller_heartbeat_age_seconds",
        "Seconds since the retrain controller last wrote its status",
        registry=registry,
    ).set_function(probe.heartbeat_age)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        probe.start()
        yield
        probe.stop()

    app = FastAPI(
        title="ChurnSense API",
        version="1.0",
        lifespan=lifespan,
        docs_url=f"{API}/docs",
        openapi_url=f"{API}/openapi.json",
    )
    setup_logging()

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        path = request.url.path if request.url.path.startswith(API) else "other"
        http_total.labels(request.method, path, str(response.status_code)).inc()
        http_seconds.labels(path).observe(elapsed)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed * 1000, 1),
            },
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            error_body(f"http_{exc.status_code}", str(exc.detail)), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def body_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body("invalid_request", "the request is not valid"), status_code=422
        )

    def ask_model_server(method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            upstream.inc()
            log.error("model server unreachable", extra={"error": str(exc)})
            raise StarletteHTTPException(503, "the prediction service is not reachable") from exc

    def record_prediction(
        request_id: str, clean: dict[str, Any], row: dict[str, Any], model_info: dict[str, Any]
    ) -> None:
        """Append to the inference log; a log failure must never fail the prediction itself."""
        try:
            store.log_prediction(
                {
                    "request_id": request_id,
                    "cid": store.hash_id(clean.get("customer_id")),
                    "inputs": {k: v for k, v in clean.items() if k != "customer_id"},
                    "probability": row["churn_probability"],
                    "band": row["risk_band"],
                    "run_id": model_info["run_id"],
                    "version": model_info["version"],
                }
            )
        except OSError as exc:
            log_errors.inc()
            log.error("could not write the inference log", extra={"error": str(exc)})

    @app.get(f"{API}/health")
    def health() -> JSONResponse:
        try:
            answer = client.get("/health")
            model = answer.json().get("model") if answer.status_code == 200 else None
            ok = answer.status_code == 200
        except httpx.HTTPError:
            ok, model = False, None
        body = {
            "status": "ok" if ok else "degraded",
            "backend": "ok",
            "model_server": "ok" if ok else "down",
            "model": model,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get(f"{API}/fields")
    def fields() -> list[dict[str, Any]]:
        return field_catalogue()

    @app.get(f"{API}/model")
    def model() -> Any:
        answer = ask_model_server("GET", "/model")
        if answer.status_code != 200:
            raise StarletteHTTPException(503, "no model is loaded")
        return answer.json()

    @app.post(f"{API}/predict")
    def predict(request: Request, customer: dict[str, Any] = Body(...)) -> Any:  # noqa: B008
        clean, problems = validate_customer(customer)
        if problems:
            invalid.inc()
            return JSONResponse(
                error_body("invalid_customer", "the customer data has problems", problems),
                status_code=422,
            )
        answer = ask_model_server("POST", "/invocations", json={"instances": [clean]})
        if answer.status_code == 503:
            raise StarletteHTTPException(503, "no model is loaded")
        if answer.status_code != 200:
            upstream.inc()
            raise StarletteHTTPException(502, "the prediction service failed")
        body = answer.json()
        row = body["predictions"][0]
        record_prediction(request.state.request_id, clean, row, body["model"])
        return {
            "customer_id": clean.get("customer_id"),
            "churn_probability": row["churn_probability"],
            "risk_band": row["risk_band"],
            "top_factors": row["top_factors"],
            "model": body["model"],
            "request_id": request.state.request_id,
        }

    @app.post(f"{API}/predict/batch")
    def predict_batch(request: Request, body: dict[str, Any] = Body(...)) -> Any:  # noqa: B008
        customers = body.get("customers")
        if not isinstance(customers, list) or not 0 < len(customers) <= MAX_BATCH:
            return JSONResponse(
                error_body(
                    "invalid_request", f"'customers' must be a list of 1 to {MAX_BATCH} customers"
                ),
                status_code=422,
            )
        results: list[dict[str, Any]] = [{} for _ in customers]
        valid: list[tuple[int, dict[str, Any]]] = []
        for index, customer in enumerate(customers):
            clean, problems = validate_customer(customer)
            if problems:
                invalid.inc()
                results[index] = {
                    "index": index,
                    "error": {"code": "invalid_customer", "details": problems},
                }
            else:
                valid.append((index, clean))
        model_info: dict[str, Any] | None = None
        if valid:
            answer = ask_model_server(
                "POST", "/invocations", json={"instances": [clean for _, clean in valid]}
            )
            if answer.status_code == 503:
                raise StarletteHTTPException(503, "no model is loaded")
            if answer.status_code != 200:
                upstream.inc()
                raise StarletteHTTPException(502, "the prediction service failed")
            scored = answer.json()
            model_info = scored["model"]
            for (index, clean), row in zip(valid, scored["predictions"], strict=True):
                record_prediction(f"{request.state.request_id}-{index}", clean, row, model_info)
                results[index] = {
                    "index": index,
                    "customer_id": clean.get("customer_id"),
                    "churn_probability": row["churn_probability"],
                    "risk_band": row["risk_band"],
                    "top_factors": row["top_factors"],
                }
        bands = {
            band: sum(1 for r in results if r.get("risk_band") == band)
            for band in ("HIGH", "MEDIUM", "LOW")
        }
        return {
            "results": results,
            "model": model_info,
            "summary": {
                "total": len(customers),
                "scored": len(valid),
                "rejected": len(customers) - len(valid),
                "bands": bands,
            },
        }

    @app.post(f"{API}/feedback")
    def feedback(body: dict[str, Any] = Body(...)) -> Any:  # noqa: B008
        problems = []
        churned = body.get("churned")
        if not isinstance(churned, bool) and churned not in (0, 1):
            problems.append({"field": "churned", "message": "must be true or false"})
        ids = {k: body.get(k) for k in ("customer_id", "request_id") if body.get(k)}
        if len(ids) != 1 or not all(isinstance(v, str) for v in ids.values()):
            problems.append(
                {
                    "field": "customer_id / request_id",
                    "message": "give exactly one of them, as text",
                }
            )
        source = body.get("source", "user")
        if source not in ("user", "simulator"):
            problems.append({"field": "source", "message": "must be 'user' or 'simulator'"})
        if problems:
            return JSONResponse(
                error_body("invalid_feedback", "the feedback has problems", problems),
                status_code=422,
            )
        try:
            record = store.log_feedback(churned=bool(churned), source=source, **ids)
        except UnknownPredictionError:
            return JSONResponse(
                error_body("unknown_prediction", "no earlier prediction matches that id"),
                status_code=404,
            )
        feedback_total.labels("churned" if record["churned"] else "stayed").inc()
        outcome = (
            "true_positive"
            if record["band"] == "HIGH" and record["churned"]
            else "false_positive"
            if record["band"] == "HIGH"
            else "false_negative"
            if record["churned"]
            else "true_negative"
        )
        return {
            "status": "recorded",
            "counted_as": outcome,
            "matched": {
                "request_id": record["request_id"],
                "risk_band": record["band"],
                "churn_probability": record["probability"],
                "model": {"version": record["version"], "run_id": record["run_id"]},
            },
        }

    @app.get(f"{API}/performance")
    def performance() -> dict[str, Any]:
        return store.performance()

    @app.get(f"{API}/stats")
    def stats() -> dict[str, Any]:
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        recent = [
            r for r in read_recent(store.directory, "predictions", 100_000) if r["ts"] >= cutoff
        ]
        bands = {
            band: sum(1 for r in recent if r["band"] == band) for band in ("HIGH", "MEDIUM", "LOW")
        }
        return {
            "window_hours": 24,
            "predictions": len(recent),
            "bands": bands,
            "mean_probability": (
                sum(r["probability"] for r in recent) / len(recent) if recent else None
            ),
            "last_prediction_at": recent[-1]["ts"] if recent else None,
        }

    @app.get(f"{API}/drift")
    def drift_status() -> Any:
        try:
            answer = drift.get("/status")
            if answer.status_code == 200:
                return answer.json()
        except (httpx.HTTPError, ValueError):
            pass
        raise StarletteHTTPException(503, "the drift monitor is not reachable")

    @app.get(f"{API}/components")
    def components_view() -> dict[str, Any]:
        results = probe.results or probe.probe_once()  # an answer even before the first round
        age = probe.heartbeat_age()
        return {
            "components": {name: {"up": bool(value)} for name, value in results.items()},
            "controller_heartbeat_age_seconds": None if math.isnan(age) else round(age, 1),
        }

    def read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text("utf-8")) if path.is_file() else None
        except (OSError, ValueError):
            return None

    @app.get(f"{API}/pipeline/stages")
    def pipeline_stages() -> dict[str, Any]:
        stages = []
        for stage, filename in STAGE_REPORTS:
            report = read_json(reports / filename)
            if report is None:
                stages.append({"stage": stage, "status": "not run"})
                continue
            finished = datetime.fromtimestamp((reports / filename).stat().st_mtime, UTC)
            detail = {"gate": report.get("decision"), "train": report.get("winner")}.get(stage)
            stages.append(
                {
                    "stage": stage,
                    "status": report.get("status", "unknown"),
                    "detail": detail,
                    "duration_seconds": report.get("duration_seconds"),
                    "rows_in": report.get("rows_in"),
                    "rows_out": report.get("rows_out"),
                    "rows_per_second": report.get("rows_per_second"),
                    "error": report.get("error"),
                    "finished_at": finished.isoformat(timespec="seconds"),
                }
            )
        return {"stages": stages, "edges": [list(edge) for edge in PIPELINE_EDGES]}

    @app.get(f"{API}/pipeline/runs")
    def pipeline_runs(limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        runs: list[dict[str, Any]] = []
        run_dir = state / "runs"
        for file in sorted(run_dir.glob("*.json"), reverse=True) if run_dir.is_dir() else []:
            record = read_json(file)
            if record:  # an unreadable file is skipped and does not use up a slot
                runs.append(record)
            if len(runs) >= limit:
                break
        pending = (
            sorted((state / "requests").glob("*.json")) if (state / "requests").is_dir() else []
        )
        return {
            "runs": runs,
            "pending_requests": [(read_json(f) or {}).get("action", "unknown") for f in pending],
        }

    @app.post(f"{API}/admin/pipeline/request")
    def pipeline_request(
        body: dict[str, Any] = Body(...),  # noqa: B008
        x_admin_token: str | None = Header(default=None),  # noqa: B008
    ) -> Any:
        if not token:
            raise StarletteHTTPException(503, "admin actions are disabled")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, token):
            raise StarletteHTTPException(401, "invalid admin token")
        action = body.get("action")
        if action not in REQUEST_ACTIONS:
            return JSONResponse(
                error_body(
                    "invalid_request", "action must be one of: " + ", ".join(REQUEST_ACTIONS)
                ),
                status_code=422,
            )
        queue = state / "requests"
        queue.mkdir(parents=True, exist_ok=True)
        if any((read_json(f) or {}).get("action") == action for f in queue.glob("*.json")):
            return JSONResponse(
                error_body("already_requested", f"a {action} request is already waiting"),
                status_code=409,
            )
        request_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        temporary = queue / f".{request_id}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "action": action,
                    "requested_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "requested_by": "web console",
                }
            ),
            encoding="utf-8",
        )
        os.replace(temporary, queue / f"{request_id}.json")
        return JSONResponse(
            {
                "status": "accepted",
                "request_id": request_id,
                "action": action,
                "message": "The controller picks this up within a few seconds.",
            },
            status_code=202,
        )

    @app.get(f"{API}/pipeline")
    def pipeline() -> dict[str, Any]:
        def read(name: str) -> dict[str, Any]:
            path = reports / name
            return json.loads(path.read_text("utf-8")) if path.is_file() else {}

        def controller_view() -> dict[str, Any]:
            def read_state(path: Path) -> dict[str, Any]:
                try:
                    return json.loads(path.read_text("utf-8")) if path.is_file() else {}
                except (OSError, ValueError):
                    return {}

            runs = sorted((state / "runs").glob("*.json")) if (state / "runs").is_dir() else []
            last = read_state(runs[-1]) if runs else {}
            keys = ("run_id", "trigger", "outcome", "reason", "finished_at", "duration_seconds")
            return {
                "watching": read_state(state / "controller_status.json"),
                "runs": read_state(state / "controller_state.json").get("runs"),
                "last_run": {k: last.get(k) for k in keys} if last else None,
            }

        gate, train, evaluation, promotion = (
            read(n)
            for n in (
                "gate_decision.json",
                "train_metrics.json",
                "eval_metrics.json",
                "promotion.json",
            )
        )
        golden = (evaluation.get("challenger") or {}).get("golden") or {}
        return {
            "gate": {"decision": gate.get("decision"), "reasons": gate.get("reasons", [])},
            "training": {
                "winner": train.get("winner"),
                "trained_run": train.get("run_id"),
                "validation_by_family": train.get("validation_by_family"),
            },
            "golden": {
                k: golden.get(k) for k in ("n", "roc_auc", "pr_auc", "recall_at_min_precision")
            },
            "last_promotion": promotion.get("status"),
            "controller": controller_view(),
        }

    @app.post(f"{API}/admin/reload-model")
    def reload_model(x_admin_token: str | None = Header(default=None)) -> Any:
        if not token:
            raise StarletteHTTPException(503, "admin actions are disabled")
        if x_admin_token != token:
            raise StarletteHTTPException(401, "invalid admin token")
        answer = ask_model_server("POST", "/admin/reload", headers={"x-admin-token": token})
        return JSONResponse(answer.json(), status_code=answer.status_code)

    @app.get(f"{API}/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
