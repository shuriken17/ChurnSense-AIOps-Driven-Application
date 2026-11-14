"""Live checks after a model switch: reload the model server, then smoke-test the whole path.

The smoke test sends the stored smoke rows (real inputs with the answers the new model gave
during evaluation) through the BACKEND, so it proves the full chain (backend, model server,
the loaded model) and not just the files on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

PROBABILITY_TOLERANCE = 1e-6


class DeploymentError(RuntimeError):
    """The live system did not accept the new model."""


class ServerNotReachableError(DeploymentError):
    """The model server cannot be reached at all (not started, or its name does not resolve)."""


def reload_model_server(url: str, token: str, expected_run_id: str, timeout: float = 90.0) -> dict:
    """Ask the model server to load ``current``; check it now serves the expected run."""
    try:
        answer = httpx.post(
            f"{url.rstrip('/')}/admin/reload", headers={"x-admin-token": token}, timeout=timeout
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise ServerNotReachableError(f"the model server is not reachable: {exc}") from exc
    except httpx.HTTPError as exc:
        raise DeploymentError(f"the model server did not answer properly: {exc}") from exc
    body = (
        answer.json()
        if answer.headers.get("content-type", "").startswith("application/json")
        else {}
    )
    if answer.status_code != 200:
        raise DeploymentError(
            f"the model server refused the new model: {body.get('error') or body}"
        )
    served = body["model"]["run_id"]
    if served != expected_run_id:
        raise DeploymentError(
            f"the model server loaded run {served[:8]}, expected {expected_run_id[:8]}"
        )
    return body["model"]


def smoke_test(backend_url: str, smoke_file: Path, expected_run_id: str) -> dict[str, Any]:
    """Send every smoke row through the backend and compare with the stored answers."""
    rows = json.loads(smoke_file.read_text("utf-8"))["rows"]
    problems: list[str] = []
    with httpx.Client(base_url=backend_url.rstrip("/"), timeout=20.0) as client:
        for index, row in enumerate(rows):
            try:
                answer = client.post("/api/v1/predict", json=row["input"])
            except httpx.HTTPError as exc:
                raise DeploymentError(f"the backend is not reachable: {exc}") from exc
            if answer.status_code != 200:
                problems.append(f"row {index}: HTTP {answer.status_code}")
                continue
            body = answer.json()
            gap = abs(body["churn_probability"] - row["expected"]["churn_probability"])
            if gap > PROBABILITY_TOLERANCE:
                problems.append(f"row {index}: probability differs by {gap:.2e}")
            if body["risk_band"] != row["expected"]["risk_band"]:
                problems.append(
                    f"row {index}: band {body['risk_band']} instead of "
                    f"{row['expected']['risk_band']}"
                )
            if body["model"]["run_id"] != expected_run_id:
                problems.append(f"row {index}: answered by another model")
    if problems:
        raise DeploymentError("smoke test failed: " + "; ".join(problems[:5]))
    return {"rows": len(rows), "all_match": True}


def main() -> None:
    """Smoke-test the RUNNING system with the stored smoke rows (``make smoke``)."""
    from churnsense.config import default_paths

    smoke_file = default_paths().report("smoke_rows.json")
    served = json.loads(smoke_file.read_text("utf-8"))["model_run_id"]
    backend = os.environ.get("BACKEND_URL", "http://backend:8000")
    print(json.dumps(smoke_test(backend, smoke_file, served)))


if __name__ == "__main__":
    main()
