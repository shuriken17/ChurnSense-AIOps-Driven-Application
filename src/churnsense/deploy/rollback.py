"""Roll back to the previous model version (``make rollback``).

Steps: pick the newest exported version older than ``current``, switch ``current`` to it, reload the
model server, check the backend still answers, and move the registry aliases (``champion`` to the
restored version, ``rolled_back`` to the one that was removed). If the reload fails, ``current`` is
put back where it was and the command fails: the system is never left half-switched.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

from churnsense import mlflow_utils as mu
from churnsense.config import default_paths, serving_dir
from churnsense.deploy import export, smoke
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.tracking import write_json

log = get_logger(__name__)


class RollbackError(RuntimeError):
    """There is nothing to roll back to, or the rollback did not work."""


def rollback(serving_root: Any, *, live: bool = True, aliases: bool = True) -> dict[str, Any]:
    """Switch to the previous version and verify it."""
    current = export.current_version(serving_root)
    if current is None:
        raise RollbackError("there is no current model")
    older = [v for v in export.list_versions(serving_root) if v < current]
    if not older:
        raise RollbackError(f"v{current} is the oldest exported version: nothing to roll back to")
    target = older[-1]
    export.switch_current(serving_root, target)
    summary: dict[str, Any] = {"from": current, "to": target}
    url = os.environ.get("MODEL_SERVER_URL")
    if live and url:
        run_id = export.read_meta(serving_root / f"v{target}").get("run_id", "")
        try:
            smoke.reload_model_server(url, os.environ.get("ADMIN_TOKEN", ""), run_id)
            backend = os.environ.get("BACKEND_URL", "http://backend:8000")
            if httpx.get(f"{backend}/api/v1/health", timeout=10).status_code != 200:
                raise smoke.DeploymentError("the backend reports the system as degraded")
        except (smoke.DeploymentError, httpx.HTTPError) as exc:
            export.switch_current(serving_root, current)
            raise RollbackError(
                f"the rollback to v{target} failed ({exc}); v{current} is current again"
            ) from exc
        summary["live"] = "reloaded and healthy"
    else:
        summary["live"] = "skipped: no model server configured"
    if aliases:
        # The rollback itself is already done; moving the registry aliases is bookkeeping and must
        # never fail the rollback (MLflow may be the very thing that is broken).
        try:
            from mlflow import MlflowClient

            mlflow_client = MlflowClient(tracking_uri=mu.tracking_uri())
            mlflow_client.set_registered_model_alias(mu.REGISTERED_MODEL, "champion", str(target))
            mlflow_client.set_registered_model_alias(
                mu.REGISTERED_MODEL, "rolled_back", str(current)
            )
            summary["aliases"] = f"champion -> {target}, rolled_back -> {current}"
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            summary["aliases"] = (
                f"skipped: MLflow did not accept the alias change ({type(exc).__name__})"
            )
    return summary


def main() -> None:
    """Command-line entry point (``make rollback``)."""
    setup_logging()
    paths = default_paths()
    try:
        summary = rollback(serving_dir())
    except Exception as exc:  # noqa: BLE001 - report everything with a clear exit code
        write_json(paths.report("rollback.json"), {"status": "failed", "error": str(exc)})
        print(f"FAILED: {exc}")
        sys.exit(1)
    write_json(paths.report("rollback.json"), {"status": "rolled_back", **summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
