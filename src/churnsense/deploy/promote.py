"""Promote the challenger: register it, export it, make it current, set the aliases.

Run as ``python -m churnsense.deploy.promote`` (``make promote``). Steps, in order, where a
failure at any step leaves the CURRENT model untouched:

1. Read the gate decision; refuse unless it is PROMOTE.
2. Stale guard: the decision must be about THIS candidate, and the champion must still be
   the one the evaluation compared with (somebody else may have promoted in the meantime).
3. Register the model in the MLflow Model Registry (a new version, no alias yet).
4. Export it to the serving folder as ``v<version>`` after a verification in a fresh process.
5. Switch ``current`` to it (atomic).
6. Reload the model server (no restart, no downtime) and run the smoke test through the
   backend. If either fails, the previous model is put back and the promotion fails; the
   aliases are untouched. Skipped when no model server is configured.
7. Only now move the registry aliases: ``champion`` to the new version, ``previous`` to the
   old champion. An alias is a promise that the model is live and healthy, so it comes last.

Exit codes: 0 promoted, 3 refused by the gate or a guard, 1 any failure.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import mlflow
from mlflow import MlflowClient

from churnsense import mlflow_utils as mu
from churnsense.config import Paths, default_paths, serving_dir
from churnsense.deploy import export, smoke
from churnsense.logging_utils import get_logger, setup_logging
from churnsense.tracking import write_json

log = get_logger(__name__)


class PromotionRefusedError(RuntimeError):
    """The gate said no, or a safety guard failed. Nothing was changed."""


def read_inputs(paths: Paths) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the gate decision and the run information of the candidate."""
    for needed in (
        paths.report("gate_decision.json"),
        paths.run_info,
        paths.report("smoke_rows.json"),
    ):
        if not needed.is_file():
            raise PromotionRefusedError(f"{needed.name} is missing: run the pipeline first")
    decision = json.loads(paths.report("gate_decision.json").read_text("utf-8"))
    run_info = json.loads(paths.run_info.read_text("utf-8"))
    return decision, run_info


def guards(decision: dict[str, Any], run_info: dict[str, Any], serving_root: Any) -> None:
    """Refuse unless the decision is PROMOTE and still valid for the current situation."""
    if decision.get("decision") != "PROMOTE":
        raise PromotionRefusedError(
            "the gate decision is REJECT: " + "; ".join(decision["reasons"])
        )
    if decision.get("candidate_run_id") != run_info["run_id"]:
        raise PromotionRefusedError(
            "the gate decision is about a different model than the candidate"
        )
    version = export.current_version(serving_root)
    champion_now = export.read_meta(serving_root / f"v{version}").get("run_id") if version else None
    if champion_now != decision.get("champion_run_id"):
        raise PromotionRefusedError(
            "the champion changed since the evaluation: run evaluate and gate again"
        )


def live_checks(
    paths: Paths, serving_root: Any, run_info: dict[str, Any], version: int, previous: int | None
) -> str:
    """Reload the model server and smoke-test; on any failure put the old model back.

    Runs only when ``MODEL_SERVER_URL`` is set (the compose file sets it for the trainer);
    otherwise it is reported as skipped (for example in unit tests or without the services).
    For the very first model, a model server that is not running yet is not an error.
    """
    url = os.environ.get("MODEL_SERVER_URL")
    if not url:
        return "skipped: MODEL_SERVER_URL is not set"
    token = os.environ.get("ADMIN_TOKEN", "")
    backend = os.environ.get("BACKEND_URL", "http://backend:8000")
    try:
        smoke.reload_model_server(url, token, run_info["run_id"])
        result = smoke.smoke_test(backend, paths.report("smoke_rows.json"), run_info["run_id"])
        return f"ok: reloaded and {result['rows']} smoke rows match"
    except smoke.DeploymentError as exc:
        if previous is None and isinstance(exc, smoke.ServerNotReachableError):
            # The very first deployment on a new machine: the model server cannot be running yet,
            # because it needs a model to start. Nothing is serving, so nothing can be harmed;
            # the server loads this model when it starts, and `make check` verifies it then.
            return f"skipped: the model server is not running yet ({exc})"
        if previous is not None:
            export.switch_current(serving_root, previous)
            rollback = f"rolled back to v{previous}"
            try:
                smoke.reload_model_server(url, token, _run_of(serving_root, previous))
            except smoke.DeploymentError as again:
                rollback += f" (but reloading it failed: {again})"
        else:
            export.clear_current(serving_root)
            rollback = "no earlier model to go back to; current was removed"
        raise smoke.DeploymentError(f"{exc}; {rollback}") from exc


def _run_of(serving_root: Any, version: int) -> str:
    return export.read_meta(serving_root / f"v{version}").get("run_id", "")


def promote(paths: Paths, *, set_aliases: bool = True) -> dict[str, Any]:
    """Run the promotion steps and return a summary."""
    serving_root = serving_dir()
    decision, run_info = read_inputs(paths)
    guards(decision, run_info, serving_root)

    mlflow.set_tracking_uri(mu.tracking_uri())
    mu.check_tracking_server(mu.tracking_uri())
    client = MlflowClient()
    summary: dict[str, Any] = {"candidate_run_id": run_info["run_id"], "steps": {}}

    registered = mu.retry()(mlflow.register_model)(
        model_uri=run_info["model_uri"], name=mu.REGISTERED_MODEL
    )
    version = int(registered.version)
    for key, value in {
        "run_id": run_info["run_id"],
        "git_commit": run_info["git_commit"],
        "family": run_info["family"],
        "gate_decision": decision["decision"],
    }.items():
        client.set_model_version_tag(mu.REGISTERED_MODEL, str(version), key, str(value))
    summary["version"] = version
    summary["steps"]["registered"] = "ok"

    previous = export.current_version(serving_root)
    meta = {
        "run_id": run_info["run_id"],
        "model_uri": run_info["model_uri"],
        "family": run_info["family"],
        "git_commit": run_info["git_commit"],
        "t_high": run_info["t_high"],
        "t_medium": run_info["t_medium"],
        "registry_version": version,
    }
    export.export_model(
        paths.root, paths.candidate, serving_root, version, paths.report("smoke_rows.json"), meta
    )
    summary["steps"]["exported"] = "ok"

    export.switch_current(serving_root, version)
    summary["steps"]["switched"] = f"v{previous} -> v{version}" if previous else f"-> v{version}"

    summary["steps"]["live_checks"] = live_checks(paths, serving_root, run_info, version, previous)

    if set_aliases:
        if previous is not None:
            client.set_registered_model_alias(mu.REGISTERED_MODEL, "previous", str(previous))
        client.set_registered_model_alias(mu.REGISTERED_MODEL, "champion", str(version))
        old = f", previous -> {previous}" if previous else ""
        summary["steps"]["aliases"] = f"champion -> {version}{old}"
    summary["pruned_versions"] = export.prune(serving_root)
    return summary


def main() -> None:
    """Command-line entry point."""
    setup_logging()
    paths = default_paths()
    try:
        summary = promote(paths)
    except PromotionRefusedError as exc:
        log.error("promotion refused", extra={"reason": str(exc)})
        write_json(paths.report("promotion.json"), {"status": "refused", "reason": str(exc)})
        print(f"REFUSED: {exc}")
        sys.exit(3)
    except Exception as exc:  # noqa: BLE001 - report every failure, current model is untouched
        log.exception("promotion failed")
        write_json(
            paths.report("promotion.json"),
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        print(f"FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    write_json(paths.report("promotion.json"), {"status": "promoted", **summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
