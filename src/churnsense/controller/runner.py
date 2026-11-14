"""One retraining run, step by step, with every step recorded.

Steps: batch the new feedback, let DVC track it, commit the data pointer (training refuses
uncommitted code or data pointers), run the DVC pipeline (train, evaluate, gate), promote if the
gate says PROMOTE (the promote step reloads the model server, smoke-tests, rolls back on
failure), push to the DVC remote, and commit the results. A failure at any step stops the
run, is recorded with the step name and the end of the error text, and leaves the serving model
untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from churnsense.config import Params, Paths
from churnsense.feedback import batcher
from churnsense.logging_utils import get_logger

log = get_logger(__name__)
OUTCOME_CODE = {"failed": 0, "rejected": 1, "promoted": 2, "skipped": 3, "rolled_back": 4}
GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "churnsense-controller",
    "GIT_AUTHOR_EMAIL": "controller@churnsense.local",
    "GIT_COMMITTER_NAME": "churnsense-controller",
    "GIT_COMMITTER_EMAIL": "controller@churnsense.local",
}

Command = Callable[[list[str], dict[str, str], float], tuple[int, str]]


def run_command(
    cmd: list[str], env: dict[str, str], timeout: float, cwd: Path | None = None
) -> tuple[int, str]:
    """Run a command; return its exit code and the last lines of its output."""
    try:
        done = subprocess.run(
            cmd,
            cwd=cwd,
            env={**os.environ, **env},
            capture_output=True,
            text=True,  # noqa: S603
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f} s"
    tail = (done.stdout + "\n" + done.stderr).strip().splitlines()[-6:]
    return done.returncode, " | ".join(line.strip() for line in tail)[-600:]


class RunRecord:
    """The record of one run, written to ``state/runs/<time>.json``."""

    def __init__(self, trigger: str, reason: str, fingerprint: str | None) -> None:
        self.started = time.time()
        self.data: dict[str, Any] = {
            "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "trigger": trigger,
            "reason": reason,
            "fingerprint": fingerprint,
            "steps": [],
            "outcome": None,
        }

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        """Add one finished step."""
        self.data["steps"].append({"step": name, "ok": ok, "detail": detail[:600]})
        log.info("controller step", extra={"step": name, "ok": ok, "detail": detail[:200]})

    def finish(self, outcome: str, note: str = "") -> dict[str, Any]:
        """Close the record with the outcome."""
        self.data.update(
            outcome=outcome,
            note=note,
            duration_seconds=round(time.time() - self.started, 1),
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return self.data


def retrain(
    paths: Paths,
    cfg: Params,
    trigger: str,
    reason: str,
    fingerprint: str | None,
    log_dir: Path,
    state_dir: Path,
    command: Command | None = None,
) -> dict[str, Any]:
    """Run one full retraining. ``command`` is replaced by a fake in tests."""
    run = command or (lambda cmd, env, timeout: run_command(cmd, env, timeout, paths.root))
    record = RunRecord(trigger, reason, fingerprint)
    timeout = float(cfg.controller.retrain_timeout_minutes) * 60
    env = {
        **GIT_IDENTITY,
        "PIPELINE_TRIGGER": trigger,
        "CONTROLLER_RUN_ID": record.data["run_id"],
        "DRIFT_ALERT_FINGERPRINT": fingerprint or "",
    }

    def fail(step: str, detail: str) -> dict[str, Any]:
        record.step(step, False, detail)
        return record.finish("failed", f"{step} failed")

    try:
        entry = batcher.build_batch(paths, log_dir, state_dir, scenario=trigger)
    except Exception as exc:  # noqa: BLE001 - recorded, the serving model is untouched
        return fail("batch", f"{type(exc).__name__}: {exc}")
    record.step(
        "batch", True, f"{entry['rows']} rows in {entry['file']}" if entry else "no new feedback"
    )
    if entry:
        for name, cmd in (
            ("dvc_add", ["dvc", "add", "data/feedback"]),
            ("git_add", ["git", "add", "data/feedback.dvc", "data/.gitignore"]),
            (
                "git_commit",
                [
                    "git",
                    "commit",
                    "-q",
                    "-m",
                    f"controller: feedback batch {entry['file']} ({trigger})",
                ],
            ),
        ):
            code, tail = run(cmd, env, timeout)
            if code != 0:
                return fail(name, tail)
            record.step(name, True)
    code, tail = run(["dvc", "repro"], env, timeout)
    if code != 0:
        return fail("dvc_repro", tail)
    record.step("dvc_repro", True)

    try:
        gate = json.loads(paths.report("gate_decision.json").read_text("utf-8"))
    except (OSError, ValueError) as exc:
        return fail("read_gate", str(exc))
    record.step("gate", True, f"{gate.get('decision')}: {'; '.join(gate.get('reasons', []))[:300]}")
    outcome = "rejected"
    if gate.get("decision") == "PROMOTE":
        code, tail = run(["python", "-m", "churnsense.deploy.promote"], env, timeout)
        if code != 0:
            return fail("promote", tail)
        record.step("promote", True)
        outcome = "promoted"
    for name, cmd in (
        ("dvc_push", ["dvc", "push"]),
        ("git_add_results", ["git", "add", "dvc.lock", "reports", "models/run_info.json"]),
        ("git_commit_results", ["git", "commit", "-q", "-m", f"controller: {outcome} ({trigger})"]),
    ):
        code, tail = run(cmd, env, timeout)
        if code != 0 and name != "git_commit_results":  # nothing to commit is fine
            return fail(name, tail)
        record.step(name, code == 0 or name == "git_commit_results", tail if code else "")
    return record.finish(outcome, gate.get("decision", ""))
