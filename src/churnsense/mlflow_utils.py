"""Helpers for experiment tracking with MLflow.

Reproducibility rule of the project: every model can be rebuilt from a Git commit AND an
MLflow run id. So every run records the commit, the branch, whether the tree was clean, and
a fingerprint of the data pointers. A training run on uncommitted code is refused, unless
``ALLOW_DIRTY=1`` is set; such a run is tagged ``git_dirty=true`` and the gate never
promotes it.

MLflow client calls are retried three times with a growing pause. If the tracking server is
down BEFORE the run starts, the stage stops at once (serving is unaffected: it uses a local
copy of the model).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import mlflow

from churnsense.logging_utils import get_logger

log = get_logger(__name__)
T = TypeVar("T")

#: Files whose uncommitted change makes a run non-reproducible (reports are excluded on purpose:
#: the same ``dvc repro`` rewrites them).
POINTER_PATHS = ("src", "params.yaml", "dvc.yaml", "MLproject", "requirements.lock", "*.dvc")
EXPERIMENT = "churnsense-pipeline"
ABLATION_EXPERIMENT = "churnsense-ablation"
REGISTERED_MODEL = "churnsense"


class TrackingError(RuntimeError):
    """The MLflow tracking server cannot be used."""


def retry(times: int = 3, backoff: float = 1.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function up to ``times`` times, pausing ``backoff``, 2x, 4x... seconds."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - any client error may be temporary
                    if attempt == times:
                        raise
                    pause = backoff * 2 ** (attempt - 1)
                    log.warning(
                        "call failed, retrying",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "error": str(exc),
                            "pause_s": pause,
                        },
                    )
                    time.sleep(pause)
            raise AssertionError("unreachable")

        return wrapper

    return decorator


def forget_project_run_env() -> str | None:
    """Clear the run and experiment variables that ``mlflow run`` sets; return the run id.

    ``mlflow run`` starts a run itself and passes it to the program in the environment. The
    program may reuse that run as its parent run, but any further top-level run (for example the
    ablation runs in another experiment) must start on its own, so the variables are removed.
    """
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    return run_id


def tracking_uri() -> str:
    """The MLflow tracking URI from the environment (the compose file sets it)."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def check_tracking_server(uri: str, attempts: int = 3, pause: float = 1.0) -> None:
    """Make sure an HTTP tracking server answers ``/health``; file and sqlite URIs need no check.

    Raises:
        TrackingError: if the server does not answer after ``attempts`` tries.
    """
    if not uri.startswith(("http://", "https://")):
        return
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        if attempt < attempts:
            time.sleep(pause * 2 ** (attempt - 1))
    raise TrackingError(f"MLflow tracking server {uri} is not reachable: {last_error}")


def _git(root: Path, *args: str) -> str | None:
    """Run ``git -C root ...``; return stdout, or None if git fails."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def dirty_pointers(root: Path) -> list[str]:
    """Return the code, config and data-pointer files with uncommitted changes."""
    status = _git(root, "status", "--porcelain", "--", *POINTER_PATHS)
    if status is None:  # fail closed: a model that cannot be tied to a commit is not reproducible
        return ["<git status could not be read>"]
    return [line.split(None, 1)[1] for line in status.splitlines() if line.strip()]


def git_info(root: Path) -> dict[str, Any]:
    """Commit, branch and the list of dirty pointer files (empty strings if not a repo)."""
    return {
        "git_commit": _git(root, "rev-parse", "HEAD") or "unknown",
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty_files": dirty_pointers(root),
    }


def file_md5(path: Path) -> str:
    """MD5 of a file, or "missing" if it does not exist."""
    if not path.is_file():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - a fingerprint, not security


def data_pointer_md5(root: Path) -> str:
    """One fingerprint of the data pointers and parameters (what the data and settings were)."""
    digest = hashlib.md5()  # noqa: S324 - a fingerprint, not security
    for name in ("data/raw/telco_churn.csv.dvc", "data/feedback.dvc", "params.yaml"):
        digest.update(file_md5(root / name).encode())
    return digest.hexdigest()


def flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries: ``{"ray": {"num_cpus": 3}}`` -> ``{"ray.num_cpus": 3}``."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def numeric_metrics(mapping: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Keep only the numeric leaves of a nested dictionary, as MLflow metrics."""
    out: dict[str, float] = {}
    for key, value in flatten(mapping, prefix).items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float) and value == value:  # drops NaN
            out[key] = float(value)
    return out


def safe_metric_name(name: str) -> str:
    """MLflow metric names allow letters, digits, ``_ - . space : /``; replace the rest."""
    return "".join(c if c.isalnum() or c in "_-. :/" else "_" for c in name)


class Tracker:
    """Thin wrapper over the active MLflow run whose calls are retried."""

    @staticmethod
    @retry()
    def params(values: dict[str, Any]) -> None:
        """Log parameters (values are turned into text, long ones are cut for MLflow)."""
        mlflow.log_params({k: str(v)[:500] for k, v in values.items()})

    @staticmethod
    @retry()
    def metrics(values: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        mlflow.log_metrics({safe_metric_name(k): float(v) for k, v in values.items()}, step=step)

    @staticmethod
    @retry()
    def tags(values: dict[str, Any]) -> None:
        """Set tags."""
        mlflow.set_tags({k: str(v) for k, v in values.items()})

    @staticmethod
    @retry()
    def artifact(path: Path, artifact_path: str | None = None) -> None:
        """Upload one file (or a whole folder) as an artifact of the active run."""
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path)
        else:
            mlflow.log_artifact(str(path), artifact_path)


def log_files(paths: Iterable[Path], artifact_path: str | None = None) -> None:
    """Upload every existing file of ``paths`` to the active run."""
    for path in paths:
        if path.exists():
            Tracker.artifact(path, artifact_path)
