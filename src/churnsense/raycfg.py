"""Ray start-up in one place.

Settings come from params.yaml (ray section): 3 CPUs, an 800 MB object store, no dashboard.
Ray's own memory monitor is switched on at 85% so that a runaway task is killed and
retried instead of the whole VM running out of memory.

Why Ray at all, on one small machine? It spreads work over 3 cores, it is required by the
course rubric for data engineering, and moving to a real cluster only means changing
``ray.init()`` to ``ray.init(address=...)``. A plain pandas script would be about as fast
at this data size; the report says so openly.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Must be set before Ray Data is imported: no progress bars in logs and containers.
os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")

import ray  # noqa: E402
from ray.data import DataContext  # noqa: E402

from churnsense.config import Params  # noqa: E402
from churnsense.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

MB = 1024 * 1024


def ensure_workers_can_import() -> None:
    """Make sure Ray worker processes can ``import churnsense``.

    Workers only inherit the PYTHONPATH environment variable, not the ``sys.path`` of the
    program that started Ray. So the folder that contains this package is added to
    PYTHONPATH before Ray starts (a no-op when it is already there, as in the Docker image).
    """
    package_parent = str(Path(__file__).resolve().parents[1])
    current = os.environ.get("PYTHONPATH", "")
    if package_parent not in current.split(os.pathsep):
        os.environ["PYTHONPATH"] = package_parent + (os.pathsep + current if current else "")


@contextmanager
def ray_session(params: Params) -> Iterator[None]:
    """Start a local Ray (or reuse one that is already running) and stop it afterwards.

    If the caller already started Ray (tests do), this context manager leaves it running.
    """
    # Must be set before Ray starts; the compose file sets it too.
    os.environ.setdefault("RAY_memory_usage_threshold", "0.85")
    ensure_workers_can_import()
    started_here = False
    if not ray.is_initialized():
        log.info(
            "starting Ray",
            extra={"num_cpus": params.ray.num_cpus, "object_store_mb": params.ray.object_store_mb},
        )
        ray.init(
            num_cpus=int(params.ray.num_cpus),
            object_store_memory=int(params.ray.object_store_mb) * MB,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            logging_level="WARNING",
        )
        started_here = True

    context = DataContext.get_current()
    context.execution_options.preserve_order = True  # same input order -> same output order
    context.max_map_retries = int(params.ray.max_task_retries)  # retry a failed map task
    for name in ("ray.data", "ray.data._internal"):
        logging.getLogger(name).setLevel(logging.WARNING)  # keep Ray Data's chatter out of logs
    try:
        yield
    finally:
        if started_here:
            ray.shutdown()
            log.info("Ray stopped")
