"""Export a model to the serving folder and switch the ``current`` pointer.

Layout of ``/srv/churnsense/serving``::

    v3/         an exported MLflow model (self-contained: it carries its own code)
    v4/
    current  -> v4        a symbolic link; the model server loads ``current``

Rules that keep serving safe:

* A model is first copied to ``v<N>.tmp``, VERIFIED there (it must load in a fresh process
  without the repository, and reproduce the stored answers of the smoke rows), and only then
  renamed to ``v<N>``. A failed export leaves nothing behind and ``current`` untouched.
* ``current`` is switched with one atomic rename of a new link, so a reader sees either the
  old model or the new one, never a half-written state.
* Old versions are pruned, but the newest few and the one in use are always kept (rollback
  needs them).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churnsense.logging_utils import get_logger

log = get_logger(__name__)

KEEP_VERSIONS = 3
VERSION_DIR = re.compile(r"^v(\d+)$")
PROBABILITY_TOLERANCE = 1e-6


class ExportError(RuntimeError):
    """An exported model failed verification (nothing was switched)."""


def list_versions(serving_root: Path) -> list[int]:
    """Version numbers of the exported models, oldest first."""
    if not serving_root.is_dir():
        return []
    found = [VERSION_DIR.match(p.name) for p in serving_root.iterdir() if p.is_dir()]
    return sorted(int(m.group(1)) for m in found if m)


def current_version(serving_root: Path) -> int | None:
    """The version the ``current`` link points to, or None if there is none (or it is broken)."""
    link = serving_root / "current"
    if not link.is_symlink():
        return None
    match = VERSION_DIR.match(os.readlink(link))
    if not match or not link.exists():
        return None
    return int(match.group(1))


def switch_current(serving_root: Path, version: int) -> None:
    """Point ``current`` at ``v<version>`` with one atomic rename."""
    target = serving_root / f"v{version}"
    if not target.is_dir():
        raise ExportError(f"{target} does not exist")
    temporary = serving_root / "current.tmp"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    os.symlink(f"v{version}", temporary)  # relative link: works the same inside any container
    os.replace(temporary, serving_root / "current")


def clear_current(serving_root: Path) -> None:
    """Remove the ``current`` link (used when the very first model fails its live checks)."""
    link = serving_root / "current"
    if link.is_symlink():
        link.unlink()


def prune(serving_root: Path, keep: int = KEEP_VERSIONS) -> list[int]:
    """Delete the oldest versions, keeping ``keep`` newest and the one in use. Returns removed."""
    versions = list_versions(serving_root)
    protected = set(versions[-keep:])
    in_use = current_version(serving_root)
    if in_use is not None:
        protected.add(in_use)
    removed = [v for v in versions if v not in protected]
    for version in removed:
        shutil.rmtree(serving_root / f"v{version}")
    return removed


def verify_export(root: Path, model_dir: Path, smoke_file: Path) -> dict[str, Any]:
    """Check an exported model in a fresh process against the stored smoke answers.

    Raises:
        ExportError: if the model cannot load, or its answers differ from the stored ones.
    """
    from churnsense.pipeline.evaluate import ScoringError, score_in_fresh_process

    smoke = json.loads(smoke_file.read_text("utf-8"))
    rows = pd.DataFrame([r["input"] for r in smoke["rows"]])
    from churnsense.model.pyfunc import model_inputs

    try:
        predictions, facts = score_in_fresh_process(root, model_dir, model_inputs(rows))
    except ScoringError as exc:
        raise ExportError(f"the exported model does not run: {exc}") from exc
    expected = pd.DataFrame([r["expected"] for r in smoke["rows"]])
    gap = float(
        np.max(
            np.abs(
                predictions["churn_probability"].to_numpy()
                - expected["churn_probability"].to_numpy()
            )
        )
    )
    if gap > PROBABILITY_TOLERANCE:
        raise ExportError(f"the exported model changed its answers (largest gap {gap:.2e})")
    if not (predictions["risk_band"].to_numpy() == expected["risk_band"].to_numpy()).all():
        raise ExportError("the exported model changed some risk bands")
    return {"smoke_rows": len(rows), "largest_probability_gap": gap, "run_id": facts.get("run_id")}


def export_model(
    root: Path,
    source: Path,
    serving_root: Path,
    version: int,
    smoke_file: Path,
    meta: dict[str, Any],
) -> Path:
    """Copy ``source`` to ``serving_root/v<version>``, verifying it before it becomes visible."""
    serving_root.mkdir(parents=True, exist_ok=True)
    final, temporary = serving_root / f"v{version}", serving_root / f"v{version}.tmp"
    if final.exists():
        raise ExportError(f"{final} already exists; model versions are never overwritten")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    try:
        verification = verify_export(root, temporary, smoke_file)
        (temporary / "meta.json").write_text(
            json.dumps(
                {
                    **meta,
                    "version": version,
                    "verification": verification,
                    "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    os.rename(temporary, final)
    log.info("model exported", extra={"version": version, "path": str(final)})
    return final


def read_meta(version_dir: Path) -> dict[str, Any]:
    """Read ``meta.json`` of an exported model (empty if missing)."""
    meta = version_dir / "meta.json"
    return json.loads(meta.read_text("utf-8")) if meta.is_file() else {}
