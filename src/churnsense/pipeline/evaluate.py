"""Stage 7: evaluate the new model (the challenger) and compare it with the champion.

Inputs:  models/candidate/, models/run_info.json, data/interim/splits/{test_golden,test_recent}
         and, if one exists, the champion in ``/srv/churnsense/serving/current``
Outputs: reports/eval_metrics.json (DVC metrics), reports/comparison.json,
         reports/smoke_rows.json (20 golden rows with the answers the model must reproduce)

The challenger and the champion are each scored in a FRESH Python process by
``scripts/score_model.py``. That proves two things at once: the exported model is
self-contained (the repository is not on the path), and the champion really runs its OWN
code (inside one process the second ``import churnsense`` would reuse the first's modules).

Two held-out sets are used, neither ever seen in training or threshold selection:

* ``test_golden``: a fixed 15% of the real data. It never changes, so scores are comparable
  across every retraining and it guards against slow degradation ("model rot").
* ``test_recent``: 30% of the newest feedback batch. It shows how the model does on
  customers that look like today's, which is what drift is about. It is empty until
  feedback exists.

This stage is ``always_changed`` in ``dvc.yaml``: the champion can change without any file
in the repository changing, so the comparison must be recomputed on every run.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churnsense import mlflow_utils as mu
from churnsense.config import Params, Paths, serving_dir
from churnsense.logging_utils import get_logger
from churnsense.model import metrics as mx
from churnsense.model.pyfunc import model_inputs
from churnsense.model.thresholds import BANDS
from churnsense.tracking import StageError, StageReport, tracked_stage, write_json

log = get_logger(__name__)

CORE_LIBRARIES = ("lightgbm", "scikit-learn", "pandas", "numpy", "pyarrow", "scipy", "mlflow")
MIN_RECENT_FOR_COMPARISON = 50


class ScoringError(RuntimeError):
    """A model could not be scored in a fresh process."""


def score_in_fresh_process(
    root: Path, model_dir: Path, raw: pd.DataFrame, *, latency: int = 0, additivity: int = 0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score ``raw`` rows with the model in ``model_dir`` in a new Python process.

    The new process runs with an empty ``PYTHONPATH`` and a scratch working folder, so the
    repository code is NOT importable: only the model folder's own code is.

    Returns:
        The predictions (same row order as ``raw``) and the facts the scoring script printed.
    """
    script = root / "scripts" / "score_model.py"
    if not script.is_file():
        raise ScoringError(f"{script} is missing")
    with tempfile.TemporaryDirectory() as tmp:
        in_file, out_file = Path(tmp) / "in.parquet", Path(tmp) / "out.parquet"
        raw.reset_index(drop=True).to_parquet(in_file)
        command = [sys.executable, str(script), str(model_dir), str(in_file), str(out_file)]
        if latency:
            command += ["--latency", str(latency)]
        if additivity:
            command += ["--additivity", str(additivity)]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        done = subprocess.run(  # noqa: S603 - fixed command, our own script
            command, cwd=tmp, env=env, capture_output=True, text=True, timeout=900, check=False
        )
        if done.returncode != 0:
            tail = " | ".join(done.stderr.strip().splitlines()[-3:])
            raise ScoringError(f"scoring {model_dir.name} failed (exit {done.returncode}): {tail}")
        json_lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
        if not json_lines:
            raise ScoringError(f"scoring {model_dir.name} printed no result")
        return pd.read_parquet(out_file), json.loads(json_lines[-1])


def read_split(paths: Paths, name: str) -> pd.DataFrame:
    """Read a split of validated raw rows."""
    return pd.read_parquet(paths.split_file(name)).reset_index(drop=True)


def set_metrics(
    split: pd.DataFrame, predictions: pd.DataFrame, t_high: float, min_precision: float
) -> dict[str, Any] | None:
    """All metrics of one scored set, or None if the set is empty or has a single class."""
    y = split["churn"].to_numpy(dtype=int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p = predictions["churn_probability"].to_numpy(dtype=float)
    metrics = mx.all_metrics(y, p, t_high, min_precision)
    metrics["by_contract"] = mx.group_metrics(split["contract"], y, p, t_high)
    return metrics


def dependency_mismatches(model_dir: Path) -> list[str]:
    """Core libraries whose installed version differs from the version the model recorded."""
    requirements = model_dir / "requirements.txt"
    if not requirements.is_file():
        return ["requirements.txt is missing from the model"]
    pinned = {}
    for line in requirements.read_text("utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            pinned[name.strip().lower().replace("_", "-")] = version.strip()
    problems = []
    for library in CORE_LIBRARIES:
        if library not in pinned:
            problems.append(f"{library}: not pinned in the model")
            continue
        try:
            installed = metadata.version(library)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != pinned[library]:
            problems.append(f"{library}: model {pinned[library]}, installed {installed}")
    return problems


def sanity_checks(
    predictions: pd.DataFrame,
    facts: dict[str, Any],
    model_dir: Path,
    golden_positive_rate: float,
    git_dirty: bool,
) -> dict[str, Any]:
    """Facts the gate turns into pass or fail (it holds the limits, this holds the facts)."""
    probabilities = predictions["churn_probability"].to_numpy(dtype=float)
    factors_ok = True
    for text in predictions["top_factors"].head(200):
        try:
            parsed = json.loads(text)
            factors_ok &= len(parsed) == 3 and all(
                {"feature", "value", "impact", "direction"} <= set(item) for item in parsed
            )
        except (TypeError, ValueError):
            factors_ok = False
    problems = dependency_mismatches(model_dir)
    additivity = facts.get("additivity_max_error")
    return {
        "loads_in_fresh_process": True,
        "no_nan_predictions": bool(np.isfinite(probabilities).all()),
        "probabilities_in_range": bool(((probabilities >= 0) & (probabilities <= 1)).all()),
        "bands_valid": bool(predictions["risk_band"].isin(BANDS).all()),
        "top_factors_valid": bool(factors_ok),
        "shap_additivity_max_error": additivity,
        "shap_additive": additivity is not None and additivity <= 1e-6,
        "latency_ms_p50": facts.get("latency_ms_p50"),
        "latency_ms_p95": facts.get("latency_ms_p95"),
        "model_size_mb": round(
            sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6, 3
        ),
        "golden_positive_rate": golden_positive_rate,
        "dependencies_match": not problems,
        "dependency_mismatches": problems,
        "git_dirty": git_dirty,
    }


def smoke_rows(
    golden: pd.DataFrame, predictions: pd.DataFrame, n_rows: int, seed: int, run_id: str
) -> dict[str, Any]:
    """Pick ``n_rows`` golden rows with the answers the challenger gave (used to verify deploys)."""
    chosen = golden.sample(n=min(n_rows, len(golden)), random_state=seed).index
    inputs = json.loads(model_inputs(golden.loc[chosen]).to_json(orient="records"))
    expected = predictions.loc[chosen, ["churn_probability", "risk_band"]]
    return {
        "model_run_id": run_id,
        "rows": [
            {
                "input": inputs[i],
                "expected": json.loads(expected.iloc[[i]].to_json(orient="records"))[0],
            }
            for i in range(len(chosen))
        ],
    }


def log_to_mlflow(run_id: str, summary: dict[str, Any], paths: Paths) -> bool:
    """Add the evaluation to the training run in MLflow. Returns False if MLflow is unavailable.

    Evaluation does not depend on MLflow being up: the reports are the source of truth for
    the gate, and a missing log line must not stop a deployment decision.
    """
    import mlflow

    uri = mu.tracking_uri()
    try:
        mu.check_tracking_server(uri, attempts=1)
        mu.forget_project_run_env()  # we resume the training run by its id, not mlflow run's
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_id=run_id):
            numbers: dict[str, float] = {}
            for name in ("golden", "recent"):
                block = summary["challenger"].get(name)
                if block:
                    numbers |= mu.numeric_metrics(
                        {k: v for k, v in block.items() if k != "by_contract"}, f"{name}_"
                    )
            champion = summary.get("champion")
            if champion and champion.get("golden"):
                numbers |= mu.numeric_metrics(
                    {k: v for k, v in champion["golden"].items() if k != "by_contract"},
                    "champion_golden_",
                )
            numbers |= mu.numeric_metrics(
                {k: v for k, v in summary["sanity"].items() if not isinstance(v, list | bool)},
                "sanity_",
            )
            numbers |= mu.numeric_metrics(summary.get("comparison", {}), "cmp_")
            mu.Tracker.metrics(numbers)
            mu.Tracker.tags({"evaluated": "true"})
            with tempfile.TemporaryDirectory() as tmp:
                summary_file = Path(tmp) / "eval_metrics.json"
                write_json(summary_file, {"status": "ok", **summary})
                mu.log_files(
                    [
                        summary_file,
                        paths.report("comparison.json"),
                        paths.report("smoke_rows.json"),
                    ],
                    "evaluation",
                )
        return True
    except Exception as exc:  # noqa: BLE001 - any tracking failure is non-fatal here
        log.warning("evaluation was not logged to MLflow", extra={"error": str(exc)})
        return False


@tracked_stage("evaluate", report="eval_metrics.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the evaluate stage."""
    cfg, gate = params.evaluate, params.gate
    min_precision = float(gate.min_precision)
    if not paths.run_info.is_file() or not paths.candidate.is_dir():
        raise StageError("no trained model found: run the train stage first")
    run_info = json.loads(paths.run_info.read_text("utf-8"))

    golden = read_split(paths, "test_golden")
    recent = read_split(paths, "test_recent")
    both = pd.concat([golden, recent], ignore_index=True)
    inputs = model_inputs(both)
    n_golden = len(golden)

    try:
        predictions, facts = score_in_fresh_process(
            paths.root, paths.candidate, inputs, latency=int(cfg.latency_calls), additivity=100
        )
    except ScoringError as exc:
        raise StageError(f"the challenger does not load or run in a fresh process: {exc}") from exc
    pred_golden = predictions.iloc[:n_golden].reset_index(drop=True)
    pred_recent = predictions.iloc[n_golden:].reset_index(drop=True)
    t_high = float(facts["t_high"])

    challenger = {
        "run_id": facts["run_id"],
        "family": facts["family"],
        "trained_at": facts["trained_at"],
        "thresholds": {"t_high": t_high, "t_medium": float(facts["t_medium"])},
        "golden": set_metrics(golden, pred_golden, t_high, min_precision),
        "recent": set_metrics(recent, pred_recent, t_high, min_precision),
    }
    if challenger["golden"] is None:
        raise StageError("the golden test set is empty or has a single class")

    sanity = sanity_checks(
        predictions,
        facts,
        paths.candidate,
        golden_positive_rate=float((pred_golden["churn_probability"] >= t_high).mean()),
        git_dirty=bool(run_info.get("git_dirty")),
    )

    # --- the champion, if there is one
    champion: dict[str, Any] | None = None
    comparison: dict[str, Any] = {"champion": False}
    current = serving_dir() / "current"
    if current.exists():
        try:
            champ_pred, champ_facts = score_in_fresh_process(paths.root, current.resolve(), inputs)
            champ_high = float(champ_facts["t_high"])
            cg, cr = champ_pred.iloc[:n_golden].reset_index(drop=True), champ_pred.iloc[n_golden:]
            champion = {
                "run_id": champ_facts["run_id"],
                "family": champ_facts["family"],
                "version_dir": current.resolve().name,
                "thresholds": {"t_high": champ_high, "t_medium": float(champ_facts["t_medium"])},
                "golden": set_metrics(golden, cg, champ_high, min_precision),
                "recent": set_metrics(recent, cr.reset_index(drop=True), champ_high, min_precision),
            }
            comparison = {"champion": True, "champion_run_id": champ_facts["run_id"]}
            y_golden = golden["churn"].to_numpy(dtype=int)
            comparison["golden"] = mx.paired_bootstrap(
                y_golden,
                pred_golden["churn_probability"].to_numpy(),
                cg["churn_probability"].to_numpy(),
                n_resamples=int(cfg.bootstrap_resamples),
                seed=int(params.seed),
                min_precision=min_precision,
            )
            y_recent = recent["churn"].to_numpy(dtype=int)
            if len(y_recent) >= MIN_RECENT_FOR_COMPARISON and y_recent.min() != y_recent.max():
                comparison["recent"] = mx.paired_bootstrap(
                    y_recent,
                    pred_recent["churn_probability"].to_numpy(),
                    cr["churn_probability"].to_numpy(),
                    n_resamples=int(cfg.bootstrap_resamples),
                    seed=int(params.seed),
                    min_precision=min_precision,
                )
        except ScoringError as exc:  # fail closed: the gate will refuse
            champion = {"error": str(exc)}
            comparison = {"champion": True, "error": str(exc)}
            log.error("the champion could not be scored", extra={"error": str(exc)})

    write_json(paths.report("comparison.json"), comparison)
    write_json(
        paths.report("smoke_rows.json"),
        smoke_rows(golden, pred_golden, int(cfg.smoke_rows), int(params.seed), facts["run_id"]),
    )

    summary = {
        "challenger": challenger,
        "champion": champion,
        "sanity": sanity,
        "comparison": comparison,
        "recent_rows": len(recent),
        "recent_positives": int(recent["churn"].sum()) if len(recent) else 0,
        "golden_rows": n_golden,
        "model_run_id": run_info["run_id"],
    }
    summary["logged_to_mlflow"] = log_to_mlflow(run_info["run_id"], summary, paths)
    return StageReport(rows_in=len(both), rows_out=len(both), extra=summary)


if __name__ == "__main__":
    run.main()
