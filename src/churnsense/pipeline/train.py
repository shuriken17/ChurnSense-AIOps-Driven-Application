"""Stage 6: train the model and package it.

Inputs:  data/processed/{train,val}.parquet, data/interim/splits/val.parquet (raw fields),
         models/reference_profile.json
Outputs: models/candidate/ (MLflow pyfunc model), models/run_info.json,
         reports/train_metrics.json, reports/plots/ (curves, importances, PR curve CSV)

What happens, in order:

1. Refuse to train on uncommitted code (reproducibility) unless ``ALLOW_DIRTY=1``, and make
   sure the MLflow server answers.
2. For each model family (logistic regression and LightGBM): a random hyperparameter search
   with cross-validation, then a final fit. Every candidate is logged as a nested MLflow run.
3. The winner is the family with the best PR-AUC on the VALIDATION set.
4. The risk-band thresholds are chosen on the validation set (see ``thresholds.py``).
5. Explanations: SHAP importance, permutation importance, plots. The model is packaged with
   its explainer and thresholds as one MLflow pyfunc model.
6. Two ablation runs (without synthetic rows, without engineered features) are logged in a
   separate MLflow experiment, to show what each design choice is really worth.

Design choice worth defending: the search uses REAL rows only. Synthetic rows are noisy
copies of real training rows, so copies of one customer would land in different
cross-validation folds and make every score look better than it is. The final fit then uses
all training rows (real and synthetic) as configured.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import mlflow.pyfunc  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.utils.autologging_utils import disable_autologging  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from churnsense import mlflow_utils as mu  # noqa: E402
from churnsense.config import Params, Paths  # noqa: E402
from churnsense.features.profile import load_profile  # noqa: E402
from churnsense.logging_utils import get_logger  # noqa: E402
from churnsense.mlflow_utils import Tracker  # noqa: E402
from churnsense.model import metrics as mx  # noqa: E402
from churnsense.model import search  # noqa: E402
from churnsense.model.bundle import ModelBundle  # noqa: E402
from churnsense.model.explain import LinearShapExplainer, TreeShapExplainer  # noqa: E402
from churnsense.model.pyfunc import model_inputs, model_kwargs  # noqa: E402
from churnsense.model.thresholds import Thresholds, assign_band, select_thresholds  # noqa: E402
from churnsense.schema import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    RAW_FEATURES,
    categorical_dtype,
)
from churnsense.tracking import StageError, StageReport, tracked_stage, write_json  # noqa: E402

log = get_logger(__name__)

BLUE, ORANGE = "#0072B2", "#E69F00"  # colour-blind safe pair
LIBRARIES = ("lightgbm", "scikit-learn", "pandas", "numpy", "pyarrow", "mlflow", "shap", "ray")
REPORTS_FOR_MLFLOW = (
    "ingest_stats",
    "validation_report",
    "split_report",
    "synthetic_stats",
    "featurize_stats",
    "eda_summary",
)


@dataclass
class FamilyResult:
    """Everything learned about one model family."""

    family: str
    estimator: Any
    search: search.SearchResult
    fit_seconds: float
    val_proba: np.ndarray
    val_metrics: dict[str, float]


# --------------------------------------------------------------------------- helpers
def _library_versions() -> dict[str, str]:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _provenance_tags(paths: Paths, info: dict[str, Any], family: str) -> dict[str, str]:
    """Tags that make the run reproducible and traceable."""
    return {
        "git_commit": info["git_commit"],
        "git_branch": info["git_branch"],
        "git_dirty": str(bool(info["dirty_files"])).lower(),
        "dvc_lock_md5": mu.file_md5(paths.root / "dvc.lock"),
        "data_pointer_md5": mu.data_pointer_md5(paths.root),
        "pipeline_trigger": os.environ.get("PIPELINE_TRIGGER", "manual"),
        "drift_alert_fingerprint": os.environ.get("DRIFT_ALERT_FINGERPRINT", ""),
        "controller_run_id": os.environ.get("CONTROLLER_RUN_ID", ""),
        "model_family": family,
        "ray_version": _library_versions()["ray"],
        "lightgbm_version": _library_versions()["lightgbm"],
        "host": platform.node(),
    }


def _restore(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the fixed categorical dtypes of the features (same categories, same order)."""
    out = frame.copy()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype("object").astype(categorical_dtype(column))
    return out


def _x(frame: pd.DataFrame, columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)) -> pd.DataFrame:
    return frame[list(columns)]


def _y(frame: pd.DataFrame) -> np.ndarray:
    return frame["churn"].to_numpy(dtype=int)


def _fit(
    family: str,
    params: Params,
    best_params: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    columns: tuple[str, ...] | None = None,
) -> Any:
    estimator = search.build_estimator(family, params, columns)
    estimator.set_params(**best_params)
    estimator.fit(x, y)
    return estimator


def _validation_metrics(y: np.ndarray, p: np.ndarray, min_precision: float) -> dict[str, float]:
    recall, _ = mx.recall_at_precision(y, p, min_precision)
    return {**mx.ranking_metrics(y, p), "recall_at_min_precision": recall}


# --------------------------------------------------------------------------- search + fit
def _search_family(
    family: str,
    params: Params,
    train_search: pd.DataFrame,
    train_fit: pd.DataFrame,
    val: pd.DataFrame,
) -> FamilyResult:
    """Search (on real rows), then fit on the final training rows and score the validation set."""
    cfg = params.train
    min_precision = float(cfg.min_precision_at_high)
    with mlflow.start_run(run_name=f"search-{family}", nested=True):
        Tracker.tags({"model_family": family, "stage": "search"})
        with disable_autologging():  # the candidates are logged explicitly below
            result = search.run_search(
                family,
                params,
                _x(train_search),
                _y(train_search),
                n_iter=int(cfg.n_iter),
                cv_folds=int(cfg.cv_folds),
            )
        for i, candidate in enumerate(result.candidates):
            with mlflow.start_run(run_name=f"candidate-{family}-{i:02d}", nested=True):
                Tracker.params(candidate["params"])
                Tracker.metrics(
                    {
                        "cv_pr_auc_mean": candidate["cv_pr_auc_mean"],
                        "cv_pr_auc_std": candidate["cv_pr_auc_std"],
                        "cv_rank": candidate["rank"],
                    }
                )
        Tracker.metrics({"best_cv_pr_auc": result.best_score, "search_seconds": result.seconds})
        log.info(
            "search finished",
            extra={
                "family": family,
                "best_cv_pr_auc": result.best_score,
                "seconds": round(result.seconds, 1),
            },
        )

    with mlflow.start_run(run_name=f"fit-{family}", nested=True):
        Tracker.tags({"model_family": family, "stage": "final_fit"})
        started = time.perf_counter()
        estimator = _fit(family, params, result.best_params, _x(train_fit), _y(train_fit))
        fit_seconds = time.perf_counter() - started
        p_val = np.asarray(estimator.predict_proba(_x(val))[:, 1], dtype=float)
        metrics = _validation_metrics(_y(val), p_val, min_precision)
        Tracker.params(search.clean_param_names(result.best_params))
        Tracker.metrics(
            {f"val_{k}": v for k, v in metrics.items()}
            | {"fit_seconds": fit_seconds, "train_rows": float(len(train_fit))}
        )
    return FamilyResult(family, estimator, result, fit_seconds, p_val, metrics)


# --------------------------------------------------------------------------- plots + importance
def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def make_plots(y: np.ndarray, p: np.ndarray, thresholds: Thresholds, out: Path) -> list[Path]:
    """ROC, precision-recall, calibration and confusion-matrix plots of the validation set."""
    out.mkdir(parents=True, exist_ok=True)
    files = []

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=BLUE)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set(title="ROC curve (validation)", xlabel="False positive rate", ylabel="Recall")
    files.append(out / "roc_curve.png")
    _save(fig, files[-1])

    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color=BLUE)
    ax.scatter(
        [thresholds.val_recall_at_high],
        [thresholds.val_precision_at_high],
        color=ORANGE,
        zorder=3,
        label="HIGH threshold",
    )
    ax.set(title="Precision-recall curve (validation)", xlabel="Recall", ylabel="Precision")
    ax.legend()
    files.append(out / "pr_curve.png")
    _save(fig, files[-1])

    frac_positive, mean_predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_predicted, frac_positive, marker="o", color=BLUE)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set(
        title="Calibration (validation)", xlabel="Predicted probability", ylabel="Share who churned"
    )
    files.append(out / "calibration.png")
    _save(fig, files[-1])

    matrix = confusion_matrix(y, (p >= thresholds.t_high).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(
        title="Confusion matrix at the HIGH threshold",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["stay", "churn"],
        yticklabels=["stay", "churn"],
    )
    files.append(out / "confusion_matrix.png")
    _save(fig, files[-1])
    return files


def importance_table(
    winner: FamilyResult, bundle: ModelBundle, val: pd.DataFrame, params: Params
) -> pd.DataFrame:
    """Mean |SHAP|, permutation importance and LightGBM gain per feature (validation set)."""
    x_val = _x(val)
    shap_values = bundle.explainer.explain(x_val).values
    result = permutation_importance(
        winner.estimator,
        x_val,
        _y(val),
        scoring="average_precision",
        n_repeats=int(params.train.get("permutation_repeats", 3)),
        random_state=int(params.seed),
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )
    if winner.family == "lightgbm":
        table["lgbm_gain"] = winner.estimator.booster_.feature_importance(importance_type="gain")
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_importance(table: pd.DataFrame, path: Path) -> None:
    """Bar chart of the 15 features with the largest mean |SHAP|."""
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE)
    ax.set(title="Global feature importance (mean |SHAP|, log-odds)", xlabel="Mean |SHAP|")
    _save(fig, path)


# --------------------------------------------------------------------------- packaging
def raw_example(paths: Paths, rows: int = 2) -> pd.DataFrame:
    """A few raw input rows (with the exact types of the signature) as the input example."""
    return model_inputs(pd.read_parquet(paths.split_file("val")).head(rows))


def package_model(paths: Paths, bundle: ModelBundle) -> tuple[Any, float]:
    """Save the pyfunc model to ``models/candidate`` and log it to the active MLflow run.

    Returns the MLflow ``ModelInfo`` and the size of the saved model in megabytes.
    """
    requirements = paths.root / "requirements.lock"
    if not requirements.is_file():
        raise StageError(f"{requirements} is missing: the model must record its libraries")
    example = raw_example(paths)
    with tempfile.TemporaryDirectory() as tmp:
        bundle_file = Path(tmp) / "bundle.joblib"
        joblib.dump(bundle, bundle_file)
        kwargs = model_kwargs(bundle_file, example, requirements)
        if paths.candidate.exists():
            shutil.rmtree(paths.candidate)
        paths.ensure_parents(paths.candidate)
        mlflow.pyfunc.save_model(path=str(paths.candidate), **kwargs)
        info = mu.retry()(mlflow.pyfunc.log_model)(name="model", **kwargs)

    # Quick in-process check: the saved model loads and gives sensible answers.
    answer = mlflow.pyfunc.load_model(str(paths.candidate)).predict(example)
    if not (answer["churn_probability"].between(0, 1).all() and len(answer) == len(example)):
        raise StageError("the packaged model gave invalid predictions on its own input example")
    size_mb = sum(f.stat().st_size for f in paths.candidate.rglob("*") if f.is_file()) / 1e6
    return info, size_mb


# --------------------------------------------------------------------------- ablations
def run_ablations(
    params: Params,
    paths: Paths,
    winner: FamilyResult,
    parent_run_id: str,
    train_all: pd.DataFrame,
    train_real: pd.DataFrame,
    val: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Log two ablation runs: without synthetic rows, and without the engineered features."""
    min_precision = float(params.train.min_precision_at_high)
    mlflow.set_experiment(mu.ABLATION_EXPERIMENT)
    base = winner.val_metrics
    cases = {
        "synthetic_rows_off": (train_real, tuple(FEATURE_COLUMNS)),
        "engineered_features_off": (train_all, tuple(RAW_FEATURES)),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (rows, columns) in cases.items():
        with mlflow.start_run(run_name=f"ablation-{name}"):
            Tracker.tags(
                {"ablation": name, "parent_run_id": parent_run_id, "model_family": winner.family}
            )
            estimator = _fit(
                winner.family,
                params,
                winner.search.best_params,
                _x(rows, columns),
                _y(rows),
                columns,
            )
            p = np.asarray(estimator.predict_proba(_x(val, columns))[:, 1], dtype=float)
            metrics = _validation_metrics(_y(val), p, min_precision)
            deltas = {f"delta_{k}_vs_main": metrics[k] - base[k] for k in metrics}
            Tracker.params({"ablation": name, "train_rows": len(rows), "features": len(columns)})
            Tracker.metrics({f"val_{k}": v for k, v in metrics.items()} | deltas)
            results[name] = {
                **metrics,
                **deltas,
                "train_rows": float(len(rows)),
                "n_features": float(len(columns)),
            }
    mlflow.set_experiment(mu.EXPERIMENT)
    return results


# --------------------------------------------------------------------------- the stage
@tracked_stage("train", report="train_metrics.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the train stage."""
    started = time.perf_counter()
    cfg = params.train
    info = mu.git_info(paths.root)
    if info["dirty_files"] and os.environ.get("ALLOW_DIRTY") != "1":
        raise StageError(
            "uncommitted changes in "
            + ", ".join(info["dirty_files"])
            + ": commit them first (every model must be tied to a Git commit), "
            "or set ALLOW_DIRTY=1 for a throw-away run that can never be promoted"
        )
    uri = mu.tracking_uri()
    try:
        mu.check_tracking_server(uri)
    except mu.TrackingError as exc:
        raise StageError(str(exc)) from exc

    mlflow.set_tracking_uri(uri)
    # Started by `mlflow run`? Then that run becomes the parent run (and its experiment is used).
    from_mlflow_run = bool(os.environ.get("MLFLOW_RUN_ID"))
    if not from_mlflow_run:
        mlflow.set_experiment(mu.EXPERIMENT)
    mlflow.lightgbm.autolog(log_models=False, log_datasets=False, silent=True)
    mlflow.sklearn.autolog(
        log_models=False,
        log_datasets=False,
        max_tuning_runs=0,
        log_post_training_metrics=False,
        silent=True,
    )

    train_all = pd.read_parquet(paths.processed_file("train"))
    val = pd.read_parquet(paths.processed_file("val"))
    train_all, val = _restore(train_all), _restore(val)
    train_real = train_all[train_all["source"] != "synthetic"]
    train_search = train_real if cfg.get("search_real_only", True) else train_all
    train_fit = train_all if cfg.get("final_fit_with_synthetic", True) else train_real
    load_profile(paths.reference_profile)  # fails early if the featurize stage did not run

    families = list(cfg.families)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parent_run = (
        mlflow.start_run() if from_mlflow_run else mlflow.start_run(run_name=f"train-{stamp}")
    )
    with parent_run as parent:
        mu.forget_project_run_env()
        run_id = parent.info.run_id
        Tracker.tags(_provenance_tags(paths, info, "pending"))
        Tracker.params(mu.flatten(dict(params)))

        results = {f: _search_family(f, params, train_search, train_fit, val) for f in families}
        winner = max(results.values(), key=lambda r: r.val_metrics["pr_auc"])
        Tracker.tags({"model_family": winner.family})
        log.info(
            "winner chosen",
            extra={"family": winner.family, "val_pr_auc": winner.val_metrics["pr_auc"]},
        )

        y_val, p_val = _y(val), winner.val_proba
        thresholds = select_thresholds(
            y_val, p_val, float(cfg.min_precision_at_high), float(cfg.medium_band_recall)
        )
        bands = assign_band(p_val, thresholds.t_high, thresholds.t_medium)
        at_high = mx.all_metrics(y_val, p_val, thresholds.t_high, float(cfg.min_precision_at_high))
        per_contract = mx.group_metrics(val["contract"], y_val, p_val, thresholds.t_high)

        explainer = (
            TreeShapExplainer(winner.estimator)
            if winner.family == "lightgbm"
            else LinearShapExplainer(winner.estimator, _x(train_real))
        )
        trained_at = datetime.now(UTC).isoformat(timespec="seconds")
        bundle = ModelBundle(
            family=winner.family,
            estimator=winner.estimator,
            explainer=explainer,
            thresholds=thresholds,
            run_id=run_id,
            trained_at=trained_at,
            metadata={
                "library_versions": _library_versions(),
                "git_commit": info["git_commit"],
                "val_metrics": at_high,
                "best_params": search.clean_param_names(winner.search.best_params),
            },
        )

        # tables and plots
        paths.plots_dir.mkdir(parents=True, exist_ok=True)
        table = importance_table(winner, bundle, val, params)
        table.to_csv(paths.plots_dir / "feature_importance.csv", index=False)
        plot_importance(table, paths.plots_dir / "feature_importance.png")
        plots = make_plots(y_val, p_val, thresholds, paths.plots_dir)
        precision, recall, cut = precision_recall_curve(y_val, p_val)
        pd.DataFrame({"recall": recall[:-1], "precision": precision[:-1], "threshold": cut}).to_csv(
            paths.plots_dir / "pr_curve_val.csv", index=False
        )

        info_model, size_mb = package_model(paths, bundle)

        # everything about the run goes to MLflow (beyond autolog: tags, data, operations)
        data_metrics: dict[str, float] = {}
        for name in REPORTS_FOR_MLFLOW:
            report_file = paths.report(f"{name}.json")
            if report_file.is_file():
                data_metrics |= mu.numeric_metrics(
                    json.loads(report_file.read_text("utf-8")), f"data.{name}."
                )
        Tracker.metrics(data_metrics)
        Tracker.params(
            {f"best.{k}": v for k, v in search.clean_param_names(winner.search.best_params).items()}
            | {"t_high": thresholds.t_high, "t_medium": thresholds.t_medium}
        )
        Tracker.metrics(
            {f"val_{k}": v for k, v in at_high.items() if isinstance(v, int | float)}
            | {f"val_band_share_{b}": float((bands == b).mean()) for b in ("HIGH", "MEDIUM", "LOW")}
            | {
                f"val_contract_{c}_{k}": v
                for c, d in per_contract.items()
                for k, v in d.items()
                if k in ("precision", "recall") and v is not None
            }
            | {
                f"val_{k}_by_family_{f}": v
                for f, r in results.items()
                for k, v in r.val_metrics.items()
            }
            | {"model_size_mb": size_mb, "time_train_seconds": time.perf_counter() - started}
            | {f"time_search_{f}_seconds": r.search.seconds for f, r in results.items()}
            | {f"time_fit_{f}_seconds": r.fit_seconds for f, r in results.items()}
        )
        mu.log_files(
            [paths.report(f"{n}.json") for n in REPORTS_FOR_MLFLOW]
            + [paths.reference_profile, paths.root / "params.yaml"],
            "reports",
        )
        mu.log_files(
            [
                *plots,
                paths.plots_dir / "feature_importance.csv",
                paths.plots_dir / "feature_importance.png",
                paths.plots_dir / "pr_curve_val.csv",
            ],
            "plots",
        )

    ablations = (
        run_ablations(params, paths, winner, run_id, train_fit, train_real, val)
        if cfg.get("ablations", True)
        else {}
    )

    run_info = {
        "run_id": run_id,
        "experiment": mu.EXPERIMENT,
        "model_uri": info_model.model_uri,
        "model_id": getattr(info_model, "model_id", None),
        "family": winner.family,
        "git_commit": info["git_commit"],
        "git_dirty": bool(info["dirty_files"]),
        "trained_at": trained_at,
        "val_pr_auc": winner.val_metrics["pr_auc"],
        "t_high": thresholds.t_high,
        "t_medium": thresholds.t_medium,
    }
    write_json(paths.run_info, run_info)
    return StageReport(
        rows_in=len(train_fit),
        rows_out=len(train_fit),
        extra={
            "run_id": run_id,
            "winner": winner.family,
            "model_size_mb": round(size_mb, 3),
            "thresholds": thresholds.as_dict(),
            "validation_at_high_threshold": at_high,
            "validation_by_family": {f: r.val_metrics for f, r in results.items()},
            "best_cv_pr_auc_by_family": {f: r.search.best_score for f, r in results.items()},
            "search_seconds_by_family": {f: round(r.search.seconds, 2) for f, r in results.items()},
            "fit_seconds_by_family": {f: round(r.fit_seconds, 2) for f, r in results.items()},
            "n_candidates_per_family": int(cfg.n_iter),
            "search_rows": len(train_search),
            "final_fit_rows": len(train_fit),
            "band_shares_validation": {
                b: float((bands == b).mean()) for b in ("HIGH", "MEDIUM", "LOW")
            },
            "recall_by_contract_validation": per_contract,
            "ablations": ablations,
            "git_dirty": bool(info["dirty_files"]),
        },
    )


def cli() -> None:
    """Command line: ``python -m churnsense.pipeline.train [--family F] [--n-iter N]``."""
    parser = argparse.ArgumentParser(description="Train the ChurnSense model.")
    parser.add_argument("--family", choices=[*search.FAMILIES, "both"], default=None)
    parser.add_argument("--n-iter", type=int, default=None, help="candidates per family")
    args = parser.parse_args()

    def configure(params: Params) -> None:
        if args.family and args.family != "both":
            params.train["families"] = [args.family]
        if args.n_iter:
            params.train["n_iter"] = args.n_iter

    run.main(configure)


if __name__ == "__main__":
    cli()
