"""Helpers that build test data (feedback batches, broken rows)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from churnsense.config import Paths
from churnsense.schema import VALIDATED_COLUMNS
from churnsense.standardise import standardise_frame
from tests.conftest import raw_frame, sha256_bytes


def canonical_rows(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """The fixture rows in canonical (validated) layout, ready for the later stages."""
    df = standardise_frame(raw_frame(), source="real").sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    new_customer = df["total_charges"].isna()
    df.loc[new_customer, "total_charges"] = 0.0
    df["total_charges_imputed"] = 0
    ints = ["senior_citizen", "churn", "tenure_months", "total_charges_imputed"]
    return df.astype({c: "int64" for c in ints})[list(VALIDATED_COLUMNS)]


def write_feedback_batch(
    paths: Paths,
    name: str,
    n_rows: int,
    *,
    created_at: str,
    prefix: str,
    seed: int = 1,
    ids: list[str] | None = None,
    corrupt: bool = False,
    list_in_manifest: bool = True,
) -> pd.DataFrame:
    """Write one feedback parquet file and (optionally) register it in the manifest."""
    frame = canonical_rows(n_rows, seed=seed).drop(columns=["total_charges_imputed"])
    frame["customer_id"] = ids if ids else [f"{prefix}-{i}" for i in range(n_rows)]
    frame = frame.drop(columns=["source", "batch_id", "event_ts"])
    path = paths.feedback_dir / name
    frame.to_parquet(path, index=False)
    digest = sha256_bytes(path.read_bytes())
    if corrupt:
        digest = "0" * 64
    if list_in_manifest:
        manifest_path = paths.feedback_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["batches"].append(
            {
                "file": name,
                "sha256": digest,
                "rows": n_rows,
                "scenario": "test",
                "synthetic_labels": True,
                "created_at": created_at,
            }
        )
        manifest_path.write_text(json.dumps(manifest), "utf-8")
    return frame


def write_params_yaml(path: Path, text: str) -> Path:
    """Write a small params file for configuration tests."""
    path.write_text(text, encoding="utf-8")
    return path


def fitted_bundle(
    family: str = "lightgbm", n_rows: int = 500, seed: int = 0, run_id: str = "test-run"
):
    """A small, quickly fitted ModelBundle (no MLflow) for the model and explanation tests."""
    from churnsense.features.derive import derive_features
    from churnsense.model.bundle import ModelBundle
    from churnsense.model.explain import LinearShapExplainer, TreeShapExplainer
    from churnsense.model.search import build_estimator
    from churnsense.model.thresholds import select_thresholds
    from tests.conftest import make_params

    rows = canonical_rows(n_rows, seed)
    features = derive_features(rows)
    y = rows["churn"].to_numpy(dtype=int)
    estimator = build_estimator(family, make_params())
    if family == "lightgbm":
        estimator.set_params(n_estimators=40, num_leaves=8, learning_rate=0.1, min_child_samples=10)
    estimator.fit(features, y)
    explainer = (
        TreeShapExplainer(estimator)
        if family == "lightgbm"
        else LinearShapExplainer(estimator, features)
    )
    thresholds = select_thresholds(y, estimator.predict_proba(features)[:, 1], 0.5, 0.9)
    bundle = ModelBundle(family, estimator, explainer, thresholds, run_id, "2026-01-01T00:00:00")
    return bundle, rows


def git_commit_all(root: Path) -> None:
    """Make ``root`` a Git repository with everything committed (generated folders ignored)."""
    import subprocess

    (root / ".gitignore").write_text(
        "data/\nreports/\nmodels/\nserving/\nartifacts/\nmlflow.db\n__pycache__/\n*.dvc.tmp\n"
    )
    for command in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "t@t.t"],
        ["config", "user.name", "tester"],
        ["add", "-A"],
        ["commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True, capture_output=True)


def export_test_model(serving: Path, version: int, family: str, run_id: str) -> Path:
    """Save a small real MLflow model as ``serving/v<version>`` (the layout the server loads)."""
    import joblib
    import mlflow.pyfunc

    from churnsense.model.pyfunc import model_inputs, model_kwargs
    from tests.conftest import REPO

    bundle, rows = fitted_bundle(family, run_id=run_id)
    target = serving / f"v{version}"
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle_file = serving.parent / f"b-{run_id}" / "bundle.joblib"
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_file)
    kwargs = model_kwargs(bundle_file, model_inputs(rows.head(2)), REPO / "requirements.lock")
    mlflow.pyfunc.save_model(path=str(target), **kwargs)
    return target
