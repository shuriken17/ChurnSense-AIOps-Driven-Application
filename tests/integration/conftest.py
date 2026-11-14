"""One trained model for all the model tests: the pipeline, train, evaluate and gate run once."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import mlflow
import pytest

from churnsense import mlflow_utils as mu
from churnsense.config import Paths
from churnsense.pipeline import (
    evaluate,
    featurize,
    gate,
    ingest,
    split,
    synthesize,
    train,
    validate,
)
from tests.conftest import REPO, make_params, make_workspace
from tests.factories import git_commit_all


@pytest.fixture(scope="session")
def trained(ray_cluster, tmp_path_factory):
    """Run ingest..featurize, train, evaluate and gate on the 500-row fixture, with real MLflow."""
    root = tmp_path_factory.mktemp("trained")
    paths = make_workspace(root)
    params = make_params()
    params.train["n_iter"] = 2
    params.train["cv_folds"] = 2
    for stage in (ingest, validate, split, synthesize, featurize):
        stage.run(params, paths)
    shutil.copy(REPO / "requirements.lock", root)
    shutil.copy(REPO / "params.yaml", root)
    shutil.copytree(REPO / "scripts", root / "scripts")
    git_commit_all(root)

    uri = f"sqlite:///{root / 'mlflow.db'}"
    patch = pytest.MonkeyPatch()
    patch.setenv("MLFLOW_TRACKING_URI", uri)
    patch.setenv("SERVING_DIR", str(root / "serving"))
    patch.delenv("MODEL_SERVER_URL", raising=False)
    patch.delenv("ALLOW_DIRTY", raising=False)
    patch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow.set_tracking_uri(uri)
    for name in (mu.EXPERIMENT, mu.ABLATION_EXPERIMENT):
        mlflow.create_experiment(name, artifact_location=str(root / "artifacts" / name))
    train.run(params, paths)
    evaluate.run(params, paths)
    gate.run(params, paths)
    yield SimpleNamespace(paths=paths, params=params, root=root, uri=uri)
    patch.undo()


@pytest.fixture
def work(trained, tmp_path, monkeypatch):
    """A private copy of the trained workspace (with its own MLflow database and serving folder)."""
    root = tmp_path / "work"
    shutil.copytree(trained.root, root, symlinks=True, ignore=shutil.ignore_patterns("serving"))
    uri = f"sqlite:///{root / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("SERVING_DIR", str(tmp_path / "serving"))
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)  # no live services in these tests
    mlflow.set_tracking_uri(uri)
    return SimpleNamespace(
        paths=Paths(root=root),
        params=trained.params,
        root=root,
        serving=tmp_path / "serving",
        uri=uri,
    )
