"""Unit tests for the search module, the MLflow helpers, the serving folder and the config files."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from churnsense import mlflow_utils as mu
from churnsense.deploy import export
from churnsense.features.derive import derive_features
from churnsense.model import search
from churnsense.schema import FEATURE_COLUMNS, RAW_FEATURES
from tests.conftest import REPO, make_params
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ search
@pytest.fixture(scope="module")
def data():
    rows = canonical_rows(500)
    return derive_features(rows), rows["churn"].to_numpy(dtype=int)


def test_estimators_have_the_documented_types_and_settings():
    params = make_params()
    assert isinstance(search.build_estimator("logreg", params), Pipeline)
    lgbm = search.build_estimator("lightgbm", params)
    assert isinstance(lgbm, LGBMClassifier)
    assert lgbm.get_params()["deterministic"] is True and lgbm.get_params()["random_state"] == 42
    assert lgbm.get_params()["class_weight"] is None  # no re-weighting, by design
    with pytest.raises(ValueError, match="unknown model family"):
        search.build_estimator("forest", params)
    with pytest.raises(ValueError, match="unknown model family"):
        search.distributions("forest")


def test_search_spaces_sample_valid_values():
    rng = np.random.default_rng(0)
    space = search.distributions("lightgbm")
    for _ in range(30):
        draw = {
            k: (v.rvs(random_state=rng) if hasattr(v, "rvs") else rng.choice(v))
            for k, v in space.items()
        }
        assert 100 <= draw["n_estimators"] <= 600 and 8 <= draw["num_leaves"] <= 64
        assert 0.6 <= draw["subsample"] <= 1.0 and draw["reg_lambda"] > 0
    assert set(search.distributions("logreg")) == {"model__C"}


def test_clean_param_names_strips_the_prefix_and_numpy_types():
    cleaned = search.clean_param_names({"model__C": np.float64(2.5), "n_estimators": np.int64(7)})
    assert cleaned == {"C": 2.5, "n_estimators": 7}
    assert type(cleaned["C"]) is float and type(cleaned["n_estimators"]) is int


@pytest.mark.parametrize("family", search.FAMILIES)
def test_run_search_returns_scored_candidates(family, data):
    x, y = data
    result = search.run_search(family, make_params(), x, y, n_iter=2, cv_folds=2)
    assert len(result.candidates) == 2 and 0 < result.best_score < 1 and result.seconds > 0
    assert {c["rank"] for c in result.candidates} == {1, 2}
    assert all(0 < c["cv_pr_auc_mean"] < 1 for c in result.candidates)
    estimator = search.build_estimator(family, make_params()).set_params(**result.best_params)
    estimator.fit(x, y)
    assert estimator.predict_proba(x).shape == (len(y), 2)


def test_the_search_is_repeatable(data):
    x, y = data
    a = search.run_search("logreg", make_params(), x, y, n_iter=3, cv_folds=2)
    b = search.run_search("logreg", make_params(), x, y, n_iter=3, cv_folds=2)
    assert a.best_params == b.best_params and a.best_score == b.best_score


@pytest.mark.parametrize("family", search.FAMILIES)
def test_models_can_use_only_the_raw_features(family, data):
    x, y = data
    columns = tuple(RAW_FEATURES)
    model = search.build_estimator(family, make_params(), columns).fit(x[list(columns)], y)
    assert len(model.predict_proba(x[list(columns)])) == len(y)
    assert len(columns) < len(FEATURE_COLUMNS)


# ------------------------------------------------------------------ MLflow helpers
def test_retry_tries_three_times_then_gives_up(monkeypatch):
    monkeypatch.setattr(mu.time, "sleep", lambda s: pauses.append(s))
    pauses, calls = [], []

    @mu.retry(times=3, backoff=1.0)
    def flaky():
        calls.append(1)
        raise OSError("down")

    with pytest.raises(OSError, match="down"):
        flaky()
    assert len(calls) == 3 and pauses == [1.0, 2.0]


def test_retry_returns_as_soon_as_a_call_works(monkeypatch):
    monkeypatch.setattr(mu.time, "sleep", lambda s: None)
    outcomes = iter([OSError("x"), "fine"])

    @mu.retry(times=3)
    def once_flaky():
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    assert once_flaky() == "fine"


class Health(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self):  # noqa: N802 - name fixed by http.server
        self.send_response(Health.status)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *args):
        pass


@pytest.fixture
def health_server():
    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Health.status = 200
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_tracking_server_check(health_server, monkeypatch):
    monkeypatch.setattr(mu.time, "sleep", lambda s: None)
    mu.check_tracking_server(health_server)  # healthy: returns quietly
    mu.check_tracking_server("sqlite:///x.db")  # not HTTP: nothing to check
    Health.status = 500
    with pytest.raises(mu.TrackingError, match="not reachable"):
        mu.check_tracking_server(health_server)
    with pytest.raises(mu.TrackingError, match="not reachable"):
        mu.check_tracking_server("http://127.0.0.1:9")


def test_flatten_and_numeric_metrics():
    nested = {
        "a": {"b": 1, "c": {"d": 2.5}},
        "ok": True,
        "none": None,
        "nan": float("nan"),
        "text": "x",
        "n": 3,
    }
    assert mu.flatten(nested)["a.c.d"] == 2.5
    assert mu.numeric_metrics(nested, "p.") == {"p.a.b": 1.0, "p.a.c.d": 2.5, "p.n": 3.0}
    assert mu.safe_metric_name("recall@P>=0.5 (x)") == "recall_P__0.5 _x_"


def test_project_run_variables_are_cleared(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "7")
    assert mu.forget_project_run_env() == "abc"
    assert "MLFLOW_RUN_ID" not in os.environ and "MLFLOW_EXPERIMENT_ID" not in os.environ
    assert mu.forget_project_run_env() is None


def init_repo(root: Path) -> None:
    for command in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "t@t.t"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True)
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "params.yaml").write_text("seed: 1\n")
    (root / ".gitignore").write_text("reports/\n__pycache__/\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)


def test_dirty_detection_in_a_real_git_repo(tmp_path):
    init_repo(tmp_path)
    assert mu.dirty_pointers(tmp_path) == []
    info = mu.git_info(tmp_path)
    assert len(info["git_commit"]) == 40 and info["git_branch"] == "main"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text("{}")  # ignored output: not a pointer
    assert mu.dirty_pointers(tmp_path) == []
    (tmp_path / "params.yaml").write_text("seed: 2\n")
    (tmp_path / "src" / "new.py").write_text("y = 2\n")  # a NEW source file counts too
    assert sorted(mu.dirty_pointers(tmp_path)) == ["params.yaml", "src/new.py"]


def test_a_folder_that_is_not_a_git_repo_counts_as_dirty(tmp_path):
    assert mu.dirty_pointers(tmp_path) == ["<git status could not be read>"]
    assert mu.git_info(tmp_path)["git_commit"] == "unknown"


def test_data_pointer_fingerprint_changes_with_params(tmp_path):
    (tmp_path / "params.yaml").write_text("seed: 1\n")
    first = mu.data_pointer_md5(tmp_path)
    (tmp_path / "params.yaml").write_text("seed: 2\n")
    assert mu.data_pointer_md5(tmp_path) != first
    assert mu.file_md5(tmp_path / "nope") == "missing"


# ------------------------------------------------------------------ serving folder (no models)
def make_versions(root: Path, versions: list[int], current: int | None = None) -> None:
    for v in versions:
        (root / f"v{v}").mkdir(parents=True)
    if current is not None:
        export.switch_current(root, current)


def test_versions_are_listed_numerically_and_tmp_folders_are_ignored(tmp_path):
    make_versions(tmp_path, [2, 10, 9])
    (tmp_path / "v11.tmp").mkdir()
    (tmp_path / "notes").mkdir()
    assert export.list_versions(tmp_path) == [2, 9, 10]
    assert export.list_versions(tmp_path / "missing") == []


def test_switching_current_uses_a_relative_link_and_can_be_repeated(tmp_path):
    make_versions(tmp_path, [1, 2])
    assert export.current_version(tmp_path) is None
    export.switch_current(tmp_path, 1)
    assert os.readlink(tmp_path / "current") == "v1" and export.current_version(tmp_path) == 1
    export.switch_current(tmp_path, 2)
    assert export.current_version(tmp_path) == 2 and not (tmp_path / "current.tmp").exists()
    with pytest.raises(export.ExportError):
        export.switch_current(tmp_path, 9)
    assert export.current_version(tmp_path) == 2  # a failed switch changes nothing


def test_a_broken_current_link_is_reported_as_no_current(tmp_path):
    make_versions(tmp_path, [1], current=1)
    (tmp_path / "v1").rmdir()
    assert export.current_version(tmp_path) is None


def test_prune_keeps_the_newest_and_the_one_in_use(tmp_path):
    make_versions(tmp_path, [1, 2, 3, 4, 5], current=2)
    removed = export.prune(tmp_path, keep=3)
    assert removed == [1] and export.list_versions(tmp_path) == [2, 3, 4, 5]
    assert export.prune(tmp_path, keep=3) == []


def test_meta_is_empty_when_missing(tmp_path):
    assert export.read_meta(tmp_path) == {}
    (tmp_path / "meta.json").write_text(json.dumps({"run_id": "r"}))
    assert export.read_meta(tmp_path) == {"run_id": "r"}


# ------------------------------------------------------------------ configuration files
def load(name):
    return yaml.safe_load((REPO / name).read_text("utf-8"))


def test_every_dvc_stage_is_well_formed():
    dvc, params = load("dvc.yaml")["stages"], load("params.yaml")
    assert list(dvc) == [
        "ingest",
        "validate",
        "eda",
        "split",
        "synthesize",
        "featurize",
        "train",
        "evaluate",
        "gate",
    ]
    produced: set[str] = set()
    for name, stage in dvc.items():
        module = stage["cmd"].split()[-1]
        assert (REPO / "src" / (module.replace(".", "/") + ".py")).is_file(), name
        for dep in stage["deps"]:
            tracked_input = dep.startswith(("data/raw/", "data/feedback"))  # fetched by make data
            assert (REPO / dep).exists() or dep in produced or tracked_input, f"{name}: {dep}"
        for key in stage.get("params", []):
            assert key.split(".")[0] in params, f"{name}: params key {key}"
        for kind in ("outs", "metrics", "plots"):
            for out in stage.get(kind, []):
                produced.add(out if isinstance(out, str) else next(iter(out)))
    assert dvc["evaluate"]["always_changed"] is True
    assert "always_changed" not in dvc["train"]


def test_mlproject_entry_points_call_real_modules():
    project = load("MLproject")
    assert project["python_env"] == "python_env.yaml" and (REPO / "python_env.yaml").is_file()
    for name, entry in project["entry_points"].items():
        command = entry["command"]
        if "churnsense.pipeline" in command:
            module = command.split("-m ")[1].split()[0]
            assert (REPO / "src" / (module.replace(".", "/") + ".py")).is_file(), name


def test_mlflow_service_flags_match_what_was_verified():
    mlflow = load("compose.yaml")["services"]["mlflow"]
    flags = " ".join(mlflow["command"])
    assert "--allowed-hosts=" in flags and "mlflow:5000" in flags  # else HTTP 403 (checked)
    assert "--expose-prometheus" not in flags  # needs a package we do not install (checked)
    assert "--serve-artifacts" in flags and "--default-artifact-root=mlflow-artifacts:/" in flags
    assert mlflow["environment"]["USER"] and mlflow["ports"] == ["5000:5000"]
    # without this MLflow 3 runs 10 processes and ~1.75 GB; with it 2 processes and ~0.45 GB
    assert mlflow["environment"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"
    assert mlflow["mem_limit"] == "1g"
    trainer = load("compose.yaml")["services"]["trainer"]
    assert trainer["depends_on"]["mlflow"]["condition"] == "service_healthy"


def test_new_train_parameters_have_safe_defaults_in_params():
    train = load("params.yaml")["train"]
    assert train["search_real_only"] is True and train["final_fit_with_synthetic"] is True
    assert train["families"] == ["logreg", "lightgbm"] and train["n_iter"] == 20
