"""Integration tests: the train stage and what it records (T-TR-01, T-TR-02)."""

from __future__ import annotations

import json

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from churnsense import mlflow_utils as mu
from churnsense.model.pyfunc import model_inputs
from churnsense.pipeline import train
from churnsense.schema import INPUT_FIELDS
from churnsense.tracking import StageError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report(trained):
    return json.loads(trained.paths.report("train_metrics.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def run_info(trained):
    return json.loads(trained.paths.run_info.read_text("utf-8"))


def test_t_tr_01_the_stage_ran_and_chose_the_family_with_the_best_validation_pr_auc(report):
    assert report["status"] == "ok" and report["rows_in"] > 6000
    by_family = report["validation_by_family"]
    assert set(by_family) == {"logreg", "lightgbm"}
    best = max(by_family, key=lambda f: by_family[f]["pr_auc"])
    assert report["winner"] == best
    assert report["n_candidates_per_family"] == 2 and report["search_rows"] == 350
    assert report["final_fit_rows"] > 6000  # the final fit includes the synthetic rows
    thresholds = report["thresholds"]
    assert 0 < thresholds["t_medium"] <= thresholds["t_high"] < 1
    assert sum(report["band_shares_validation"].values()) == pytest.approx(1.0)


def test_t_tr_01_run_info_points_to_the_run_and_the_commit(run_info, trained):
    assert len(run_info["run_id"]) == 32 and run_info["model_uri"].startswith("models:/m-")
    assert run_info["git_dirty"] is False and len(run_info["git_commit"]) == 40
    assert run_info["experiment"] == mu.EXPERIMENT


def test_t_tr_01_the_run_carries_tags_params_metrics_and_artifacts(run_info, trained):
    client = MlflowClient(tracking_uri=trained.uri)
    run = client.get_run(run_info["run_id"])
    tags = run.data.tags
    for key in (
        "git_commit",
        "git_branch",
        "git_dirty",
        "dvc_lock_md5",
        "data_pointer_md5",
        "pipeline_trigger",
        "model_family",
        "ray_version",
        "lightgbm_version",
        "host",
        "drift_alert_fingerprint",
        "controller_run_id",
    ):
        assert key in tags, key
    assert tags["git_dirty"] == "false" and tags["pipeline_trigger"] == "manual"
    assert tags["model_family"] == run_info["family"]
    assert run.data.params["seed"] == "42" and "t_high" in run.data.params
    assert any(k.startswith("best.") for k in run.data.params)
    metrics = run.data.metrics
    for key in (
        "val_roc_auc",
        "val_pr_auc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_accuracy",
        "val_brier",
        "val_recall_at_min_precision",
        "model_size_mb",
        "time_train_seconds",
        "time_search_logreg_seconds",
        "time_fit_lightgbm_seconds",
        "val_band_share_HIGH",
    ):
        assert key in metrics, key
    assert any(k.startswith("data.ingest_stats.") for k in metrics)
    assert any(k.startswith("val_contract_") for k in metrics)
    artifacts = {a.path for a in client.list_artifacts(run_info["run_id"])}
    assert {"plots", "reports"} <= artifacts
    plots = {a.path for a in client.list_artifacts(run_info["run_id"], "plots")}
    assert {
        "plots/roc_curve.png",
        "plots/pr_curve.png",
        "plots/calibration.png",
        "plots/confusion_matrix.png",
        "plots/feature_importance.csv",
    } <= plots


def test_t_tr_01_every_candidate_is_a_nested_run(run_info, trained):
    client = MlflowClient(tracking_uri=trained.uri)
    experiment = client.get_experiment_by_name(mu.EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id], max_results=100)
    parent = run_info["run_id"]
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent]
    names = sorted(r.info.run_name for r in children)
    assert names == ["fit-lightgbm", "fit-logreg", "search-lightgbm", "search-logreg"]
    candidates = [r for r in runs if r.info.run_name.startswith("candidate-")]
    assert len(candidates) == 4  # 2 families x 2 candidates
    assert all("cv_pr_auc_mean" in r.data.metrics for r in candidates)
    assert len(runs) == 1 + 4 + 4


def test_ablations_are_logged_in_their_own_experiment(run_info, report, trained):
    client = MlflowClient(tracking_uri=trained.uri)
    experiment = client.get_experiment_by_name(mu.ABLATION_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert sorted(r.data.tags["ablation"] for r in runs) == [
        "engineered_features_off",
        "synthetic_rows_off",
    ]
    assert all(r.data.tags["parent_run_id"] == run_info["run_id"] for r in runs)
    assert all("delta_pr_auc_vs_main" in r.data.metrics for r in runs)
    assert set(report["ablations"]) == {"synthetic_rows_off", "engineered_features_off"}
    assert report["ablations"]["engineered_features_off"]["n_features"] < 26


def test_t_tr_01_the_packaged_model_loads_and_answers_in_the_documented_format(trained, run_info):
    model = mlflow.pyfunc.load_model(str(trained.paths.candidate))
    names = [c.name for c in model.metadata.signature.inputs.inputs]
    assert names == [f.name for f in INPUT_FIELDS]
    split = pd.read_parquet(trained.paths.split_file("test_golden")).head(10)
    out = model.predict(model_inputs(split))
    assert list(out.columns) == ["churn_probability", "risk_band", "top_factors", "model_run_id"]
    assert out["churn_probability"].between(0, 1).all()
    assert (out["model_run_id"] == run_info["run_id"]).all()
    code = trained.paths.candidate / "code" / "churnsense"
    assert (code / "features" / "derive.py").is_file()  # self-contained: carries its own code
    assert (trained.paths.candidate / "requirements.txt").is_file()


def test_the_registered_logged_model_exists_in_mlflow(run_info, trained):
    client = MlflowClient(tracking_uri=trained.uri)
    assert client.get_logged_model(run_info["model_id"]).model_id == run_info["model_id"]


def test_importance_tables_and_the_pr_curve_file_are_written(trained):
    plots = trained.paths.plots_dir
    table = pd.read_csv(plots / "feature_importance.csv")
    assert len(table) == 26 and table["mean_abs_shap"].is_monotonic_decreasing
    assert {"permutation_importance_mean", "permutation_importance_std"} <= set(table.columns)
    curve = pd.read_csv(plots / "pr_curve_val.csv")
    assert list(curve.columns) == ["recall", "precision", "threshold"] and len(curve) > 10


# ------------------------------------------------------------------ T-TR-02 guards
def test_t_tr_02_uncommitted_code_is_refused(work):
    (work.root / "params.yaml").write_text("seed: 43\n")  # a change nobody committed
    with pytest.raises(StageError, match="uncommitted changes in params.yaml"):
        train.run(work.params, work.paths)


def test_t_tr_02_allow_dirty_trains_but_marks_the_run_and_the_gate_refuses_it(work, monkeypatch):
    from churnsense.pipeline import evaluate, gate

    (work.root / "params.yaml").write_text("seed: 43\n")
    monkeypatch.setenv("ALLOW_DIRTY", "1")
    work.params.train["families"] = ["logreg"]
    work.params.train["n_iter"] = 1
    work.params.train["ablations"] = False
    train.run(work.params, work.paths)
    info = json.loads(work.paths.run_info.read_text("utf-8"))
    assert info["git_dirty"] is True
    run = MlflowClient(tracking_uri=work.uri).get_run(info["run_id"])
    assert run.data.tags["git_dirty"] == "true"
    evaluate.run(work.params, work.paths)
    gate.run(work.params, work.paths)
    decision = json.loads(work.paths.report("gate_decision.json").read_text("utf-8"))
    assert decision["decision"] == "REJECT"
    assert any(r.startswith("trained_on_committed_code") for r in decision["reasons"])


def test_an_unreachable_tracking_server_stops_the_stage_before_any_work(work, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setattr(mu.time, "sleep", lambda seconds: None)
    before = work.paths.run_info.read_text("utf-8")
    with pytest.raises(StageError, match="not reachable"):
        train.run(work.params, work.paths)
    assert work.paths.run_info.read_text("utf-8") == before  # nothing was overwritten


def test_the_command_line_flags_restrict_the_families_and_candidates(monkeypatch):
    seen = {}
    monkeypatch.setattr(train.run, "main", lambda configure: seen.update(configure=configure))
    monkeypatch.setattr("sys.argv", ["train", "--family", "logreg", "--n-iter", "5"])
    train.cli()
    from tests.conftest import make_params

    params = make_params()
    seen["configure"](params)
    assert params.train["families"] == ["logreg"] and params.train["n_iter"] == 5
