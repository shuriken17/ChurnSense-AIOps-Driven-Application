"""Integration tests: evaluate, the gate on real output, export and promotion (T-DP-01, T-DP-02)."""

from __future__ import annotations

import copy
import json
import os
import shutil

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from churnsense import mlflow_utils as mu
from churnsense.deploy import export, promote
from churnsense.model.pyfunc import model_inputs
from churnsense.pipeline import evaluate, gate

pytestmark = pytest.mark.integration


def read(paths, name):
    return json.loads(paths.report(name).read_text("utf-8"))


# ------------------------------------------------------------------ evaluate
def test_evaluation_report_structure_and_numbers(trained):
    ev = read(trained.paths, "eval_metrics.json")
    assert ev["status"] == "ok" and ev["champion"] is None and ev["golden_rows"] == 75
    golden = ev["challenger"]["golden"]
    assert golden["n"] == 75 and 0.5 < golden["roc_auc"] <= 1
    assert golden["tp"] + golden["fn"] == golden["n_positive"]
    assert golden["tn"] + golden["fp"] + golden["fn"] + golden["tp"] == 75
    assert set(golden["by_contract"]) == {"Month-to-month", "One year", "Two year"}
    assert ev["challenger"]["recent"] is None  # no feedback exists yet
    assert ev["model_run_id"] == ev["challenger"]["run_id"]


def test_sanity_facts_are_all_good_for_a_healthy_model(trained):
    sanity = read(trained.paths, "eval_metrics.json")["sanity"]
    for key in (
        "loads_in_fresh_process",
        "no_nan_predictions",
        "probabilities_in_range",
        "bands_valid",
        "top_factors_valid",
        "shap_additive",
        "dependencies_match",
    ):
        assert sanity[key] is True, key
    assert sanity["shap_additivity_max_error"] < 1e-9
    assert 0 < sanity["latency_ms_p50"] <= sanity["latency_ms_p95"] < 1000
    assert sanity["dependency_mismatches"] == [] and sanity["git_dirty"] is False


def test_the_smoke_rows_are_real_inputs_with_the_model_answers(trained):
    smoke = read(trained.paths, "smoke_rows.json")
    assert len(smoke["rows"]) == 20
    first = smoke["rows"][0]
    assert set(first["input"]) == {
        c.name
        for c in mlflow.pyfunc.load_model(
            str(trained.paths.candidate)
        ).metadata.signature.inputs.inputs
    }
    assert set(first["expected"]) == {"churn_probability", "risk_band"}


def test_with_no_champion_the_comparison_is_empty_and_the_gate_has_a_first_model_decision(trained):
    assert read(trained.paths, "comparison.json") == {"champion": False}
    decision = read(trained.paths, "gate_decision.json")
    assert decision["status"] == "ok" and decision["first_model"] is True
    assert decision["decision"] in ("PROMOTE", "REJECT")
    recomputed = gate.decide(
        read(trained.paths, "eval_metrics.json"),
        {"champion": False},
        trained.params.gate,
        read(trained.paths, "gate_decision.json")["candidate_run_id"],
    )
    assert recomputed["decision"] == decision["decision"]  # the stage applies exactly decide()


def test_the_evaluation_was_added_to_the_training_run(trained):
    info = json.loads(trained.paths.run_info.read_text("utf-8"))
    run = MlflowClient(tracking_uri=trained.uri).get_run(info["run_id"])
    assert run.data.tags["evaluated"] == "true"
    assert "golden_roc_auc" in run.data.metrics and "sanity_latency_ms_p95" in run.data.metrics
    names = {
        a.path
        for a in MlflowClient(tracking_uri=trained.uri).list_artifacts(info["run_id"], "evaluation")
    }
    assert {
        "evaluation/eval_metrics.json",
        "evaluation/comparison.json",
        "evaluation/smoke_rows.json",
    } <= names


def test_a_model_that_cannot_be_loaded_stops_the_evaluation(work):
    (work.paths.candidate / "artifacts" / "bundle.joblib").unlink()
    with pytest.raises(evaluate.StageError, match="fresh process"):
        evaluate.run(work.params, work.paths)


def test_scoring_errors_are_clear(work):
    raw = model_inputs(pd.read_parquet(work.paths.split_file("test_golden")).head(3))
    with pytest.raises(evaluate.ScoringError, match="failed"):
        evaluate.score_in_fresh_process(work.root, work.root / "no-such-model", raw)
    (work.root / "scripts" / "score_model.py").unlink()
    with pytest.raises(evaluate.ScoringError, match="missing"):
        evaluate.score_in_fresh_process(work.root, work.paths.candidate, raw)


def test_a_library_version_mismatch_is_detected(work):
    requirements = work.paths.candidate / "requirements.txt"
    text = requirements.read_text("utf-8").replace("lightgbm==", "lightgbm==0.0.1+")
    requirements.write_text(text, "utf-8")
    problems = evaluate.dependency_mismatches(work.paths.candidate)
    assert len(problems) == 1 and problems[0].startswith("lightgbm: model 0.0.1+")
    requirements.unlink()
    assert evaluate.dependency_mismatches(work.paths.candidate) == [
        "requirements.txt is missing from the model"
    ]


def test_the_scoring_process_cannot_see_the_repository_code(work):
    """The exported model must run from its own folder: PYTHONPATH is emptied for the process."""
    os.environ["PYTHONPATH"] = str(work.root / "does-not-exist")
    try:
        raw = model_inputs(pd.read_parquet(work.paths.split_file("test_golden")).head(5))
        predictions, facts = evaluate.score_in_fresh_process(work.root, work.paths.candidate, raw)
    finally:
        del os.environ["PYTHONPATH"]
    assert len(predictions) == 5 and facts["family"] in ("logreg", "lightgbm")


# ------------------------------------------------------------------ export (T-DP-01, T-DP-02)
def do_export(work, version, **overrides):
    meta = {"run_id": json.loads(work.paths.run_info.read_text("utf-8"))["run_id"]}
    return export.export_model(
        work.root,
        overrides.get("source", work.paths.candidate),
        work.serving,
        version,
        work.paths.report("smoke_rows.json"),
        meta,
    )


def test_t_dp_01_export_verifies_copies_and_never_leaves_a_tmp_folder(work):
    final = do_export(work, 1)
    assert final == work.serving / "v1" and not (work.serving / "v1.tmp").exists()
    meta = export.read_meta(final)
    assert meta["version"] == 1 and meta["verification"]["smoke_rows"] == 20
    assert meta["verification"]["largest_probability_gap"] < 1e-6
    assert (final / "MLmodel").is_file() and (final / "code" / "churnsense").is_dir()
    export.switch_current(work.serving, 1)
    assert export.current_version(work.serving) == 1
    do_export(work, 2)
    export.switch_current(work.serving, 2)
    assert export.list_versions(work.serving) == [1, 2]
    assert export.current_version(work.serving) == 2


def test_t_dp_01_a_version_is_never_overwritten(work):
    do_export(work, 1)
    with pytest.raises(export.ExportError, match="already exists"):
        do_export(work, 1)


def test_t_dp_02_a_model_that_changes_its_answers_is_not_exported(work):
    do_export(work, 1)
    export.switch_current(work.serving, 1)
    smoke = read(work.paths, "smoke_rows.json")
    smoke["rows"][3]["expected"]["churn_probability"] += 0.01  # as if the export changed a result
    work.paths.report("smoke_rows.json").write_text(json.dumps(smoke), "utf-8")
    with pytest.raises(export.ExportError, match="changed its answers"):
        do_export(work, 2)
    assert export.list_versions(work.serving) == [1]  # nothing half-exported
    assert not (work.serving / "v2.tmp").exists()
    assert export.current_version(work.serving) == 1  # serving is untouched


def test_t_dp_02_a_broken_model_is_not_exported(work):
    broken = work.root / "broken-model"
    shutil.copytree(work.paths.candidate, broken)
    (broken / "artifacts" / "bundle.joblib").write_bytes(b"not a model")
    with pytest.raises(export.ExportError, match="does not run"):
        do_export(work, 1, source=broken)
    assert export.list_versions(work.serving) == [] and not (work.serving / "v1.tmp").exists()


# ------------------------------------------------------------------ promotion
def write_decision(work, *, decision="PROMOTE", champion_run_id=None, candidate=None):
    info = json.loads(work.paths.run_info.read_text("utf-8"))
    payload = {
        "decision": decision,
        "reasons": [] if decision == "PROMOTE" else ["because"],
        "candidate_run_id": candidate or info["run_id"],
        "champion_run_id": champion_run_id,
        "first_model": champion_run_id is None,
        "checks": [],
    }
    work.paths.report("gate_decision.json").write_text(json.dumps(payload), "utf-8")
    return info


def test_first_promotion_registers_exports_switches_and_sets_the_alias_last(work):
    info = write_decision(work)
    summary = promote.promote(work.paths)
    assert summary["version"] == 1 and summary["steps"]["switched"] == "-> v1"
    assert "skipped" in summary["steps"]["live_checks"]
    client = MlflowClient(tracking_uri=work.uri)
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "champion").version == 1
    version = client.get_model_version(mu.REGISTERED_MODEL, "1")
    assert version.tags["run_id"] == info["run_id"] and version.tags["gate_decision"] == "PROMOTE"
    assert export.current_version(work.serving) == 1
    assert export.read_meta(work.serving / "v1")["run_id"] == info["run_id"]


def test_second_promotion_moves_the_champion_alias_and_keeps_the_previous_one(work):
    info = write_decision(work)
    promote.promote(work.paths)
    write_decision(work, champion_run_id=info["run_id"])  # the evaluation compared with v1
    summary = promote.promote(work.paths)
    assert summary["version"] == 2 and summary["steps"]["switched"] == "v1 -> v2"
    client = MlflowClient(tracking_uri=work.uri)
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "champion").version == 2
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "previous").version == 1
    assert export.current_version(work.serving) == 2


def test_a_reject_decision_is_never_promoted(work):
    write_decision(work, decision="REJECT")
    with pytest.raises(promote.PromotionRefusedError, match="REJECT"):
        promote.promote(work.paths)
    assert export.list_versions(work.serving) == []


def test_a_decision_about_another_model_is_refused(work):
    write_decision(work, candidate="e" * 32)
    with pytest.raises(promote.PromotionRefusedError, match="different model"):
        promote.promote(work.paths)


def test_a_champion_that_changed_since_the_evaluation_blocks_the_promotion(work):
    write_decision(work)
    promote.promote(work.paths)  # somebody promoted in the meantime
    write_decision(work, champion_run_id=None)  # but this decision still thinks there is none
    with pytest.raises(promote.PromotionRefusedError, match="champion changed"):
        promote.promote(work.paths)
    assert export.current_version(work.serving) == 1


def test_missing_inputs_are_a_clear_refusal(work):
    work.paths.report("gate_decision.json").unlink(missing_ok=True)
    with pytest.raises(promote.PromotionRefusedError, match="gate_decision.json is missing"):
        promote.promote(work.paths)


def test_a_failed_verification_leaves_current_and_the_alias_where_they_were(work):
    info = write_decision(work)
    promote.promote(work.paths)
    write_decision(work, champion_run_id=info["run_id"])
    smoke = read(work.paths, "smoke_rows.json")
    smoke["rows"][0]["expected"]["churn_probability"] = 0.123456  # cannot be reproduced
    work.paths.report("smoke_rows.json").write_text(json.dumps(smoke), "utf-8")
    with pytest.raises(export.ExportError):
        promote.promote(work.paths)
    client = MlflowClient(tracking_uri=work.uri)
    assert export.current_version(work.serving) == 1
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "champion").version == 1
    assert export.list_versions(work.serving) == [1]


def test_main_uses_distinct_exit_codes_and_writes_a_promotion_report(work, monkeypatch, capsys):
    monkeypatch.setattr(promote, "default_paths", lambda: work.paths)
    write_decision(work, decision="REJECT")
    with pytest.raises(SystemExit) as refused:
        promote.main()
    assert refused.value.code == 3 and read(work.paths, "promotion.json")["status"] == "refused"
    write_decision(work)
    promote.main()
    assert read(work.paths, "promotion.json")["status"] == "promoted"
    assert '"version": 1' in capsys.readouterr().out
    monkeypatch.setattr(promote, "promote", lambda paths: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(SystemExit) as failed:
        promote.main()
    assert failed.value.code == 1 and read(work.paths, "promotion.json")["status"] == "failed"


# ------------------------------------------------------------------ champion in place
def test_with_a_champion_the_evaluation_compares_and_the_gate_waits_for_recent_data(work):
    do_export(work, 1)
    export.switch_current(work.serving, 1)
    evaluate.run(work.params, work.paths)
    gate.run(work.params, work.paths)
    ev, comparison = read(work.paths, "eval_metrics.json"), read(work.paths, "comparison.json")
    assert ev["champion"]["version_dir"] == "v1" and ev["champion"]["golden"]["n"] == 75
    assert comparison["champion"] is True and "recent" not in comparison
    # the same model against itself: no difference at all
    for key in ("d_auc_mean", "d_auc_p05", "d_recall_p05"):
        assert comparison["golden"][key] == 0.0
    decision = read(work.paths, "gate_decision.json")
    assert decision["decision"] == "REJECT" and decision["first_model"] is False
    rules = [r.split(":")[0] for r in decision["reasons"]]
    # (the tiny fixture model may also miss the floors; the champion rule must be listed anyway)
    assert "recent_data_sufficient" in rules
    assert "golden_auc_noninferior" not in rules  # the same model is not worse than itself
    assert decision["champion_run_id"] == ev["challenger"]["run_id"]


def test_a_champion_that_cannot_be_scored_makes_the_gate_refuse(work):
    do_export(work, 1)
    export.switch_current(work.serving, 1)
    (work.serving / "v1" / "artifacts" / "bundle.joblib").write_bytes(b"corrupt")
    evaluate.run(work.params, work.paths)
    gate.run(work.params, work.paths)
    decision = read(work.paths, "gate_decision.json")
    assert decision["decision"] == "REJECT"
    assert any(r.startswith("champion_scored") for r in decision["reasons"])


def test_the_gate_inputs_are_not_modified_by_the_stage(work):
    before = copy.deepcopy(read(work.paths, "eval_metrics.json"))
    gate.run(work.params, work.paths)
    assert read(work.paths, "eval_metrics.json") == before


# ------------------------------------------------------------------ live checks and rollback
class LiveServices:
    """Stands in for the model server and the backend; records what promote asks of them."""

    def __init__(self, monkeypatch, work, fail_smoke_for=(), down=False):
        self.calls, self.fail_for, self.down = [], set(fail_smoke_for), down
        monkeypatch.setenv("MODEL_SERVER_URL", "http://model-server:8001")
        monkeypatch.setenv("ADMIN_TOKEN", "t")
        monkeypatch.setattr(promote.smoke, "reload_model_server", self.reload)
        monkeypatch.setattr(promote.smoke, "smoke_test", self.smoke_test)
        self.work = work

    def reload(self, url, token, expected_run_id):
        self.calls.append(("reload", export.current_version(self.work.serving)))
        if self.down:
            raise promote.smoke.ServerNotReachableError("[Errno -2] Name or service not known")
        return {"run_id": expected_run_id}

    def smoke_test(self, backend, smoke_file, run_id):
        self.calls.append(("smoke", export.current_version(self.work.serving)))
        if export.current_version(self.work.serving) in self.fail_for:
            raise promote.smoke.DeploymentError("smoke test failed: row 0: band HIGH")
        return {"rows": 20, "all_match": True}


def test_live_checks_run_after_the_switch_and_before_the_alias(work, monkeypatch):
    live = LiveServices(monkeypatch, work)
    write_decision(work)
    summary = promote.promote(work.paths)
    assert live.calls == [("reload", 1), ("smoke", 1)]  # both saw the NEW current already
    assert (
        summary["steps"]["live_checks"].startswith("ok")
        and "champion" in summary["steps"]["aliases"]
    )


def test_a_failed_smoke_test_puts_the_previous_model_back_and_keeps_the_alias(work, monkeypatch):
    info = write_decision(work)
    promote.promote(work.paths)  # v1 is the champion
    live = LiveServices(monkeypatch, work, fail_smoke_for={2})
    write_decision(work, champion_run_id=info["run_id"])
    with pytest.raises(promote.smoke.DeploymentError, match="rolled back to v1"):
        promote.promote(work.paths)
    assert export.current_version(work.serving) == 1
    assert live.calls == [("reload", 2), ("smoke", 2), ("reload", 1)]  # the old model was reloaded
    client = MlflowClient(tracking_uri=work.uri)
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "champion").version == 1


def test_a_first_model_that_fails_live_leaves_no_current_and_no_alias(work, monkeypatch):
    LiveServices(monkeypatch, work, fail_smoke_for={1})
    write_decision(work)
    with pytest.raises(promote.smoke.DeploymentError, match="current was removed"):
        promote.promote(work.paths)
    assert export.current_version(work.serving) is None
    with pytest.raises(Exception, match="not found|champion"):
        MlflowClient(tracking_uri=work.uri).get_model_version_by_alias(
            mu.REGISTERED_MODEL, "champion"
        )


def test_main_reports_a_failed_live_check_with_exit_code_1(work, monkeypatch, capsys):
    LiveServices(monkeypatch, work, fail_smoke_for={1})
    monkeypatch.setattr(promote, "default_paths", lambda: work.paths)
    write_decision(work)
    with pytest.raises(SystemExit) as failed:
        promote.main()
    assert failed.value.code == 1 and "smoke test failed" in capsys.readouterr().out


def test_a_first_model_is_kept_when_the_model_server_is_not_running_yet(work, monkeypatch):
    """New machine: the server needs a model to start, so it is not running before the first."""
    live = LiveServices(monkeypatch, work, down=True)
    write_decision(work)
    summary = promote.promote(work.paths)
    assert summary["steps"]["live_checks"].startswith(
        "skipped: the model server is not running yet"
    )
    assert export.current_version(work.serving) == 1 and live.calls == [("reload", 1)]
    client = MlflowClient(tracking_uri=work.uri)
    assert client.get_model_version_by_alias(mu.REGISTERED_MODEL, "champion").version == 1


def test_a_later_model_is_still_rolled_back_when_the_server_cannot_be_reached(work, monkeypatch):
    info = write_decision(work)
    promote.promote(work.paths)  # v1 is the champion
    LiveServices(monkeypatch, work, down=True)
    write_decision(work, champion_run_id=info["run_id"])
    with pytest.raises(promote.smoke.DeploymentError, match="rolled back to v1"):
        promote.promote(work.paths)
    assert export.current_version(work.serving) == 1  # the first-model exception does not apply


def test_only_an_unreachable_server_counts_as_not_running_yet(monkeypatch):
    import httpx

    def raising(error):
        def post(*args, **kwargs):
            raise error

        return post

    smoke = promote.smoke
    monkeypatch.setattr(
        smoke.httpx, "post", raising(httpx.ConnectError("Name or service not known"))
    )
    with pytest.raises(smoke.ServerNotReachableError):
        smoke.reload_model_server("http://x", "t", "run")
    monkeypatch.setattr(smoke.httpx, "post", raising(httpx.ReadTimeout("too slow")))
    with pytest.raises(smoke.DeploymentError) as slow:
        smoke.reload_model_server("http://x", "t", "run")
    assert not isinstance(slow.value, smoke.ServerNotReachableError)
