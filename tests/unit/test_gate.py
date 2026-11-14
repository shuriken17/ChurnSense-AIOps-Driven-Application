"""Unit tests for the promotion gate (T-GA-01, T-GA-02, T-GA-03)."""

from __future__ import annotations

import copy
import json

import pytest

from churnsense.pipeline import gate
from tests.conftest import make_params

pytestmark = pytest.mark.unit

RUN = "a" * 32
CFG = make_params().gate


def good_evaluation() -> dict:
    """An evaluation that passes every rule (no champion)."""
    return {
        "model_run_id": RUN,
        "challenger": {"golden": {"roc_auc": 0.86, "recall_at_min_precision": 0.85}},
        "champion": None,
        "sanity": {
            "loads_in_fresh_process": True,
            "no_nan_predictions": True,
            "probabilities_in_range": True,
            "bands_valid": True,
            "top_factors_valid": True,
            "shap_additive": True,
            "dependencies_match": True,
            "dependency_mismatches": [],
            "git_dirty": False,
            "latency_ms_p95": 40.0,
            "model_size_mb": 0.3,
            "golden_positive_rate": 0.40,
        },
        "recent_rows": 0,
        "recent_positives": 0,
    }


def with_champion(
    recent_rows=400, recent_positives=90, d_auc_p05=0.0, d_recall_p05=0.0, d_auc_p10=0.02
):
    evaluation = good_evaluation()
    evaluation["champion"] = {"run_id": "c" * 32, "family": "logreg"}
    evaluation["recent_rows"], evaluation["recent_positives"] = recent_rows, recent_positives
    comparison = {
        "champion": True,
        "golden": {"d_auc_p05": d_auc_p05, "d_recall_p05": d_recall_p05},
        "recent": {"d_auc_p10": d_auc_p10},
    }
    return evaluation, comparison


def decide(evaluation, comparison=None):
    return gate.decide(evaluation, comparison or {"champion": False}, CFG, RUN)


def failed_rules(decision):
    return {c["rule"] for c in decision["checks"] if not c["passed"]}


def test_a_good_first_model_is_promoted_with_every_check_recorded():
    decision = decide(good_evaluation())
    assert decision["decision"] == "PROMOTE" and decision["reasons"] == []
    assert decision["first_model"] is True and decision["champion_run_id"] is None
    assert len(decision["checks"]) >= 14 and all(c["passed"] for c in decision["checks"])


def mutate(path, value):
    def apply(evaluation):
        node = evaluation
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return evaluation

    return apply


REJECTIONS = {
    "does not load": (
        mutate(("sanity", "loads_in_fresh_process"), False),
        "loads_in_fresh_process",
    ),
    "NaN predictions": (mutate(("sanity", "no_nan_predictions"), False), "no_nan_predictions"),
    "probability out of range": (
        mutate(("sanity", "probabilities_in_range"), False),
        "probabilities_in_range",
    ),
    "invalid band": (mutate(("sanity", "bands_valid"), False), "bands_valid"),
    "bad top factors": (mutate(("sanity", "top_factors_valid"), False), "top_factors_valid"),
    "SHAP does not add up": (mutate(("sanity", "shap_additive"), False), "shap_additive"),
    "library mismatch": (mutate(("sanity", "dependencies_match"), False), "dependencies_match"),
    "trained on uncommitted code": (
        mutate(("sanity", "git_dirty"), True),
        "trained_on_committed_code",
    ),
    "too slow": (mutate(("sanity", "latency_ms_p95"), 200.1), "latency"),
    "too big": (mutate(("sanity", "model_size_mb"), 50.1), "model_size"),
    "flags almost nobody": (
        mutate(("sanity", "golden_positive_rate"), 0.049),
        "golden_positive_rate",
    ),
    "flags almost everybody": (
        mutate(("sanity", "golden_positive_rate"), 0.601),
        "golden_positive_rate",
    ),
    "AUC below the floor": (
        mutate(("challenger", "golden", "roc_auc"), 0.7999),
        "golden_roc_auc_floor",
    ),
    "recall below the floor": (
        mutate(("challenger", "golden", "recall_at_min_precision"), 0.6999),
        "golden_recall_floor",
    ),
    "evaluation of another model": (
        mutate(("model_run_id",), "b" * 32),
        "evaluation_matches_candidate",
    ),
    "AUC is NaN": (
        mutate(("challenger", "golden", "roc_auc"), float("nan")),
        "golden_roc_auc_floor",
    ),
    "recall is text": (
        mutate(("challenger", "golden", "recall_at_min_precision"), "n/a"),
        "golden_recall_floor",
    ),
    "latency missing": (mutate(("sanity", "latency_ms_p95"), None), "latency"),
}


@pytest.mark.parametrize(("change", "rule"), REJECTIONS.values(), ids=REJECTIONS.keys())
def test_t_ga_01_each_rule_alone_rejects(change, rule):
    decision = decide(change(good_evaluation()))
    assert decision["decision"] == "REJECT"
    assert failed_rules(decision) == {rule}  # only that rule failed, and it names itself
    assert any(rule in reason for reason in decision["reasons"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("challenger", "golden", "roc_auc"), 0.80),
        (("challenger", "golden", "recall_at_min_precision"), 0.70),
        (("sanity", "latency_ms_p95"), 200.0),
        (("sanity", "model_size_mb"), 50.0),
        (("sanity", "golden_positive_rate"), 0.05),
        (("sanity", "golden_positive_rate"), 0.60),
    ],
)
def test_limits_are_inclusive(path, value):
    assert decide(mutate(path, value)(good_evaluation()))["decision"] == "PROMOTE"


def test_every_failure_is_listed_not_just_the_first():
    evaluation = good_evaluation()
    evaluation["challenger"]["golden"]["roc_auc"] = 0.5
    evaluation["sanity"]["git_dirty"] = True
    evaluation["sanity"]["latency_ms_p95"] = 900
    assert failed_rules(decide(evaluation)) == {
        "golden_roc_auc_floor",
        "trained_on_committed_code",
        "latency",
    }


# ------------------------------------------------------------------ against the champion
def test_t_ga_02_better_on_recent_data_and_not_worse_on_golden_is_promoted():
    decision = decide(*with_champion())
    assert decision["decision"] == "PROMOTE" and decision["first_model"] is False
    assert decision["champion_run_id"] == "c" * 32


@pytest.mark.parametrize(
    ("kwargs", "rule"),
    [
        ({"d_auc_p05": -0.0101}, "golden_auc_noninferior"),
        ({"d_recall_p05": -0.0201}, "golden_recall_noninferior"),
        ({"recent_rows": 299}, "recent_data_sufficient"),
        ({"recent_positives": 49}, "recent_data_sufficient"),
        ({"d_auc_p10": 0.0}, "recent_auc_improvement"),
        ({"d_auc_p10": -0.01}, "recent_auc_improvement"),
    ],
)
def test_t_ga_02_champion_rules_reject(kwargs, rule):
    decision = decide(*with_champion(**kwargs))
    assert decision["decision"] == "REJECT" and rule in failed_rules(decision)


def test_t_ga_02_allowed_losses_are_inclusive():
    assert decide(*with_champion(d_auc_p05=-0.01, d_recall_p05=-0.02))["decision"] == "PROMOTE"


def test_without_enough_recent_data_the_champion_stays():
    evaluation, comparison = with_champion(recent_rows=0, recent_positives=0)
    comparison.pop("recent")
    decision = decide(evaluation, comparison)
    assert decision["decision"] == "REJECT"
    assert failed_rules(decision) == {"recent_data_sufficient"}


def test_a_champion_that_could_not_be_scored_blocks_promotion():
    evaluation = good_evaluation()
    evaluation["champion"] = {"error": "scoring v3 failed"}
    decision = decide(evaluation, {"champion": True, "error": "x"})
    assert decision["decision"] == "REJECT" and "champion_scored" in failed_rules(decision)


# ------------------------------------------------------------------ the stage fails closed
def write(paths, evaluation, comparison, run_id=RUN):
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.report("eval_metrics.json").write_text(json.dumps(evaluation), "utf-8")
    paths.report("comparison.json").write_text(json.dumps(comparison), "utf-8")
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.run_info.write_text(json.dumps({"run_id": run_id}), "utf-8")


def test_t_ga_03_missing_input_files_give_reject_not_a_crash(workspace, params):
    report = gate.run(params, workspace)  # no evaluation was ever written
    assert report is not None
    saved = json.loads(workspace.report("gate_decision.json").read_text("utf-8"))
    assert saved["status"] == "ok" and saved["decision"] == "REJECT"
    assert saved["reasons"][0].startswith("gate_error:")


def test_t_ga_03_a_malformed_evaluation_gives_reject(workspace, params):
    write(workspace, {"challenger": {}}, {})
    gate.run(params, workspace)
    saved = json.loads(workspace.report("gate_decision.json").read_text("utf-8"))
    assert saved["decision"] == "REJECT" and "gate_error" in saved["reasons"][0]


def test_the_stage_writes_the_decision_of_a_good_evaluation(workspace, params):
    write(workspace, good_evaluation(), {"champion": False})
    gate.run(params, workspace)
    saved = json.loads(workspace.report("gate_decision.json").read_text("utf-8"))
    assert saved["decision"] == "PROMOTE" and saved["candidate_run_id"] == RUN


def test_a_stale_evaluation_is_rejected(workspace, params):
    write(workspace, good_evaluation(), {"champion": False}, run_id="f" * 32)
    gate.run(params, workspace)
    saved = json.loads(workspace.report("gate_decision.json").read_text("utf-8"))
    assert saved["decision"] == "REJECT" and "evaluation_matches_candidate" in saved["reasons"][0]


def test_decide_does_not_change_its_inputs():
    evaluation, comparison = with_champion()
    before = copy.deepcopy((evaluation, comparison))
    decide(evaluation, comparison)
    assert (evaluation, comparison) == before
