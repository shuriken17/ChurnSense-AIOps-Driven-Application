"""Stage 8: the promotion gate. It decides whether the new model may replace the champion.

Inputs:  reports/eval_metrics.json, reports/comparison.json, models/run_info.json
Output:  reports/gate_decision.json  with ``decision`` = PROMOTE or REJECT and every reason

The gate FAILS CLOSED: if anything is missing, unreadable or not a number, the answer is
REJECT. A rejected model costs a day; a wrongly promoted model costs customers. The stage
itself always succeeds (exit 0) so that the pipeline records the decision; the deployment
step reads the decision.

Rules, in three groups (all limits come from ``params.yaml`` under ``gate``):

1. Sanity: the model loads in a fresh process, its predictions are finite and between 0 and 1,
   its risk bands and top factors are well formed, its SHAP values add up, its libraries match
   the ones it was trained with, it was trained on committed code, it answers fast enough, it
   is small enough, and it does not flag almost nobody or almost everybody.
2. Absolute floors on the fixed golden test set: ROC-AUC and the recall reached at a minimum
   precision. The recall is read from the whole precision-recall curve ("what recall can you
   get while keeping at least that precision"), so it does not depend on one stored
   threshold and is not thrown off by a lucky or unlucky threshold on a small test set.
3. Against the champion (if there is one): not worse on the golden set (5th percentile of a
   paired bootstrap difference above a small allowed loss) AND better on the recent data
   (10th percentile of the ROC-AUC difference above zero), with enough recent rows and
   churners to say anything at all. With too little recent data the champion stays.
"""

from __future__ import annotations

import json
from typing import Any

from churnsense.config import Params, Paths
from churnsense.logging_utils import get_logger
from churnsense.tracking import StageReport, tracked_stage

log = get_logger(__name__)


def _number(value: Any) -> float:
    """Return ``value`` as a float, or NaN (which fails every comparison, so fails closed)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def decide(
    evaluation: dict[str, Any], comparison: dict[str, Any], gate: Any, candidate_run_id: str
) -> dict[str, Any]:
    """Apply the rules and return the decision with every check recorded."""
    checks: list[dict[str, Any]] = []

    def check(rule: str, passed: bool, detail: str) -> None:
        checks.append({"rule": rule, "passed": bool(passed), "detail": detail})

    challenger = evaluation["challenger"]
    golden = challenger["golden"]
    sanity = evaluation["sanity"]

    check(
        "evaluation_matches_candidate",
        evaluation.get("model_run_id") == candidate_run_id,
        f"evaluation is for run {str(evaluation.get('model_run_id'))[:8]}, "
        f"candidate is {candidate_run_id[:8]}",
    )

    # 1. sanity
    for key, text in (
        ("loads_in_fresh_process", "loads in a fresh process"),
        ("no_nan_predictions", "all predictions are finite"),
        ("probabilities_in_range", "probabilities are between 0 and 1"),
        ("bands_valid", "risk bands are HIGH, MEDIUM or LOW"),
        ("top_factors_valid", "top factors are well formed"),
        ("shap_additive", "SHAP values add up to the model score"),
        ("dependencies_match", "libraries match the training environment"),
    ):
        check(
            key,
            sanity.get(key) is True,
            f"{text}: {'yes' if sanity.get(key) is True else 'NO'}"
            + (
                f" ({'; '.join(sanity['dependency_mismatches'])})"
                if key == "dependencies_match" and sanity.get(key) is not True
                else ""
            ),
        )
    check(
        "trained_on_committed_code",
        sanity.get("git_dirty") is False,
        "model was trained from a clean Git tree"
        if sanity.get("git_dirty") is False
        else "model was trained from UNCOMMITTED code",
    )
    latency = _number(sanity.get("latency_ms_p95"))
    check(
        "latency",
        latency <= float(gate.max_p95_latency_ms),
        f"p95 single-row latency {latency:.1f} ms (limit {gate.max_p95_latency_ms})",
    )
    size = _number(sanity.get("model_size_mb"))
    check(
        "model_size",
        size <= float(gate.max_model_mb),
        f"model is {size:.2f} MB (limit {gate.max_model_mb})",
    )
    low, high = (float(x) for x in gate.pos_rate_bounds)
    rate = _number(sanity.get("golden_positive_rate"))
    check(
        "golden_positive_rate",
        low <= rate <= high,
        f"flags {rate:.1%} of golden customers as HIGH (allowed {low:.0%} to {high:.0%})",
    )

    # 2. absolute floors on the golden set
    auc = _number(golden.get("roc_auc"))
    check(
        "golden_roc_auc_floor",
        auc >= float(gate.min_roc_auc),
        f"golden ROC-AUC {auc:.4f} (floor {gate.min_roc_auc})",
    )
    recall = _number(golden.get("recall_at_min_precision"))
    check(
        "golden_recall_floor",
        recall >= float(gate.min_recall),
        f"golden recall {recall:.3f} at precision >= {gate.min_precision} "
        f"(floor {gate.min_recall})",
    )

    # 3. against the champion
    champion = evaluation.get("champion")
    first_model = champion is None
    if champion is not None and "error" in champion:
        check("champion_scored", False, f"the champion could not be scored: {champion['error']}")
    elif champion is not None:
        diff = comparison.get("golden", {})
        d_auc = _number(diff.get("d_auc_p05"))
        check(
            "golden_auc_noninferior",
            d_auc >= -float(gate.golden_auc_noninferiority),
            f"5th percentile of the ROC-AUC change on golden: {d_auc:+.4f} "
            f"(allowed loss {gate.golden_auc_noninferiority})",
        )
        d_recall = _number(diff.get("d_recall_p05"))
        check(
            "golden_recall_noninferior",
            d_recall >= -float(gate.golden_recall_noninferiority),
            f"5th percentile of the recall change on golden: {d_recall:+.4f} "
            f"(allowed loss {gate.golden_recall_noninferiority})",
        )
        rows, positives = (
            int(evaluation.get("recent_rows", 0)),
            int(evaluation.get("recent_positives", 0)),
        )
        enough = (
            rows >= int(gate.recent_min_rows)
            and positives >= int(gate.recent_min_positives)
            and "recent" in comparison
        )
        check(
            "recent_data_sufficient",
            enough,
            f"{rows} recent rows with {positives} churners "
            f"(need {gate.recent_min_rows} and {gate.recent_min_positives})",
        )
        if enough:
            d10 = _number(comparison["recent"].get("d_auc_p10"))
            check(
                "recent_auc_improvement",
                d10 > 0,
                f"10th percentile of the ROC-AUC change on recent data: {d10:+.4f} (must be > 0)",
            )

    reasons = [f"{c['rule']}: {c['detail']}" for c in checks if not c["passed"]]
    return {
        "decision": "REJECT" if reasons else "PROMOTE",
        "reasons": reasons,
        "first_model": first_model,
        "candidate_run_id": candidate_run_id,
        "champion_run_id": None if champion is None else champion.get("run_id"),
        "checks": checks,
    }


@tracked_stage("gate", report="gate_decision.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the gate stage (never raises for a bad model: it records REJECT instead)."""
    try:
        evaluation = json.loads(paths.report("eval_metrics.json").read_text("utf-8"))
        comparison = json.loads(paths.report("comparison.json").read_text("utf-8"))
        run_info = json.loads(paths.run_info.read_text("utf-8"))
        decision = decide(evaluation, comparison, params.gate, run_info["run_id"])
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY problem
        decision = {
            "decision": "REJECT",
            "reasons": [f"gate_error: {type(exc).__name__}: {exc}"],
            "first_model": None,
            "candidate_run_id": None,
            "champion_run_id": None,
            "checks": [],
        }
    log.info(
        "gate decision",
        extra={"decision": decision["decision"], "reasons": len(decision["reasons"])},
    )
    return StageReport(
        rows_in=len(decision["checks"]), rows_out=len(decision["checks"]), extra=decision
    )


if __name__ == "__main__":
    run.main()
