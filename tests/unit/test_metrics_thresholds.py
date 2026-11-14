"""Unit tests for the metrics and the risk-band thresholds (T-TR-04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnsense.model import metrics as mx
from churnsense.model.thresholds import BANDS, assign_band, select_thresholds

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(3)
Y = RNG.integers(0, 2, 400)
P = np.clip(0.5 * Y + RNG.normal(0.25, 0.2, 400), 0.001, 0.999)


def brute_force_recall_at_precision(y, p, min_precision):
    best = 0.0
    for t in np.unique(p):
        predicted = p >= t
        precision = (predicted & (y == 1)).sum() / predicted.sum()
        if precision >= min_precision:
            best = max(best, (predicted & (y == 1)).sum() / (y == 1).sum())
    return best


@pytest.mark.parametrize("min_precision", [0.3, 0.5, 0.7, 0.9])
def test_recall_at_precision_matches_a_brute_force_search(min_precision):
    recall, threshold = mx.recall_at_precision(Y, P, min_precision)
    assert recall == pytest.approx(brute_force_recall_at_precision(Y, P, min_precision))
    if recall > 0:
        predicted = threshold <= P
        assert (predicted & (Y == 1)).sum() / predicted.sum() >= min_precision


def test_recall_at_precision_is_zero_when_the_precision_cannot_be_reached():
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5])  # the one churner has the lowest score
    assert mx.recall_at_precision(y, p, 0.9) == (0.0, pytest.approx(float("nan"), nan_ok=True))


def test_threshold_metrics_count_correctly():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.8, 0.1, 0.7, 0.2])
    m = mx.threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["positive_rate"] == pytest.approx(0.5)  # >= threshold counts as positive


def test_all_metrics_has_every_reported_number():
    m = mx.all_metrics(Y, P, 0.4, 0.5)
    for key in (
        "roc_auc",
        "pr_auc",
        "brier",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "recall_at_min_precision",
        "n",
        "n_positive",
        "positive_rate",
    ):
        assert key in m
    assert 0.5 < m["roc_auc"] <= 1 and m["n"] == 400


def test_group_metrics_reports_none_recall_for_a_group_without_churners():
    groups = pd.Series(["a"] * 4 + ["b"] * 2)
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.3, 0.1, 0.6, 0.2])
    out = mx.group_metrics(groups, y, p, 0.5)
    assert out["a"]["recall"] == pytest.approx(0.5) and out["a"]["n_positive"] == 2
    assert out["b"]["recall"] is None and out["b"]["n"] == 2


def test_paired_bootstrap_of_a_model_against_itself_is_exactly_zero():
    out = mx.paired_bootstrap(Y, P, P, n_resamples=50, seed=1, min_precision=0.5)
    assert out["d_auc_mean"] == 0 and out["d_auc_p05"] == 0 and out["d_recall_p05"] == 0


def test_paired_bootstrap_sees_a_clearly_better_model_and_is_repeatable():
    worse = np.clip(P + RNG.normal(0, 0.35, 400), 0, 1)
    a = mx.paired_bootstrap(Y, P, worse, n_resamples=100, seed=1, min_precision=0.5)
    b = mx.paired_bootstrap(Y, P, worse, n_resamples=100, seed=1, min_precision=0.5)
    assert a == b and a["d_auc_p05"] > 0
    flipped = mx.paired_bootstrap(Y, worse, P, n_resamples=100, seed=1, min_precision=0.5)
    assert flipped["d_auc_p50"] < 0


# ------------------------------------------------------------------ thresholds
def test_t_tr_04_thresholds_match_an_independent_brute_force_search():
    t = select_thresholds(Y, P, 0.5, 0.9)
    best_recall, best_threshold = 0.0, None
    for cut in np.unique(P):
        predicted = cut <= P
        precision = (predicted & (Y == 1)).sum() / predicted.sum()
        recall = (predicted & (Y == 1)).sum() / (Y == 1).sum()
        if precision >= 0.5 and recall >= best_recall:
            best_recall, best_threshold = recall, cut
    assert t.t_high == pytest.approx(best_threshold)
    assert t.val_recall_at_high == pytest.approx(best_recall)
    assert t.val_precision_at_high >= 0.5 and t.t_high_feasible
    assert t.val_recall_at_medium >= 0.9
    assert t.t_medium <= t.t_high
    higher = P[t.t_medium < P]  # t_medium is the HIGHEST threshold that still reaches 90% recall
    next_cut = higher.min() if higher.size else None
    if next_cut is not None and next_cut <= t.t_high:
        predicted = next_cut <= P
        assert (predicted & (Y == 1)).sum() / (Y == 1).sum() < 0.9


def test_bands_use_greater_or_equal_and_cover_every_probability():
    bands = assign_band(np.array([0.0, 0.19, 0.2, 0.49, 0.5, 1.0]), t_high=0.5, t_medium=0.2)
    assert bands.tolist() == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]
    assert set(assign_band(P, 0.5, 0.2)) <= set(BANDS)


def test_an_unreachable_precision_makes_nobody_high_and_says_so():
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5])
    t = select_thresholds(y, p, 0.9, 0.9)
    assert not t.t_high_feasible
    assert (assign_band(p, t.t_high, t.t_medium) != "HIGH").all()


def test_thresholds_serialise_to_a_plain_dict():
    d = select_thresholds(Y, P, 0.5, 0.9).as_dict()
    assert set(d) >= {"t_high", "t_medium", "t_high_feasible", "val_recall_at_high"}
