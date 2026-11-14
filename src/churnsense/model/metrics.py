"""Evaluation metrics shared by training, evaluation and the promotion gate.

The primary success metric of the project is the RECALL OF THE CHURN CLASS at a minimum
PRECISION (a missed churner costs more than a false alarm, but a retention team that is sent
too many wrong customers stops trusting the list). That metric is ``recall_at_precision``.
It is computed from the whole precision-recall curve, so it does not depend on one stored
threshold and is a fair way to compare two models on the same rows.

Alongside it the project reports ROC-AUC, PR-AUC, accuracy, precision, recall, F1 and the
Brier score, the usual set for an imbalanced two-class problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics: ROC-AUC, PR-AUC and the Brier score (probability quality)."""
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, y_proba=p)),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics when everyone with ``p >= threshold`` is called a churner."""
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "positive_rate": float(predicted.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def recall_at_precision(y: np.ndarray, p: np.ndarray, min_precision: float) -> tuple[float, float]:
    """Best recall that can be had while keeping precision at or above ``min_precision``.

    Returns:
        ``(recall, threshold)``; ``(0.0, nan)`` if no threshold reaches the precision.
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    usable = precision[:-1] >= min_precision
    if not usable.any():
        return 0.0, float("nan")
    recall = recall[:-1]
    best = int(np.flatnonzero(usable & (recall >= recall[usable].max()))[-1])
    return float(recall[best]), float(thresholds[best])


def all_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, min_precision: float
) -> dict[str, Any]:
    """Every metric of one scored set: ranking, at-threshold, and recall at precision."""
    recall, _ = recall_at_precision(y, p, min_precision)
    return {
        "n": int(len(y)),
        "n_positive": int(np.sum(y)),
        **ranking_metrics(y, p),
        **threshold_metrics(y, p, threshold),
        "recall_at_min_precision": recall,
        "min_precision": float(min_precision),
    }


def group_metrics(
    groups: pd.Series, y: np.ndarray, p: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    """Precision and recall inside every group (for example per contract type).

    A group with no churners reports ``recall = None``. These per-segment numbers are the only
    fairness-style check in the project; fairness is otherwise not assessed.
    """
    out: dict[str, dict[str, float]] = {}
    for name in sorted(groups.astype(str).unique()):
        mask = (groups.astype(str) == name).to_numpy()
        predicted = (p[mask] >= threshold).astype(int)
        positives = int(y[mask].sum())
        out[name] = {
            "n": int(mask.sum()),
            "n_positive": positives,
            "precision": float(precision_score(y[mask], predicted, zero_division=0)),
            "recall": float(recall_score(y[mask], predicted, zero_division=0))
            if positives
            else None,
        }
    return out


def paired_bootstrap(
    y: np.ndarray,
    p_new: np.ndarray,
    p_old: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    min_precision: float,
) -> dict[str, float]:
    """Paired bootstrap of (new model - old model) on the same rows.

    Both models are scored on the SAME resampled rows each time, so the difference is not
    blurred by sampling noise. Churners and non-churners are resampled separately, which keeps
    the class counts fixed and avoids resamples with a single class.

    Returns summary numbers for the difference in ROC-AUC and in recall at the precision:
    the mean and the 5th, 10th and 50th percentiles (``d_auc_p05`` and so on).
    """
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d_auc = np.empty(n_resamples)
    d_recall = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        yy = y[pick]
        d_auc[i] = roc_auc_score(yy, p_new[pick]) - roc_auc_score(yy, p_old[pick])
        d_recall[i] = (
            recall_at_precision(yy, p_new[pick], min_precision)[0]
            - recall_at_precision(yy, p_old[pick], min_precision)[0]
        )
    summary: dict[str, float] = {"n_resamples": int(n_resamples)}
    for name, values in (("d_auc", d_auc), ("d_recall", d_recall)):
        summary[f"{name}_mean"] = float(values.mean())
        for q in (5, 10, 50):
            summary[f"{name}_p{q:02d}"] = float(np.percentile(values, q))
    return summary
