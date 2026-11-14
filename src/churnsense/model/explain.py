"""Explanations: which features pushed a customer's churn score up or down.

Both model families give EXACT additive explanations in log-odds, so that for every row

    base_value + sum(feature contributions) == the model's raw score (log-odds)

holds to floating-point precision. That identity is tested (additivity check) and is also a
gate rule: a model whose explanations do not add up is never promoted.

* LightGBM: TreeSHAP, through LightGBM's own ``predict(..., pred_contrib=True)``. The ``shap``
  library's ``TreeExplainer`` calls the same routine, so the two agree by construction; the
  tests confirm it. Using the native call keeps the serving image free of ``shap``'s heavy
  numerical dependencies.
* Logistic regression: for a linear model with independent inputs the SHAP value of input j
  is ``w_j * (x_j - mean_j)``. One-hot columns are summed back to the original feature, so
  the answer is still one number per original feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from churnsense.schema import (
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

TOP_K = 3


@dataclass(frozen=True)
class Explanation:
    """SHAP result for a batch of rows (all numbers are log-odds)."""

    values: np.ndarray  # shape (n_rows, n_features), in FEATURE_COLUMNS order
    base: float  # the model's average score

    @property
    def margin(self) -> np.ndarray:
        """Base value plus all contributions: should equal the model's raw score."""
        return self.base + self.values.sum(axis=1)


class Explainer(Protocol):
    """What the serving wrapper needs from an explainer."""

    def explain(self, features: pd.DataFrame) -> Explanation:
        """Return SHAP values for the rows of ``features`` (output of derive_features)."""

    def raw_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Return the model's own raw score (log-odds) for the same rows."""


class TreeShapExplainer:
    """TreeSHAP for a fitted ``lightgbm.LGBMClassifier``."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def explain(self, features: pd.DataFrame) -> Explanation:
        """Return exact TreeSHAP values (the last column LightGBM returns is the base value)."""
        contributions = np.asarray(self.estimator.predict(features, pred_contrib=True))
        return Explanation(values=contributions[:, :-1], base=float(contributions[0, -1]))

    def raw_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Raw LightGBM score (log-odds)."""
        return np.asarray(self.estimator.predict(features, raw_score=True))


def transformed_owners() -> list[str]:
    """Original feature behind each column of the logistic-regression preprocessing.

    The preprocessing is ``[one-hot of every categorical feature..., scaled numeric...]``
    (built in ``search.build_logreg_pipeline``); this list must stay in that order.
    """
    owners = [f for f in CATEGORICAL_FEATURES for _ in CATEGORY_VALUES[f]]
    return owners + list(NUMERIC_FEATURES)


class LinearShapExplainer:
    """Exact SHAP values for a fitted logistic-regression pipeline (preprocessing + model)."""

    def __init__(self, pipeline: Any, background: pd.DataFrame) -> None:
        self.pipeline = pipeline
        prep = pipeline.named_steps["prep"]
        self.weights = np.asarray(pipeline.named_steps["model"].coef_[0], dtype=float)
        self.intercept = float(pipeline.named_steps["model"].intercept_[0])
        self.means = np.asarray(prep.transform(background), dtype=float).mean(axis=0)
        owners = transformed_owners()
        if len(owners) != len(self.weights):
            raise ValueError(f"{len(owners)} preprocessing columns but {len(self.weights)} weights")
        # indicator[j, k] = 1 if transformed column j belongs to feature k (FEATURE_COLUMNS order)
        self.indicator = np.zeros((len(owners), len(FEATURE_COLUMNS)))
        position = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
        for j, owner in enumerate(owners):
            self.indicator[j, position[owner]] = 1.0

    def explain(self, features: pd.DataFrame) -> Explanation:
        """SHAP value of feature k = sum over its columns of w_j * (x_j - mean_j)."""
        transformed = np.asarray(self.pipeline.named_steps["prep"].transform(features), dtype=float)
        per_column = (transformed - self.means) * self.weights
        base = self.intercept + float(self.weights @ self.means)
        return Explanation(values=per_column @ self.indicator, base=base)

    def raw_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Raw logistic-regression score (log-odds)."""
        return np.asarray(self.pipeline.decision_function(features))


def additivity_error(explainer: Explainer, features: pd.DataFrame) -> float:
    """Largest ``|base + sum(contributions) - raw score|`` over the rows (should be ~1e-12)."""
    explanation = explainer.explain(features)
    return float(np.max(np.abs(explanation.margin - explainer.raw_margin(features))))


def _display_value(value: Any) -> Any:
    """Make a feature value JSON-friendly: whole floats become ints, NaN becomes None."""
    if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return int(value) if float(value).is_integer() else round(float(value), 2)
    return str(value)


def top_factors(
    contributions: np.ndarray, features: pd.DataFrame, row: int, k: int = TOP_K
) -> list[dict[str, Any]]:
    """The ``k`` features with the largest effect on one row, strongest first.

    Each entry has the feature name, its value, the signed ``impact`` in log-odds and a
    ``direction`` ("increases_risk" or "decreases_risk"). The frontend turns this into plain
    language; the numbers are kept so that a test can check them.
    """
    row_values = contributions[row]
    order = np.argsort(-np.abs(row_values), kind="stable")[:k]
    return [
        {
            "feature": FEATURE_COLUMNS[i],
            "value": _display_value(features.iloc[row][FEATURE_COLUMNS[i]]),
            "impact": round(float(row_values[i]), 6),
            "direction": "increases_risk" if row_values[i] > 0 else "decreases_risk",
        }
        for i in order
    ]
