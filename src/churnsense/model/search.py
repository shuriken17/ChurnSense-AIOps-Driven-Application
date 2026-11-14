"""The two model families and their hyperparameter search.

* ``logreg``: one-hot encoding + scaling + logistic regression. The simple baseline.
* ``lightgbm``: gradient-boosted trees with native categorical handling. Usually better on
  this kind of table of mixed features.

Both are searched the same way: ``n_iter`` random candidates, stratified k-fold
cross-validation, scored by PR-AUC (average precision), which suits an imbalanced label.
``error_score="raise"`` makes a failing candidate stop the search instead of silently
becoming a bad score.

NO class re-weighting and no oversampling (SMOTE): the decision threshold is tuned instead
(see ``thresholds.py``), which keeps the probabilities close to calibrated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnsense.config import Params
from churnsense.schema import CATEGORICAL_FEATURES, CATEGORY_VALUES, NUMERIC_FEATURES

FAMILIES = ("logreg", "lightgbm")
SCORING = "average_precision"  # PR-AUC


def build_logreg_pipeline(seed: int, columns: tuple[str, ...] | None = None) -> Pipeline:
    """One-hot categoricals, scale numbers, logistic regression.

    The column order (categoricals first, then numbers) is relied upon by the explainer.
    ``columns`` restricts the model to a subset of features (used by the ablation that drops
    the engineered features); ``None`` means all 26.
    """
    categorical = [c for c in CATEGORICAL_FEATURES if columns is None or c in columns]
    numeric = [c for c in NUMERIC_FEATURES if columns is None or c in columns]
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[list(CATEGORY_VALUES[c]) for c in categorical],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        [("prep", prep), ("model", LogisticRegression(max_iter=2000, random_state=seed))]
    )


def build_lightgbm(params: Params, seed: int) -> LGBMClassifier:
    """LightGBM with settings that make a run repeatable (deterministic, fixed seed)."""
    cfg = params.train.lightgbm
    return LGBMClassifier(
        objective="binary",
        deterministic=True,
        force_col_wise=True,
        num_threads=int(cfg.num_threads),
        max_bin=int(cfg.max_bin),
        random_state=seed,
        subsample_freq=1,
        verbose=-1,
    )


def build_estimator(family: str, params: Params, columns: tuple[str, ...] | None = None) -> Any:
    """Return an unfitted estimator of the given family (optionally on fewer features)."""
    seed = int(params.seed)
    if family == "logreg":
        return build_logreg_pipeline(seed, columns)
    if family == "lightgbm":
        return build_lightgbm(params, seed)
    raise ValueError(f"unknown model family {family!r}; use one of {FAMILIES}")


def distributions(family: str) -> dict[str, Any]:
    """The random-search space of a family."""
    if family == "logreg":
        return {"model__C": loguniform(1e-3, 1e2)}
    if family == "lightgbm":
        return {
            "n_estimators": randint(100, 601),
            "learning_rate": loguniform(0.02, 0.2),
            "num_leaves": randint(8, 65),
            "max_depth": [-1, 4, 6, 8],
            "min_child_samples": randint(10, 101),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_lambda": loguniform(1e-3, 10),
            "reg_alpha": loguniform(1e-3, 10),
        }
    raise ValueError(f"unknown model family {family!r}; use one of {FAMILIES}")


def clean_param_names(search_params: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``model__`` prefix and turn numpy numbers into plain Python numbers."""
    cleaned: dict[str, Any] = {}
    for name, value in search_params.items():
        plain = value.item() if isinstance(value, np.generic) else value
        cleaned[name.removeprefix("model__")] = plain
    return cleaned


@dataclass
class SearchResult:
    """Outcome of one family's random search."""

    family: str
    best_params: dict[str, Any]  # in the estimator's own names (with model__ for logreg)
    best_score: float
    candidates: list[dict[str, Any]]  # params, mean and std of PR-AUC, rank
    seconds: float


def run_search(
    family: str, params: Params, x: pd.DataFrame, y: np.ndarray, n_iter: int, cv_folds: int
) -> SearchResult:
    """Random search with cross-validation on the rows ``x``, ``y``."""
    seed = int(params.seed)
    search = RandomizedSearchCV(
        build_estimator(family, params),
        distributions(family),
        n_iter=n_iter,
        scoring=SCORING,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
        refit=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(x, y)
    seconds = time.perf_counter() - started

    results = search.cv_results_
    candidates = [
        {
            "params": clean_param_names(results["params"][i]),
            "cv_pr_auc_mean": float(results["mean_test_score"][i]),
            "cv_pr_auc_std": float(results["std_test_score"][i]),
            "rank": int(results["rank_test_score"][i]),
        }
        for i in range(len(results["params"]))
    ]
    return SearchResult(
        family=family,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        candidates=candidates,
        seconds=seconds,
    )
