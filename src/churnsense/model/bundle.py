"""The trained model as one picklable object: estimator, explainer and band thresholds.

``ModelBundle.score`` is the whole serving logic: raw customer fields in, probability, risk
band and top factors out. It has no dependency on MLflow, so unit tests and the pipeline can
call it directly; ``pyfunc.ChurnSenseModel`` is only a thin MLflow wrapper around it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from churnsense.features.derive import derive_features
from churnsense.model.explain import Explainer, additivity_error, top_factors
from churnsense.model.thresholds import Thresholds, assign_band

OUTPUT_COLUMNS = ("churn_probability", "risk_band", "top_factors", "model_run_id")


@dataclass
class ModelBundle:
    """Everything needed to score customers.

    Attributes:
        family: "logreg" or "lightgbm".
        estimator: the fitted estimator (a sklearn pipeline or an LGBMClassifier).
        explainer: gives exact SHAP values for ``estimator``.
        thresholds: the HIGH / MEDIUM band thresholds chosen on the validation set.
        run_id: MLflow run that trained the model (returned with every prediction).
        trained_at: UTC time of training (ISO format).
        metadata: free-form extras (library versions, git commit, validation metrics).
    """

    family: str
    estimator: Any
    explainer: Explainer
    thresholds: Thresholds
    run_id: str
    trained_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Churn probability for rows of ``derive_features`` output."""
        return np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=float)

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score raw customer rows.

        Returns:
            One row per input row with ``churn_probability`` (0 to 1), ``risk_band``
            (HIGH, MEDIUM or LOW), ``top_factors`` (JSON text: the three features that moved
            the score most, with signed log-odds impact) and ``model_run_id``.
        """
        features = derive_features(raw)
        probability = self.probabilities(features)
        explanation = self.explainer.explain(features)
        factors = [
            json.dumps(top_factors(explanation.values, features, i), separators=(",", ":"))
            for i in range(len(features))
        ]
        return pd.DataFrame(
            {
                "churn_probability": probability,
                "risk_band": assign_band(
                    probability, self.thresholds.t_high, self.thresholds.t_medium
                ),
                "top_factors": factors,
                "model_run_id": self.run_id,
            }
        )

    def additivity_check(self, raw: pd.DataFrame, n_rows: int = 100) -> float:
        """Largest gap between ``base + sum(SHAP)`` and the raw model score on ``n_rows``."""
        return additivity_error(self.explainer, derive_features(raw.head(n_rows)))
