"""Data drift: the Population Stability Index (PSI) of live inputs against the training profile.

PSI compares two distributions over the same bins: ``sum((a - e) * ln(a / e))`` with ``e`` the
share in training and ``a`` the share now. Common reading: under 0.10 no change, 0.10 to 0.25 a
moderate shift, above 0.25 a large shift. The bins come from the reference profile saved by the
featurize stage, so training and monitoring always use the same yardstick. Values outside the
training range fall in the first or last bin; a category never seen in training counts in an
extra "(other)" bucket, so brand-new categories show up as drift too.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-4
OTHER = "(other)"


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = EPSILON) -> float:
    """PSI of two proportion vectors over the same bins (zero shares are floored at ``eps``)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_shares(values: pd.Series, edges: Sequence[float]) -> list[float]:
    """Share of ``values`` in each bin ``(-inf, e0], (e0, e1], ..., (ek, +inf)``."""
    x = values.dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges) + 1)
    return (counts / max(x.size, 1)).tolist()


def categorical_shares(values: pd.Series, categories: Sequence[str]) -> list[float]:
    """Share of ``values`` per known category, plus one share for everything else."""
    text = values.astype("object").where(values.notna(), OTHER).astype(str)
    counts = [float((text == c).sum()) for c in categories]
    counts.append(float(len(text) - sum(counts)))
    return [c / max(len(text), 1) for c in counts]


@dataclass
class DriftReport:
    """The outcome of one drift check."""

    n_rows: int
    psi_by_feature: dict[str, float] = field(default_factory=dict)
    max_psi: float = 0.0
    worst_feature: str | None = None
    drifted: list[str] = field(default_factory=list)  # features at or above the alert level
    enough_data: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary for JSON."""
        return {
            "n_rows": self.n_rows,
            "enough_data": self.enough_data,
            "max_psi": round(self.max_psi, 5),
            "worst_feature": self.worst_feature,
            "drifted_features": self.drifted,
            "psi_by_feature": {k: round(v, 5) for k, v in self.psi_by_feature.items()},
        }


def compute_drift(
    profile: dict[str, Any], features: pd.DataFrame, *, min_rows: int, psi_alert: float
) -> DriftReport:
    """Compare ``features`` (output of ``derive_features``) with the reference ``profile``."""
    if len(features) < min_rows:
        return DriftReport(n_rows=len(features), enough_data=False)
    scores: dict[str, float] = {}
    for name, spec in profile["numeric"].items():
        scores[name] = psi(spec["proportions"], numeric_shares(features[name], spec["edges"]))
    for name, spec in profile["categorical"].items():
        expected = [*spec["proportions"], 0.0]  # nothing unseen was in training
        scores[name] = psi(expected, categorical_shares(features[name], spec["values"]))
    worst = max(scores, key=scores.get)
    return DriftReport(
        n_rows=len(features),
        psi_by_feature=scores,
        max_psi=scores[worst],
        worst_feature=worst,
        drifted=sorted((k for k, v in scores.items() if v >= psi_alert), key=lambda k: -scores[k]),
    )
