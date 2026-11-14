"""Risk bands: how a probability becomes HIGH, MEDIUM or LOW.

Two thresholds are chosen on the VALIDATION set (never on a test set):

* ``t_high``: the threshold with the highest recall among those whose precision is still at
  least ``min_precision`` (0.50 by default). Everyone at or above it is HIGH risk: at least
  half of them really churn. Of several thresholds with the same recall, the highest is used
  (same churners found, fewer false alarms).
* ``t_medium``: the highest threshold at which recall reaches ``medium_recall`` (0.90).
  Everyone at or above it (and below ``t_high``) is MEDIUM; below it is LOW. A LOW customer
  is therefore one the model would have missed less than 10% of churners to ignore.

The thresholds are stored inside the model, so the band rule always travels with the model.
No class re-weighting or oversampling is used, so probabilities stay close to calibrated and
a band means what it says.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
BANDS = (HIGH, MEDIUM, LOW)


@dataclass(frozen=True)
class Thresholds:
    """The two band thresholds plus how they performed on the validation set."""

    t_high: float
    t_medium: float
    t_high_feasible: bool
    val_precision_at_high: float
    val_recall_at_high: float
    val_precision_at_medium: float
    val_recall_at_medium: float

    def as_dict(self) -> dict[str, float | bool]:
        """Return the values as a plain dictionary (for JSON and MLflow)."""
        return asdict(self)


def select_thresholds(
    y: np.ndarray, p: np.ndarray, min_precision_high: float, medium_recall: float
) -> Thresholds:
    """Choose ``t_high`` and ``t_medium`` on validation labels ``y`` and scores ``p``."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # one entry per threshold

    usable = precision >= min_precision_high
    if usable.any():
        # Highest recall that the precision allows; if several thresholds reach that same
        # recall, take the HIGHEST of them: same churners found, fewer false alarms.
        best_recall = recall[usable].max()
        index_high = int(np.flatnonzero(usable & (recall >= best_recall))[-1])
        feasible = True
    else:  # the model cannot reach the precision: nobody is HIGH, and the gate will notice
        index_high = int(np.argmax(precision))
        feasible = False
    t_high = float(thresholds[index_high])
    if not feasible:
        t_high = float(np.nextafter(p.max(), np.inf))

    reaches = recall >= medium_recall
    index_medium = int(np.flatnonzero(reaches)[-1]) if reaches.any() else 0
    t_medium = min(float(thresholds[index_medium]), t_high)

    def at(threshold: float) -> tuple[float, float]:
        predicted = p >= threshold
        tp = float(np.sum(predicted & (y == 1)))
        found = float(np.sum(predicted))
        return (tp / found if found else 0.0), tp / float(np.sum(y == 1))

    precision_high, recall_high = at(t_high)
    precision_medium, recall_medium = at(t_medium)
    return Thresholds(
        t_high=t_high,
        t_medium=t_medium,
        t_high_feasible=feasible,
        val_precision_at_high=precision_high,
        val_recall_at_high=recall_high,
        val_precision_at_medium=precision_medium,
        val_recall_at_medium=recall_medium,
    )


def assign_band(p: np.ndarray, t_high: float, t_medium: float) -> np.ndarray:
    """Return "HIGH", "MEDIUM" or "LOW" for each probability."""
    p = np.asarray(p, dtype=float)
    return np.where(p >= t_high, HIGH, np.where(p >= t_medium, MEDIUM, LOW))
