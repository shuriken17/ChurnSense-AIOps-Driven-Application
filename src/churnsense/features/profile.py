"""Reference profile: what the training data looked like.

The drift monitor compares live requests with this profile using the Population Stability
Index (PSI). The profile stores, per feature, only the shape of the distribution:

* numeric features: interior quantile edges and the share of rows in each bin,
* categorical features: the share of rows per category.

It is computed on REAL training rows only (never on synthetic rows), because synthetic rows
are copies of real ones and would make the reference look more certain than it is.
The output is deterministic: the same data gives the same bytes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churnsense.schema import (
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    DRIFT_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
)

PROFILE_VERSION = 1
N_BINS = 10


def numeric_profile(values: pd.Series, n_bins: int = N_BINS) -> dict[str, Any]:
    """Quantile bins of a numeric feature.

    The bins are ``(-inf, e0], (e0, e1], ..., (ek, +inf)``, so values outside the training
    range still land in the first or last bin. Duplicate edges (common for small whole
    numbers like n_addons) are merged, so there can be fewer than ``n_bins`` bins.
    """
    x = values.dropna().to_numpy(dtype=float)
    if x.size == 0:
        raise ValueError("cannot build a profile from an empty column")
    interior = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, interior))
    bin_index = np.searchsorted(edges, x, side="left")  # x <= edges[i]  ->  bin i
    counts = np.bincount(bin_index, minlength=len(edges) + 1)
    return {
        "edges": [float(e) for e in edges],
        "proportions": [float(c) / x.size for c in counts],
        "n": int(x.size),
    }


def categorical_profile(values: pd.Series, column: str) -> dict[str, Any]:
    """Share of rows per allowed category (in the fixed category order)."""
    categories = list(CATEGORY_VALUES[column])
    counts = values.astype("object").value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        raise ValueError(f"cannot build a profile for empty column {column!r}")
    return {
        "values": categories,
        "proportions": [float(counts.get(c, 0)) / total for c in categories],
        "n": total,
    }


def build_reference_profile(features: pd.DataFrame) -> dict[str, Any]:
    """Build the profile of a feature frame (output of ``derive_features``)."""
    return {
        "version": PROFILE_VERSION,
        "n_rows": int(len(features)),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric": {c: numeric_profile(features[c]) for c in DRIFT_NUMERIC_FEATURES},
        "categorical": {c: categorical_profile(features[c], c) for c in CATEGORICAL_FEATURES},
    }


def save_profile(path: Path, profile: dict[str, Any]) -> None:
    """Write the profile as JSON (sorted keys, atomic rename, so the bytes are repeatable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_profile(path: Path) -> dict[str, Any]:
    """Read a profile written by ``save_profile``."""
    return json.loads(path.read_text(encoding="utf-8"))
