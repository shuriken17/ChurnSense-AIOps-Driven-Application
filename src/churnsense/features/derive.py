"""Feature engineering: ONE pure function, used by training and by serving.

Because the training pipeline (``featurize`` stage) and the model wrapper (at serving time)
both call ``derive_features``, the features cannot disagree between the two places. That
removes the most common source of train/serve skew.

The function is stateless: the same input always gives the same output, with no fitted
statistics. It never touches the target column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnsense.schema import (
    ADDON_COLUMNS,
    CATEGORICAL_INPUTS,
    FEATURE_COLUMNS,
    INPUT_COLUMNS,
    TENURE_BINS,
    TENURE_LABELS,
    SchemaError,
    categorical_dtype,
)

_AUTO_PAY = ("Bank transfer (automatic)", "Credit card (automatic)")
_REQUIRED = tuple(c for c in INPUT_COLUMNS if c != "customer_id")


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 26 model features from the input fields.

    Args:
        df: a frame with the 19 input fields (all of INPUT_COLUMNS except customer_id).
            ``total_charges`` may be missing; it is then estimated as tenure * monthly.

    Returns:
        A new frame with exactly the columns of ``FEATURE_COLUMNS``, in that order.
        Categorical features use a fixed ``CategoricalDtype``; a value that is not a known
        category becomes NaN (LightGBM treats it as missing). Numeric features are float64.

    Raises:
        SchemaError: if an input column is missing.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"cannot derive features, missing columns: {missing}")

    out = pd.DataFrame(index=df.index)

    # Raw categorical features: fixed category list; unknown values turn into NaN.
    for column in CATEGORICAL_INPUTS:
        out[column] = df[column].astype("object").astype(categorical_dtype(column))

    # Raw numeric features.
    tenure = pd.to_numeric(df["tenure_months"], errors="coerce").astype(float)
    monthly = pd.to_numeric(df["monthly_charges"], errors="coerce").astype(float)
    total = pd.to_numeric(df["total_charges"], errors="coerce").astype(float)
    total = total.fillna(tenure * monthly)  # estimate when the bill total is unknown
    out["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce").astype(float)
    out["tenure_months"] = tenure
    out["monthly_charges"] = monthly
    out["total_charges"] = total

    # Derived features.
    average = total / np.maximum(tenure, 1.0)  # a brand-new customer (tenure 0) divides by 1
    out["avg_charge_per_month"] = average
    out["charge_delta"] = monthly - average
    out["n_addons"] = (df[list(ADDON_COLUMNS)] == "Yes").sum(axis=1).astype(float)
    out["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(float)
    out["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(float)
    out["is_auto_pay"] = df["payment_method"].isin(_AUTO_PAY).astype(float)
    out["tenure_bucket"] = pd.cut(
        tenure, bins=list(TENURE_BINS), labels=list(TENURE_LABELS), include_lowest=True
    ).astype(categorical_dtype("tenure_bucket"))

    return out[list(FEATURE_COLUMNS)]
