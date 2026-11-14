"""Unit tests for feature engineering (T-FE-01, T-FE-02) and the reference profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnsense.features.derive import derive_features
from churnsense.features.profile import build_reference_profile, numeric_profile
from churnsense.schema import CATEGORY_VALUES, FEATURE_COLUMNS, NUMERIC_FEATURES, SchemaError
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit

IBM_ROW_1 = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 1,
    "phone_service": "No",
    "multiple_lines": "No phone service",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 29.85,
    "total_charges": 29.85,
}


def one_row(**changes):
    return pd.DataFrame([{**IBM_ROW_1, **changes}])


def test_t_fe_01_known_input_gives_known_output():
    out = derive_features(one_row()).iloc[0]
    assert out["tenure_bucket"] == "0-6"
    assert out["avg_charge_per_month"] == pytest.approx(29.85)
    assert out["charge_delta"] == pytest.approx(0.0)
    assert out["n_addons"] == 1  # only online_backup is "Yes"
    assert (out["is_month_to_month"], out["is_fiber"], out["is_auto_pay"]) == (1, 0, 0)


def test_t_fe_01_brand_new_customer_has_no_nan_or_inf():
    out = derive_features(one_row(tenure_months=0, total_charges=0.0))
    assert out["avg_charge_per_month"].iloc[0] == 0.0  # divides by max(tenure, 1)
    assert out["charge_delta"].iloc[0] == pytest.approx(29.85)
    assert out["tenure_bucket"].iloc[0] == "0-6"  # tenure 0 belongs to the first bucket
    assert np.isfinite(out[list(NUMERIC_FEATURES)].to_numpy(dtype=float)).all()


@pytest.mark.parametrize(
    ("tenure", "bucket"),
    [
        (0, "0-6"),
        (6, "0-6"),
        (7, "7-12"),
        (12, "7-12"),
        (13, "13-24"),
        (24, "13-24"),
        (25, "25-48"),
        (48, "25-48"),
        (49, "49+"),
        (72, "49+"),
    ],
)
def test_tenure_bucket_edges(tenure, bucket):
    assert derive_features(one_row(tenure_months=tenure)).iloc[0]["tenure_bucket"] == bucket


def test_auto_pay_fiber_and_addon_count():
    row = one_row(
        payment_method="Credit card (automatic)",
        internet_service="Fiber optic",
        online_security="Yes",
        tech_support="Yes",
        streaming_tv="Yes",
    )
    out = derive_features(row).iloc[0]
    assert (out["is_auto_pay"], out["is_fiber"], out["n_addons"]) == (1, 1, 4)


def test_t_fe_02_unseen_category_becomes_missing():
    out = derive_features(one_row(contract="Weekly"))
    assert pd.isna(out["contract"].iloc[0])
    assert out["is_month_to_month"].iloc[0] == 0
    assert not out.drop(columns="contract").isna().any().any()


def test_missing_total_charges_is_estimated_from_tenure():
    out = derive_features(one_row(tenure_months=10, total_charges=np.nan))
    assert out["total_charges"].iloc[0] == pytest.approx(298.5)


def test_missing_input_column_is_an_error():
    with pytest.raises(SchemaError):
        derive_features(one_row().drop(columns=["contract"]))


def test_output_columns_order_and_dtypes_are_fixed():
    out = derive_features(canonical_rows(100))
    assert list(out.columns) == list(FEATURE_COLUMNS)
    for column, allowed in CATEGORY_VALUES.items():
        assert list(out[column].cat.categories) == list(allowed)
    assert all(str(out[c].dtype) == "float64" for c in NUMERIC_FEATURES)


def test_derive_features_is_pure():
    frame = canonical_rows(50)
    before = frame.copy()
    first, second = derive_features(frame), derive_features(frame)
    pd.testing.assert_frame_equal(frame, before)  # input untouched
    pd.testing.assert_frame_equal(first, second)  # same input, same output


def test_all_fixture_rows_give_finite_features():
    out = derive_features(canonical_rows(500))
    assert np.isfinite(out[list(NUMERIC_FEATURES)].to_numpy(dtype=float)).all()
    assert not out.isna().any().any()


def test_numeric_profile_bins_cover_everything_and_sum_to_one():
    profile = numeric_profile(pd.Series(np.arange(1000, dtype=float)))
    assert sum(profile["proportions"]) == pytest.approx(1.0)
    assert len(profile["proportions"]) == len(profile["edges"]) + 1
    assert profile["edges"] == sorted(profile["edges"])


def test_numeric_profile_merges_duplicate_edges():
    profile = numeric_profile(pd.Series([0.0] * 90 + [1.0] * 10))
    assert len(profile["edges"]) < 9  # most quantiles are 0.0, so edges collapse


def test_reference_profile_content():
    features = derive_features(canonical_rows(300))
    profile = build_reference_profile(features)
    assert profile["n_rows"] == 300
    assert set(profile["numeric"]) == {
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_charge_per_month",
        "charge_delta",
        "n_addons",
    }
    contract = profile["categorical"]["contract"]
    assert contract["values"] == ["Month-to-month", "One year", "Two year"]
    assert sum(contract["proportions"]) == pytest.approx(1.0)
