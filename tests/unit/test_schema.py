"""Unit tests for the canonical schema (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnsense import schema as s
from churnsense.config import Paths
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


def test_field_counts():
    assert len(s.INPUT_COLUMNS) == 20
    assert len(s.RAW_COLUMNS) == 21
    assert len(s.ADDON_COLUMNS) == 6
    assert len(s.FEATURE_COLUMNS) == 26
    assert len(s.INGESTED_COLUMNS) == 24
    assert s.VALIDATED_COLUMNS[-1] == "total_charges_imputed"


def test_features_are_unique_and_partitioned():
    assert len(set(s.FEATURE_COLUMNS)) == len(s.FEATURE_COLUMNS)
    categorical, numeric = set(s.CATEGORICAL_FEATURES), set(s.NUMERIC_FEATURES)
    assert categorical | numeric == set(s.FEATURE_COLUMNS)
    assert not categorical & numeric
    assert "customer_id" not in s.FEATURE_COLUMNS and "churn" not in s.FEATURE_COLUMNS


def test_both_header_spellings_map_to_the_canonical_name():
    for spec in s.INPUT_FIELDS:
        assert s.canonical_name(spec.ibm_name) == spec.name
        assert s.canonical_name(spec.name) == spec.name
    assert s.canonical_name("Churn") == "churn"
    assert s.canonical_name("tenure") == "tenure_months"
    assert s.canonical_name("SOMETHING ELSE") is None


def test_rename_keeps_unknown_columns_so_drift_can_be_reported():
    out = s.rename_to_canonical(pd.DataFrame({"customerID": ["a"], "mystery": [1]}))
    assert list(out.columns) == ["customer_id", "mystery"]


def test_rename_rejects_two_headers_for_one_field():
    with pytest.raises(s.SchemaError):
        s.rename_to_canonical(pd.DataFrame({"tenure": [1], "tenure_months": [2]}))


def test_clean_rows_have_no_cross_field_violations():
    assert (s.cross_field_violations(canonical_rows(500)) == "").all()


def test_cross_field_phone_and_internet_rules():
    df = canonical_rows(3)
    df.loc[0, ["phone_service", "multiple_lines"]] = ["No", "Yes"]
    df.loc[1, ["internet_service", "online_security"]] = ["No", "Yes"]
    df.loc[2, "phone_service"] = "No"
    df.loc[2, "multiple_lines"] = "Yes"
    df.loc[2, "internet_service"] = "No"
    df.loc[2, "online_security"] = "Yes"
    reasons = s.cross_field_violations(df)
    assert "phone_service" in reasons[0]
    assert "internet_service" in reasons[1]
    assert "phone_service" in reasons[2] and "internet_service" in reasons[2]


def test_normalise_categories_fixes_case_and_spaces_but_keeps_unknown_values():
    df = pd.DataFrame(
        {"contract": [" month-to-month ", "TWO YEAR", "Weekly"], "gender": ["male", "Female", "x"]}
    )
    out = s.normalise_categories(df)
    assert out["contract"].tolist() == ["Month-to-month", "Two year", "Weekly"]
    assert out["gender"].tolist() == ["Male", "Female", "x"]


def test_field_catalogue_describes_all_inputs():
    catalogue = s.field_catalogue()
    assert [f["name"] for f in catalogue] == list(s.INPUT_COLUMNS)
    optional = {f["name"] for f in catalogue if not f["required"]}
    assert optional == {"customer_id", "total_charges"}
    contract = next(f for f in catalogue if f["name"] == "contract")
    assert contract["allowed"] == ["Month-to-month", "One year", "Two year"]


def test_categorical_dtype_is_fixed_and_rejects_numeric_columns():
    dtype = s.categorical_dtype("contract")
    assert list(dtype.categories) == ["Month-to-month", "One year", "Two year"]
    with pytest.raises(s.SchemaError):
        s.categorical_dtype("tenure_months")


def test_params_agree_with_the_schema_and_paths(params):
    assert params.features.tenure_bins == list(s.TENURE_BINS)
    paths = Paths(root=Path("."))
    assert str(paths.raw_csv).endswith(params.ingest.raw_path)
    assert str(paths.feedback_dir).endswith(params.ingest.feedback_dir)
