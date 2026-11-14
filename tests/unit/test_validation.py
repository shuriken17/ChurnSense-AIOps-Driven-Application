"""Unit tests for cleaning and validation (T-VA-01, T-VA-02, T-VA-03)."""

from __future__ import annotations

import pandas as pd
import pytest

from churnsense.pipeline import validate
from churnsense.schema import ADDON_COLUMNS, NO_INTERNET
from churnsense.standardise import standardise_frame
from churnsense.tracking import StageError
from churnsense.validation import (
    INGESTED,
    VALIDATED,
    SchemaDriftError,
    expected_columns,
    validate_rows,
)
from tests.conftest import raw_frame
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ T-VA-01 cleaning rules
def test_t_va_01_blank_total_charges_rules():
    df = standardise_frame(raw_frame(), source="real")
    estimated = df.index[(df["total_charges"].notna()) & (df["tenure_months"] > 5)][0]
    df.loc[estimated, "total_charges"] = float("nan")  # a blank total for a NON-new customer
    out = validate.clean(df)

    new_customers = df["total_charges"].isna() & (df["tenure_months"] == 0)
    assert new_customers.sum() == 3
    assert (out.loc[new_customers, "total_charges"] == 0.0).all()
    assert (out.loc[new_customers, "total_charges_imputed"] == 0).all()

    expected = df.loc[estimated, "tenure_months"] * df.loc[estimated, "monthly_charges"]
    assert out.loc[estimated, "total_charges"] == pytest.approx(expected)
    assert out.loc[estimated, "total_charges_imputed"] == 1
    assert out["total_charges"].isna().sum() == 0


# ------------------------------------------------------------------ T-VA-02 every violation
CASES = {
    "phone says no, lines say yes": (
        {"phone_service": "No", "multiple_lines": "Yes"},
        "phone_service",
    ),
    "lines marker without phone": (
        {"phone_service": "Yes", "multiple_lines": "No phone service"},
        "phone_service",
    ),
    "no internet but an add-on": (
        {"internet_service": "No", "online_security": "Yes"},
        "internet_service",
    ),
    "internet but every add-on marked none": (
        {"internet_service": "DSL", **dict.fromkeys(ADDON_COLUMNS, NO_INTERNET)},
        "internet_service",
    ),
    "unknown contract": ({"contract": "Weekly"}, "contract"),
    "unknown gender": ({"gender": "Other"}, "gender"),
    "tenure too large": ({"tenure_months": 130}, "tenure_months"),
    "tenure not a whole number": ({"tenure_months": 5.5}, "whole_number"),
    "monthly charge zero": ({"monthly_charges": 0.0}, "monthly_charges"),
    "negative total": ({"total_charges": -5.0}, "total_charges"),
    "churn label invalid": ({"churn": 2}, "churn"),
    "senior flag invalid": ({"senior_citizen": 3}, "senior_citizen"),
}


@pytest.mark.parametrize(("changes", "expected_word"), CASES.values(), ids=CASES.keys())
def test_t_va_02_each_violation_is_quarantined_with_its_reason(changes, expected_word):
    df = canonical_rows(30).astype({"tenure_months": "float64"})
    for column, value in changes.items():
        df.loc[0, column] = value
    outcome = validate_rows(df, VALIDATED)
    assert len(outcome.rejected) == 1 and len(outcome.valid) == 29
    assert expected_word in outcome.rejected.iloc[0]["reason"]


def test_duplicate_customer_ids_are_rejected():
    df = canonical_rows(10)
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    outcome = validate_rows(df, VALIDATED)
    assert len(outcome.rejected) == 2 and len(outcome.valid) == 8


def test_clean_frame_passes_completely():
    outcome = validate_rows(canonical_rows(500), VALIDATED)
    assert len(outcome.valid) == 500 and outcome.rejected.empty


def test_ingested_stage_allows_a_missing_total_charges_but_validated_does_not():
    df = standardise_frame(raw_frame(), source="real")
    assert validate_rows(df, INGESTED).rejected.empty  # blanks are allowed before cleaning
    flagged = validate_rows(df.assign(total_charges_imputed=0), VALIDATED).rejected
    assert len(flagged) == 3  # the same blanks are errors after cleaning


def test_schema_drift_is_an_error_not_a_quarantine():
    df = canonical_rows(10)
    with pytest.raises(SchemaDriftError):
        validate_rows(df.assign(surprise=1), VALIDATED)
    with pytest.raises(SchemaDriftError):
        validate_rows(df.drop(columns=["contract"]), VALIDATED)
    with pytest.raises(ValueError):
        expected_columns("nonsense")


# ------------------------------------------------------------------ stage with guardrails
def write_ingested(paths, mutate=None):
    df = standardise_frame(raw_frame(), source="real")
    if mutate:
        mutate(df)
    paths.ensure_parents(paths.ingested)
    df.to_parquet(paths.ingested, index=False)


def test_stage_runs_on_clean_data_and_writes_outputs(workspace, params):
    write_ingested(workspace)
    report = validate.run(params, workspace)
    assert report.rows_in == 500 and report.rows_out == 500
    assert report.extra["total_charges_new_customers_set_to_zero"] == 3
    assert report.extra["rows_rejected"] == 0
    valid = pd.read_parquet(workspace.validated)
    assert str(valid["tenure_months"].dtype) == "int64" and len(valid) == 500
    assert workspace.quarantine.is_file()
    assert workspace.report("validation_report.json").is_file()


def test_t_va_03_more_than_five_percent_bad_rows_fails_the_stage(workspace, params):
    def break_rows(df):
        df.loc[df.index[:40], "contract"] = "Weekly"  # 8% of 500 rows

    write_ingested(workspace, break_rows)
    with pytest.raises(StageError, match="rejected"):
        validate.run(params, workspace)
    quarantined = pd.read_parquet(workspace.quarantine)
    assert len(quarantined) == 40 and "contract" in quarantined["reason"].iloc[0]
    assert '"status": "failed"' in workspace.report("validation_report.json").read_text()


def test_few_bad_rows_are_quarantined_and_the_stage_still_passes(workspace, params):
    def break_rows(df):
        df.loc[df.index[:10], "contract"] = "Weekly"  # 2% of the rows

    write_ingested(workspace, break_rows)
    report = validate.run(params, workspace)
    assert report.rows_out == 490 and report.extra["rows_rejected"] == 10
    assert report.extra["rejection_reasons"] == {"contract": 10}


def test_too_many_missing_values_fail_the_stage(workspace, params):
    def blank_gender(df):
        df.loc[df.index[:60], "gender"] = None  # 12% missing

    write_ingested(workspace, blank_gender)
    with pytest.raises(StageError, match="missing values"):
        validate.run(params, workspace)


def test_implausible_churn_rate_fails_the_stage(workspace, params):
    def nobody_churns(df):
        df["churn"] = 0.0

    write_ingested(workspace, nobody_churns)
    with pytest.raises(StageError, match="churn rate"):
        validate.run(params, workspace)


def test_too_few_valid_rows_fail_the_stage(workspace, params):
    params.validation["min_rows"] = 1000
    write_ingested(workspace)
    with pytest.raises(StageError, match="valid rows"):
        validate.run(params, workspace)


def test_schema_drift_in_the_input_fails_the_stage(workspace, params):
    write_ingested(workspace, lambda df: df.__setitem__("new_column", 1))
    with pytest.raises(SchemaDriftError):
        validate.run(params, workspace)
