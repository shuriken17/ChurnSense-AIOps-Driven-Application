"""Unit tests for standardising raw input (T-IN-01)."""

from __future__ import annotations

import pandas as pd
import pytest

from churnsense.schema import INGESTED_COLUMNS, NOMINAL_RAW_EVENT_TS, canonical_name
from churnsense.standardise import standardise_batch, standardise_frame
from tests.conftest import raw_frame

pytestmark = pytest.mark.unit


def test_t_in_01_headers_types_and_lineage():
    out = standardise_frame(raw_frame(), source="real")
    assert list(out.columns) == list(INGESTED_COLUMNS)
    assert (out["source"] == "real").all() and (out["batch_id"] == "raw").all()
    assert out["event_ts"].iloc[0] == NOMINAL_RAW_EVENT_TS
    assert str(out["event_ts"].dtype) == "datetime64[us, UTC]"
    assert out["total_charges"].isna().sum() == 3  # the three blank-total fixture rows
    assert set(out["churn"].unique()) == {0.0, 1.0}
    assert str(out["tenure_months"].dtype) == "float64"


def test_snake_case_headers_give_the_same_result():
    ibm = standardise_frame(raw_frame(), source="real")
    snake = standardise_frame(raw_frame().rename(columns=canonical_name), source="real")
    pd.testing.assert_frame_equal(ibm, snake)


def test_whitespace_and_yes_no_and_bad_numbers():
    df = pd.DataFrame(
        {
            "customerID": [" A-1 ", "A-2"],
            "Churn": ["Yes ", "maybe"],
            "SeniorCitizen": ["1", "0"],
            "tenure": ["12", "abc"],
            "TotalCharges": [" ", "55.5"],
            "gender": [" Male", "Female "],
        }
    )
    out = standardise_frame(df, source="real")
    assert out["customer_id"].tolist() == ["A-1", "A-2"]
    assert out["gender"].tolist() == ["Male", "Female"]
    assert out["churn"].iloc[0] == 1 and pd.isna(out["churn"].iloc[1])  # unknown label -> NaN
    assert pd.isna(out["tenure_months"].iloc[1])  # reported later by validation
    assert pd.isna(out["total_charges"].iloc[0]) and out["total_charges"].iloc[1] == 55.5


def test_given_event_time_and_batch_id_are_used():
    out = standardise_frame(
        raw_frame().head(2), source="feedback", event_ts="2026-03-01T12:00:00", batch_id="b7"
    )
    assert out["event_ts"].iloc[0] == pd.Timestamp("2026-03-01T12:00:00", tz="UTC")
    assert (out["batch_id"] == "b7").all() and (out["source"] == "feedback").all()


def test_existing_lineage_columns_are_kept():
    df = standardise_frame(raw_frame().head(2), source="real")
    again = standardise_frame(df, source="feedback", batch_id="other")
    assert (again["source"] == "real").all() and (again["batch_id"] == "raw").all()


def test_ray_adapter_gives_the_same_frame():
    direct = standardise_frame(raw_frame(), source="real", event_ts="", batch_id="raw")
    adapted = standardise_batch(raw_frame(), "real", "", "raw")
    pd.testing.assert_frame_equal(direct, adapted)
