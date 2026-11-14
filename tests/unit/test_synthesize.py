"""Unit tests for the synthetic row generator, called directly (no Ray)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnsense.pipeline.synthesize import make_chunk, quality_report
from churnsense.schema import VALIDATED_COLUMNS
from churnsense.validation import VALIDATED, validate_rows
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def real() -> pd.DataFrame:
    return canonical_rows(400)


def test_chunk_has_the_requested_size_columns_and_ids(real):
    chunk = make_chunk(real, 500, seed=1, first_number=1000, noise_frac=0.05)
    assert len(chunk) == 500 and list(chunk.columns) == list(VALIDATED_COLUMNS)
    assert (
        chunk["customer_id"].iloc[0] == "SYN-1000" and chunk["customer_id"].iloc[-1] == "SYN-1499"
    )
    assert set(chunk["source"]) == {"synthetic"} and set(chunk["batch_id"]) == {"synthetic"}


def test_same_seed_same_rows_other_seed_other_rows(real):
    first = make_chunk(real, 300, seed=5, first_number=0, noise_frac=0.05)
    again = make_chunk(real, 300, seed=5, first_number=0, noise_frac=0.05)
    other = make_chunk(real, 300, seed=6, first_number=0, noise_frac=0.05)
    pd.testing.assert_frame_equal(first, again)
    assert not first["monthly_charges"].equals(other["monthly_charges"])


def test_churn_share_is_kept_almost_exactly(real):
    chunk = make_chunk(real, 2000, seed=2, first_number=0, noise_frac=0.05)
    assert abs(chunk["churn"].mean() - real["churn"].mean()) < 0.002


def test_numbers_stay_inside_the_real_range_and_new_customers_stay_new(real):
    chunk = make_chunk(real, 2000, seed=3, first_number=0, noise_frac=0.20)  # big noise on purpose
    assert (
        chunk["monthly_charges"]
        .between(real["monthly_charges"].min(), real["monthly_charges"].max())
        .all()
    )
    assert chunk["tenure_months"].between(0, real["tenure_months"].max()).all()
    new_customers = chunk["tenure_months"] == 0
    assert (chunk.loc[new_customers, "total_charges"] == 0).all()
    assert (chunk.loc[~new_customers, "tenure_months"] >= 1).all()
    assert (chunk["total_charges"] >= 0).all() and np.isfinite(chunk["total_charges"]).all()


def test_rows_pass_the_same_validation_as_real_data(real):
    chunk = make_chunk(real, 1000, seed=4, first_number=0, noise_frac=0.05)
    assert validate_rows(chunk, VALIDATED).rejected.empty


def test_quality_report_measures_distance_to_the_real_rows(real):
    chunk = make_chunk(real, 4000, seed=8, first_number=0, noise_frac=0.05)
    report = quality_report(real, chunk)
    assert set(report["ks_statistic"]) == {"tenure_months", "monthly_charges", "total_charges"}
    assert report["max_ks_statistic"] < 0.15 and report["max_category_gap"] < 0.05
    identical = quality_report(real, real)
    assert identical["max_ks_statistic"] == 0 and identical["max_category_gap"] == 0
