"""Unit test for the EDA stage."""

from __future__ import annotations

import json

import pytest

from churnsense.pipeline import eda
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


def test_eda_writes_figures_and_a_summary(workspace, params):
    workspace.ensure_parents(workspace.validated)
    canonical_rows(500).to_parquet(workspace.validated, index=False)
    report = eda.run(params, workspace)
    figures = sorted(p.name for p in workspace.eda_dir.glob("*.png"))
    assert figures == sorted(report.extra["figures"]) and len(figures) == 5
    assert all((workspace.eda_dir / name).stat().st_size > 1000 for name in figures)

    summary = json.loads(workspace.report("eda_summary.json").read_text("utf-8"))
    assert summary["rows_real"] == 500
    assert sum(summary["class_balance"].values()) == 500
    assert set(summary["churn_rate_by_segment"]["contract"]) == {
        "Month-to-month",
        "One year",
        "Two year",
    }
    assert summary["duplicate_customer_ids"] == 0 and summary["missing_values_after_cleaning"] == 0


def test_iqr_outlier_rule():
    import pandas as pd

    stats = eda.iqr_outliers(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
    assert stats["outliers"] == 1
