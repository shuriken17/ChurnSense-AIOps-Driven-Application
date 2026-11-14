"""Unit tests for duplicate handling in the ingest stage."""

from __future__ import annotations

import pandas as pd
import pytest

from churnsense.pipeline.ingest import remove_duplicates

pytestmark = pytest.mark.unit


def frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "event_ts", "batch_id", "value"])


def test_no_duplicates_changes_nothing():
    df = frame([("a", "2026-01-01", "b1", 1), ("b", "2026-01-01", "b1", 2)])
    kept, removed, within = remove_duplicates(df)
    assert removed == 0 and within == 0 and len(kept) == 2


def test_the_latest_report_wins_and_row_order_is_kept():
    df = frame(
        [("a", "2026-01-01", "b1", 1), ("b", "2026-01-01", "b1", 2), ("a", "2026-02-01", "b2", 3)]
    )
    kept, removed, within = remove_duplicates(df)
    assert kept["value"].tolist() == [2, 3] and removed == 1
    assert within == 0  # across two batches: an update


def test_the_same_customer_twice_in_one_batch_is_counted_as_a_within_batch_duplicate():
    df = frame(
        [("a", "2026-01-01", "b1", 1), ("a", "2026-01-01", "b1", 2), ("b", "2026-01-01", "b1", 3)]
    )
    kept, removed, within = remove_duplicates(df)
    assert (
        removed == 1 and within == 1 and kept["value"].tolist() == [2, 3]
    )  # the later row wins ties


def test_kinds_are_counted_separately_in_a_mixed_frame():
    df = frame(
        [
            ("a", "2026-01-01", "b1", 1),
            ("a", "2026-02-01", "b2", 2),  # across batches
            ("c", "2026-02-01", "b2", 3),
            ("c", "2026-02-01", "b2", 4),  # inside one batch
            ("d", "2026-03-01", "b3", 5),
            ("d", "2026-01-01", "b1", 6),  # across batches, older second
        ]
    )
    kept, removed, within = remove_duplicates(df)
    assert removed == 3 and within == 1
    assert sorted(kept["value"].tolist()) == [2, 4, 5]
