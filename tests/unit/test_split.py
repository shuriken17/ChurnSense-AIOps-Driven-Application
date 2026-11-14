"""Unit tests for the split stage (T-SP-01)."""

from __future__ import annotations

import pandas as pd
import pytest

from churnsense.pipeline import split
from churnsense.tracking import StageError
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit


def test_t_sp_01_sizes_stratification_and_no_overlap(params):
    df = canonical_rows(500)
    frames = split.split_frames(df, params)
    sizes = {name: len(frame) for name, frame in frames.items()}
    assert sizes == {"train_real": 350, "val": 75, "test_golden": 75, "test_recent": 0}

    overall = df["churn"].mean()
    for name in ("train_real", "val", "test_golden"):
        assert abs(frames[name]["churn"].mean() - overall) < 0.01, name  # within one point
    split.check_no_overlap(frames)
    ids = pd.concat([f["customer_id"] for f in frames.values()])
    assert ids.is_unique and len(ids) == 500


def test_split_is_repeatable_and_depends_on_the_seed(params):
    df = canonical_rows(500)
    first = split.split_frames(df, params)
    again = split.split_frames(df, params)
    pd.testing.assert_frame_equal(first["test_golden"], again["test_golden"])
    params["seed"] = 7
    other = split.split_frames(df, params)
    assert set(other["test_golden"]["customer_id"]) != set(first["test_golden"]["customer_id"])


def test_with_no_feedback_the_recent_test_set_is_empty_but_well_formed(params):
    frames = split.split_frames(canonical_rows(500), params)
    recent = frames["test_recent"]
    assert recent.empty and list(recent.columns) == list(frames["val"].columns)
    assert recent.dtypes.astype(str).to_dict() == frames["val"].dtypes.astype(str).to_dict()


def feedback_frame(prefix: str, n: int, stamp: str, batch: str) -> pd.DataFrame:
    frame = canonical_rows(n, seed=3)
    frame["customer_id"] = [f"{prefix}-{i}" for i in range(n)]
    frame["source"] = "feedback"
    frame["batch_id"] = batch
    frame["event_ts"] = pd.Timestamp(stamp, tz="UTC")
    frame["event_ts"] = frame["event_ts"].astype("datetime64[us, UTC]")
    return frame


def test_recent_holdout_comes_from_the_newest_feedback_batch(params):
    real = canonical_rows(500)
    old = feedback_frame("old", 100, "2026-01-01", "batch_old")
    new = feedback_frame("new", 100, "2026-02-01", "batch_new")
    params.split["recent_window_min_rows"] = 1  # a window of exactly the newest batch
    frames = split.split_frames(pd.concat([real, old, new], ignore_index=True), params)

    recent = frames["test_recent"]
    assert len(recent) == 30 and recent["customer_id"].str.startswith("new-").all()
    train_ids = set(frames["train_real"]["customer_id"])
    assert {f"old-{i}" for i in range(100)} <= train_ids  # older feedback trains the model
    assert sum(c.startswith("new-") for c in train_ids) == 70  # the rest of the newest batch
    assert len(frames["train_real"]) == 350 + 100 + 70
    split.check_no_overlap(frames)


def test_tiny_or_one_class_feedback_still_splits(params):
    real = canonical_rows(500)
    tiny = feedback_frame("t", 3, "2026-02-01", "tiny")
    frames = split.split_frames(pd.concat([real, tiny], ignore_index=True), params)
    assert (
        len(frames["test_recent"])
        + sum(c.startswith("t-") for c in frames["train_real"]["customer_id"])
        == 3
    )


def test_overlap_between_splits_is_detected():
    frame = canonical_rows(5)
    with pytest.raises(StageError, match="both"):
        split.check_no_overlap({"a": frame, "b": frame.head(1)})


def test_stage_writes_four_files_and_a_report(workspace, params):
    validated = canonical_rows(500)
    workspace.ensure_parents(workspace.validated)
    validated.to_parquet(workspace.validated, index=False)
    report = split.run(params, workspace)
    assert report.rows_in == 500 and report.rows_out == 500
    for name in split.SPLIT_NAMES:
        assert workspace.split_file(name).is_file()
    assert report.extra["max_stratification_gap"] < 0.01


def test_a_tiny_newest_batch_is_pooled_with_older_ones_so_the_recent_set_is_usable(params):
    real = canonical_rows(500)
    first = feedback_frame("big", 500, "2026-01-01", "batch_big")
    second = feedback_frame("mid", 500, "2026-01-15", "batch_mid")
    tiny = feedback_frame("tiny", 3, "2026-02-01", "batch_tiny")
    frames = split.split_frames(pd.concat([real, first, second, tiny], ignore_index=True), params)
    recent = frames["test_recent"]
    assert 295 <= len(recent) <= 305  # 30% of the pooled 1,003 rows, not 30% of 3
    assert recent["customer_id"].str.startswith(("big-", "mid-", "tiny-")).all()
    split.check_no_overlap(frames)
    assert (
        len(frames["train_real"]) + len(recent) + len(frames["val"]) + len(frames["test_golden"])
        == 1503
    )


def test_pooling_stops_as_soon_as_there_are_enough_rows(params):
    real = canonical_rows(500)
    parts = [feedback_frame(f"b{i}", 60, f"2026-0{i + 1}-01", f"batch_{i}") for i in range(4)]
    params.split["recent_window_min_rows"] = 100
    frames = split.split_frames(pd.concat([real, *parts], ignore_index=True), params)
    recent_ids = frames["test_recent"]["customer_id"]
    assert recent_ids.str.startswith(("b3-", "b2-")).all()  # the two newest batches, 120 rows
    assert len(recent_ids) == 36
    assert (
        frames["train_real"]["customer_id"].str.startswith(("b0-", "b1-")).sum() == 120
    )  # older ones train
