"""Integration test: live feedback becomes a batch that the real pipeline accepts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churnsense.feedback import batcher
from churnsense.pipeline import ingest, split, validate
from churnsense.schema import INPUT_COLUMNS
from churnsense.store import LogStore
from tests.conftest import make_params, make_workspace
from tests.factories import canonical_rows

pytestmark = pytest.mark.integration
N_FEEDBACK = 450


def fill_store(store: LogStore) -> pd.DataFrame:
    """Predictions and outcomes for 450 customers (outcomes follow the real labels)."""
    rows = canonical_rows(N_FEEDBACK, seed=11).reset_index(drop=True)
    inputs = json.loads(
        rows[[c for c in INPUT_COLUMNS if c != "customer_id"]].to_json(orient="records")
    )
    for i, row in enumerate(inputs):
        who = f"F-{i}"
        store.log_prediction(
            {
                "request_id": f"r{i}",
                "cid": store.hash_id(who),
                "inputs": row,
                "probability": 0.5,
                "band": "HIGH",
                "run_id": "m",
                "version": "v1",
            }
        )
        store.log_feedback(churned=bool(rows["churn"].iloc[i]), customer_id=who, source="simulator")
    return rows


def test_feedback_flows_through_ingest_validate_and_split(ray_cluster, tmp_path):
    paths, params = make_workspace(tmp_path / "repo"), make_params()
    store = LogStore(tmp_path / "logs", "salt")
    rows = fill_store(store)
    entry = batcher.build_batch(paths, store.directory, tmp_path / "state", scenario="live")
    assert entry["rows"] == N_FEEDBACK and entry["synthetic_labels"] is True

    ingested = ingest.run(params, paths)
    assert ingested.extra["rows_by_source"] == {"real": 500, "feedback": N_FEEDBACK}
    assert ingested.extra["feedback_batches_accepted"] == [entry["file"]]
    assert ingested.extra["feedback_batches_quarantined"] == []

    checked = validate.run(params, paths)
    assert checked.rows_out == 500 + N_FEEDBACK and checked.extra["rows_rejected"] == 0

    split.run(params, paths)
    recent = pd.read_parquet(paths.split_file("test_recent"))
    assert len(recent) == round(0.3 * N_FEEDBACK) and recent["source"].eq("feedback").all()
    assert recent["churn"].mean() == pytest.approx(rows["churn"].mean(), abs=0.1)
    train = pd.read_parquet(paths.split_file("train_real"))
    assert (train["source"] == "feedback").sum() == N_FEEDBACK - len(recent)  # the rest trains


def test_a_tampered_batch_is_excluded_by_the_checksum(ray_cluster, tmp_path):
    paths, params = make_workspace(tmp_path / "repo"), make_params()
    store = LogStore(tmp_path / "logs", "salt")
    fill_store(store)
    entry = batcher.build_batch(paths, store.directory, tmp_path / "state")
    with (paths.feedback_dir / entry["file"]).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(Exception, match="quarantined"):  # all feedback rows are untrusted
        ingest.run(params, paths)
