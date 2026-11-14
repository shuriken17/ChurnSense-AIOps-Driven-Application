"""Integration tests for the ingest stage with Ray (T-IN-02, T-IN-03)."""

from __future__ import annotations

import pandas as pd
import pytest

from churnsense.pipeline import ingest
from churnsense.pipeline.ingest import IntegrityError
from churnsense.schema import INGESTED_COLUMNS
from churnsense.standardise import standardise_frame
from churnsense.tracking import StageError
from tests.conftest import raw_frame
from tests.factories import write_feedback_batch

pytestmark = pytest.mark.integration


def test_ray_ingest_equals_a_single_process_run(ray_cluster, workspace, params):
    report = ingest.run(params, workspace)
    assert report.rows_in == 500 and report.rows_out == 500
    assert report.extra["rows_by_source"] == {"real": 500}
    assert report.extra["raw_rows"] == 500 and report.extra["duplicates_removed"] == 0

    produced = pd.read_parquet(workspace.ingested)
    expected = standardise_frame(raw_frame(), source="real")
    assert list(produced.columns) == list(INGESTED_COLUMNS)
    pd.testing.assert_frame_equal(
        produced.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert "ray_rows_per_second" in report.extra and report.extra["ray_version"]


def test_a_changed_raw_file_is_refused(ray_cluster, workspace, params):
    workspace.raw_csv.write_bytes(workspace.raw_csv.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        ingest.run(params, workspace)


def test_missing_raw_file_and_missing_source_note_are_clear_errors(workspace, params):
    workspace.source_md.write_text("no checksum here", "utf-8")
    with pytest.raises(IntegrityError, match="SHA-256"):
        ingest.verify_raw(workspace.raw_csv, workspace.source_md)
    workspace.raw_csv.unlink()
    with pytest.raises(IntegrityError, match="not found"):
        ingest.verify_raw(workspace.raw_csv, workspace.source_md)


def test_wrong_row_count_fails_the_stage(ray_cluster, workspace, params):
    params.ingest["raw_expected_rows"] = 499
    with pytest.raises(StageError, match="expected 499"):
        ingest.run(params, workspace)


def test_a_missing_feedback_folder_is_valid(ray_cluster, workspace, params):
    (workspace.feedback_dir / "manifest.json").unlink()
    workspace.feedback_dir.rmdir()
    assert ingest.run(params, workspace).rows_out == 500


def test_t_in_02_corrupt_feedback_batch_is_excluded_and_reported(ray_cluster, workspace, params):
    write_feedback_batch(
        workspace, "good.parquet", 400, created_at="2026-01-01T00:00:00", prefix="g"
    )
    write_feedback_batch(
        workspace, "bad.parquet", 10, created_at="2026-01-02T00:00:00", prefix="b", corrupt=True
    )
    report = ingest.run(params, workspace)
    assert report.extra["rows_by_source"] == {"real": 500, "feedback": 400}
    quarantined = report.extra["feedback_batches_quarantined"]
    assert [q["file"] for q in quarantined] == ["bad.parquet"]
    assert quarantined[0]["reason"] == "checksum mismatch"
    ids = pd.read_parquet(workspace.ingested)["customer_id"]
    assert not ids.str.startswith("b-").any()  # nothing from the corrupt batch got in


def test_t_in_02_mostly_corrupt_feedback_fails_the_stage(ray_cluster, workspace, params):
    write_feedback_batch(
        workspace, "bad.parquet", 50, created_at="2026-01-01T00:00:00", prefix="b", corrupt=True
    )
    with pytest.raises(StageError, match="quarantined"):
        ingest.run(params, workspace)


def test_a_file_not_in_the_manifest_is_quarantined(ray_cluster, workspace, params):
    write_feedback_batch(
        workspace, "good.parquet", 400, created_at="2026-01-01T00:00:00", prefix="g"
    )
    write_feedback_batch(
        workspace,
        "stray.parquet",
        5,
        created_at="2026-01-02T00:00:00",
        prefix="s",
        list_in_manifest=False,
    )
    report = ingest.run(params, workspace)
    reasons = {q["file"]: q["reason"] for q in report.extra["feedback_batches_quarantined"]}
    assert reasons == {"stray.parquet": "not listed in manifest.json"}


def test_manifest_row_count_must_match(ray_cluster, workspace, params):
    write_feedback_batch(
        workspace, "good.parquet", 400, created_at="2026-01-01T00:00:00", prefix="g"
    )
    manifest = workspace.feedback_dir / "manifest.json"
    manifest.write_text(manifest.read_text("utf-8").replace('"rows": 400', '"rows": 401'), "utf-8")
    accepted, quarantined = ingest.verify_feedback(workspace.feedback_dir)
    assert not accepted and "differs from manifest" in quarantined[0]["reason"]


def test_broken_manifest_is_an_integrity_error(workspace):
    (workspace.feedback_dir / "manifest.json").write_text("{not json", "utf-8")
    with pytest.raises(IntegrityError, match="not a valid manifest"):
        ingest.read_manifest(workspace.feedback_dir)


def test_t_in_03_raw_plus_two_feedback_batches_with_one_duplicate(ray_cluster, workspace, params):
    first = write_feedback_batch(
        workspace, "batch1.parquet", 100, created_at="2026-01-01T00:00:00", prefix="f", seed=1
    )
    shared_id = first["customer_id"].iloc[0]
    ids = [shared_id] + [f"h-{i}" for i in range(1, 100)]
    write_feedback_batch(
        workspace,
        "batch2.parquet",
        100,
        created_at="2026-02-01T00:00:00",
        prefix="h",
        seed=2,
        ids=ids,
    )
    report = ingest.run(params, workspace)
    assert report.rows_in == 700 and report.rows_out == 699
    assert (
        report.extra["duplicates_removed"] == 1 and report.extra["duplicates_across_batches"] == 1
    )
    assert report.extra["rows_by_source"] == {"real": 500, "feedback": 199}
    assert report.extra["feedback_batches_accepted"] == ["batch1.parquet", "batch2.parquet"]

    out = pd.read_parquet(workspace.ingested)
    kept = out[out["customer_id"] == shared_id]
    assert len(kept) == 1 and kept.iloc[0]["batch_id"] == "batch2"  # the later row won
    assert kept.iloc[0]["event_ts"] == pd.Timestamp("2026-02-01", tz="UTC")
    assert set(out["source"]) == {"real", "feedback"}


def test_the_same_customer_in_two_batches_is_an_update_not_an_error(ray_cluster, workspace, params):
    first = write_feedback_batch(
        workspace, "batch1.parquet", 100, created_at="2026-01-01T00:00:00", prefix="f", seed=1
    )
    write_feedback_batch(
        workspace,
        "batch2.parquet",
        100,
        created_at="2026-02-01T00:00:00",
        prefix="x",
        seed=2,
        ids=list(first["customer_id"]),
    )  # 100% of batch 2 repeats batch 1's customers: reported again later
    report = ingest.run(params, workspace)
    assert (
        report.extra["duplicates_removed"] == 100 and report.extra["duplicates_within_batch"] == 0
    )
    assert report.extra["duplicates_across_batches"] == 100
    out = pd.read_parquet(workspace.ingested)
    assert (
        out["customer_id"].is_unique
        and (out[out["source"] == "feedback"]["batch_id"] == "batch2").all()
    )


def test_duplicates_inside_one_batch_fail_the_stage(ray_cluster, workspace, params):
    write_feedback_batch(
        workspace,
        "broken.parquet",
        100,
        created_at="2026-01-01T00:00:00",
        prefix="f",
        seed=1,
        ids=["same-customer"] * 100,
    )
    with pytest.raises(StageError, match="duplicates inside one batch"):
        ingest.run(params, workspace)
