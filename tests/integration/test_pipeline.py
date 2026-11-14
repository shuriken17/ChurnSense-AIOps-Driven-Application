"""Integration tests: synthesize and featurize with Ray, and the whole chain (T-SY-01, T-FE-03)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churnsense.features.profile import load_profile
from churnsense.pipeline import featurize, ingest, split, synthesize, validate
from churnsense.pipeline.featurize import PROCESSED_COLUMNS
from churnsense.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from churnsense.tracking import StageError
from churnsense.validation import VALIDATED, validate_rows
from tests.conftest import make_params, make_workspace
from tests.factories import canonical_rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def built(ray_cluster, tmp_path_factory):
    """Run ingest -> validate -> split -> synthesize -> featurize once on the fixture."""
    paths = make_workspace(tmp_path_factory.mktemp("built"))
    params = make_params()
    for stage in (ingest, validate, split, synthesize, featurize):
        stage.run(params, paths)
    return paths, params


def test_whole_chain_produces_the_expected_files_and_sizes(built):
    paths, _ = built
    sizes = {
        n: len(pd.read_parquet(paths.processed_file(n)))
        for n in ("train", "val", "test_golden", "test_recent")
    }
    assert sizes == {"train": 350 + 6000, "val": 75, "test_golden": 75, "test_recent": 0}
    train = pd.read_parquet(paths.processed_file("train"))
    assert list(train.columns) == list(PROCESSED_COLUMNS)
    assert train["source"].value_counts().to_dict() == {"synthetic": 6000, "real": 350}
    assert not train.isna().any().any()
    for column in CATEGORICAL_FEATURES:
        assert str(train[column].dtype) == "category"


def test_reference_profile_uses_real_training_rows_only(built):
    paths, _ = built
    profile = load_profile(paths.reference_profile)
    assert profile["n_rows"] == 350  # not 6,350: synthetic rows are excluded
    assert profile["feature_columns"] == list(FEATURE_COLUMNS)


def test_every_stage_left_a_report_with_speed_numbers(built):
    paths, _ = built
    for name in (
        "ingest_stats",
        "validation_report",
        "split_report",
        "synthetic_stats",
        "featurize_stats",
    ):
        report = json.loads(paths.report(f"{name}.json").read_text("utf-8"))
        assert report["status"] == "ok" and report["duration_seconds"] >= 0, name
    stats = json.loads(paths.report("featurize_stats.json").read_text("utf-8"))
    assert stats["rows_by_split"]["train"] == 6350 and stats["overall_rows_per_second"] > 0


# ------------------------------------------------------------------ T-SY-01 synthetic data
def test_t_sy_01_synthetic_rows_are_valid_and_look_like_the_real_ones(built):
    paths, params = built
    synthetic = pd.read_parquet(paths.synthetic)
    assert len(synthetic) == 6000
    assert validate_rows(synthetic, VALIDATED).rejected.empty  # same schema rules as real data
    assert synthetic["customer_id"].is_unique
    assert synthetic["customer_id"].str.startswith(synthesize.SYNTHETIC_ID_PREFIX).all()
    assert set(synthetic["source"]) == {"synthetic"}

    real = pd.read_parquet(paths.split_file("train_real"))
    assert abs(synthetic["churn"].mean() - real["churn"].mean()) < 0.01  # within one point
    stats = json.loads(paths.report("synthetic_stats.json").read_text("utf-8"))
    assert stats["max_ks_statistic"] < params.synthetic.max_ks_stat
    assert stats["max_category_gap"] < params.synthetic.max_category_gap
    assert not set(synthetic["customer_id"]) & set(real["customer_id"])


def test_t_sy_01_same_seed_gives_identical_rows_and_another_seed_does_not(ray_cluster):
    params = make_params()
    real = canonical_rows(350)

    def fingerprint(df):
        return int(pd.util.hash_pandas_object(df, index=False).sum())

    first = synthesize.generate_synthetic(real, params)
    second = synthesize.generate_synthetic(real, params)
    assert fingerprint(first) == fingerprint(second)
    params["seed"] = 43
    assert fingerprint(synthesize.generate_synthetic(real, params)) != fingerprint(first)


def test_chunks_cover_all_rows():
    assert sum(synthesize.chunk_sizes(200_000, 12)) == 200_000
    assert sorted(set(synthesize.chunk_sizes(10, 3))) == [3, 4]


def test_cross_field_rules_are_enforced_on_broken_input():
    df = canonical_rows(20)
    df["phone_service"] = "No"
    df["multiple_lines"] = "Yes"
    df["internet_service"] = "No"
    df["online_security"] = "Yes"
    fixed = synthesize.enforce_cross_field(df)
    assert (fixed["multiple_lines"] == "No phone service").all()
    assert (fixed["online_security"] == "No internet service").all()


def test_synthetic_data_can_be_switched_off(ray_cluster, workspace, params):
    for stage in (ingest, validate, split):
        stage.run(params, workspace)
    params.synthetic["enabled"] = False
    report = synthesize.run(params, workspace)
    assert report.rows_out == 0 and report.extra["enabled"] is False
    assert pd.read_parquet(workspace.synthetic).empty
    featurize.run(params, workspace)  # the featurize stage copes with no synthetic rows
    assert len(pd.read_parquet(workspace.processed_file("train"))) == 350


def test_quality_guardrail_stops_poor_synthetic_data(ray_cluster, workspace, params):
    for stage in (ingest, validate, split):
        stage.run(params, workspace)
    params.synthetic["max_ks_stat"] = 0.0001
    with pytest.raises(StageError, match="quality checks"):
        synthesize.run(params, workspace)
    assert '"status": "failed"' in workspace.report("synthetic_stats.json").read_text("utf-8")


# ------------------------------------------------------------------ T-FE-03 Ray equals pandas
def test_t_fe_03_ray_featurize_equals_plain_pandas(ray_cluster):
    params = make_params()
    frame = canonical_rows(500)
    with_ray = featurize.featurize_with_ray(frame, params)
    plain = featurize.restore_dtypes(featurize.featurize_frame(frame))
    pd.testing.assert_frame_equal(with_ray.reset_index(drop=True), plain.reset_index(drop=True))


def test_empty_frame_skips_ray_and_keeps_its_columns(ray_cluster):
    empty = featurize.featurize_with_ray(canonical_rows(10).iloc[0:0], make_params())
    assert empty.empty and list(empty.columns) == list(PROCESSED_COLUMNS)


def test_unseen_categories_or_nan_fail_the_model_ready_check():
    frame = featurize.restore_dtypes(featurize.featurize_frame(canonical_rows(20)))
    featurize.check_model_ready(frame, "ok")
    broken = frame.copy()
    broken["contract"] = broken["contract"].astype("object")
    broken.loc[0, "contract"] = "Weekly"
    broken["contract"] = broken["contract"].astype(frame["contract"].dtype)
    with pytest.raises(StageError, match="missing or unseen"):
        featurize.check_model_ready(broken, "x")
    with pytest.raises(StageError, match="NaN"):
        featurize.check_model_ready(frame.assign(monthly_charges=float("nan")), "x")
    with pytest.raises(StageError, match="unexpected columns"):
        featurize.check_model_ready(frame.drop(columns=["tenure_months"]), "x")
