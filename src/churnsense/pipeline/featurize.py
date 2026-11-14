"""Stage 5: turn rows into model-ready features, with Ray Data.

Inputs:  the four split files and the synthetic file.
Outputs: data/processed/{train,val,test_golden,test_recent}.parquet,
         models/reference_profile.json, reports/featurize_stats.json

``train`` is the real (and feedback) training rows followed by the synthetic rows (when they
are switched on). Every processed file keeps ``customer_id``, ``source`` and ``churn`` next to
the 26 features, so later stages can filter by source (for example for the ablation run
without synthetic data).

The features come from ``churnsense.features.derive.derive_features``, the SAME function the
model uses at serving time. Ray only runs it on many chunks at once.

Checks: no NaN or infinite number, no missing category (an unseen category would be a bug in
the data), and exactly the expected column list.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import ray

from churnsense.config import Params, Paths
from churnsense.features.derive import derive_features
from churnsense.features.profile import build_reference_profile, save_profile
from churnsense.frames import to_native_dtypes
from churnsense.logging_utils import get_logger
from churnsense.raycfg import ray_session
from churnsense.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    categorical_dtype,
)
from churnsense.tracking import StageError, StageReport, tracked_stage

log = get_logger(__name__)

KEEP_COLUMNS = ("customer_id", "source", "churn")
PROCESSED_COLUMNS = (*KEEP_COLUMNS, *FEATURE_COLUMNS)


def featurize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``customer_id``, ``source``, ``churn`` and the 26 features of ``df``."""
    features = derive_features(df).reset_index(drop=True)
    # Plain object strings, so the Ray path and the pandas path give the same dtypes.
    keep = df[list(KEEP_COLUMNS)].reset_index(drop=True)
    keep = keep.astype({"customer_id": "object", "source": "object"})
    return pd.concat([keep, features], axis=1)


def featurize_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Ray ``map_batches`` adapter for ``featurize_frame``."""
    return featurize_frame(batch)


def restore_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the fixed categorical dtypes (a trip through Ray may relabel categories)."""
    out = df.copy()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype("object").astype(categorical_dtype(column))
    return out


def check_model_ready(df: pd.DataFrame, name: str) -> None:
    """Raise ``StageError`` unless the frame is exactly what the model expects."""
    if list(df.columns) != list(PROCESSED_COLUMNS):
        raise StageError(f"{name}: unexpected columns {list(df.columns)}")
    if len(df) == 0:
        return
    numeric = df[list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise StageError(f"{name}: contains NaN or infinite numbers")
    missing = df[list(CATEGORICAL_FEATURES)].isna().sum()
    if int(missing.sum()):
        bad = {c: int(n) for c, n in missing.items() if n}
        raise StageError(f"{name}: missing or unseen categories: {bad}")


def featurize_with_ray(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Featurize a frame in parallel blocks. An empty frame skips Ray and stays empty."""
    if len(df) == 0:
        return restore_dtypes(featurize_frame(df))
    dataset = ray.data.from_pandas(df).repartition(int(params.ray.num_cpus))
    dataset = dataset.map_batches(
        featurize_batch, batch_format="pandas", batch_size=int(params.ray.batch_size)
    )
    return restore_dtypes(to_native_dtypes(dataset.to_pandas()))


@tracked_stage("featurize", report="featurize_stats.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the featurize stage."""
    train_real = pd.read_parquet(paths.split_file("train_real"))
    synthetic = pd.read_parquet(paths.synthetic)
    inputs = {
        "train": pd.concat([train_real, synthetic], ignore_index=True),
        "val": pd.read_parquet(paths.split_file("val")),
        "test_golden": pd.read_parquet(paths.split_file("test_golden")),
        "test_recent": pd.read_parquet(paths.split_file("test_recent")),
    }

    outputs: dict[str, pd.DataFrame] = {}
    seconds: dict[str, float] = {}
    with ray_session(params):
        for name, frame in inputs.items():
            started = time.perf_counter()
            outputs[name] = featurize_with_ray(frame, params)
            seconds[name] = time.perf_counter() - started

    paths.ensure_parents(paths.processed_file("train"))
    for name, frame in outputs.items():
        check_model_ready(frame, name)
        frame.to_parquet(paths.processed_file(name), index=False)

    # The reference profile uses REAL training rows only (never synthetic ones).
    train = outputs["train"]
    real_rows = train[train["source"] == "real"]
    profile = build_reference_profile(real_rows[list(FEATURE_COLUMNS)])
    save_profile(paths.reference_profile, profile)

    rows = {name: len(frame) for name, frame in outputs.items()}
    rate = {
        name: round(rows[name] / seconds[name], 1) if seconds[name] > 0 else 0.0 for name in rows
    }
    total_rows = sum(rows.values())
    total_seconds = sum(seconds.values())
    return StageReport(
        rows_in=sum(len(f) for f in inputs.values()),
        rows_out=total_rows,
        extra={
            "rows_by_split": rows,
            "seconds_by_split": {k: round(v, 3) for k, v in seconds.items()},
            "rows_per_second_by_split": rate,
            "overall_rows_per_second": round(total_rows / total_seconds, 1)
            if total_seconds
            else 0.0,
            "feature_count": len(FEATURE_COLUMNS),
            "reference_profile_rows": profile["n_rows"],
        },
    )


if __name__ == "__main__":
    run.main()
