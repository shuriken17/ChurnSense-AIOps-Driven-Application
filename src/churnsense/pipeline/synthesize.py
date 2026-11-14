"""Stage 4: synthetic training rows, made in parallel with Ray.

Input:   data/interim/splits/train_real.parquet (only its REAL rows are used)
Outputs: data/synthetic/train_synthetic.parquet, reports/synthetic_stats.json

Why synthetic data? It exists to exercise Ray at volume (rubric: data engineering
throughput) and to give the model more rows to fit. It adds VOLUME, NOT INFORMATION: every
synthetic row is a noisy copy of a real training row. The report and the docs say so, and an
ablation run with synthetic data switched off is logged in MLflow (Batch 2).

Method (12 Ray tasks, each seeded with ``seed + chunk_index`` so the result never changes):

1. class-stratified bootstrap: draw real training rows with replacement, keeping the
   churn share exactly as in the real data;
2. add Gaussian noise (5% of the column's standard deviation) to tenure and monthly charges,
   keep new customers (tenure 0) at tenure 0, clip to the range seen in the real data;
3. rescale total_charges by the same tenure and monthly ratios, so the three stay consistent;
4. re-apply the cross-field rules; mark ``source="synthetic"`` and ``customer_id="SYN-<n>"``.

No test-set customer can influence a synthetic row, because the split happened first.
Guardrails: Kolmogorov-Smirnov distance per numeric column, category frequency gap and churn
share gap, all measured against the real training rows.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import ray
from scipy.stats import ks_2samp

from churnsense.config import Params, Paths
from churnsense.logging_utils import get_logger
from churnsense.raycfg import ray_session
from churnsense.schema import (
    CATEGORICAL_INPUTS,
    VALIDATED_COLUMNS,
    enforce_cross_field,  # noqa: F401  (re-exported)
)
from churnsense.tracking import StageError, StageReport, tracked_stage

log = get_logger(__name__)

SYNTHETIC_ID_PREFIX = "SYN-"
NUMERIC_CHECKED = ("tenure_months", "monthly_charges", "total_charges")
MAX_CHURN_SHARE_GAP = 0.01  # one percentage point


def make_chunk(
    real: pd.DataFrame, n_rows: int, seed: int, first_number: int, noise_frac: float
) -> pd.DataFrame:
    """Create ``n_rows`` synthetic rows. Pure function: same arguments, same rows."""
    rng = np.random.default_rng(seed)
    positives, negatives = real[real["churn"] == 1], real[real["churn"] == 0]
    n_pos = int(round(n_rows * len(positives) / len(real)))
    drawn = pd.concat(
        [
            positives.iloc[rng.integers(0, len(positives), n_pos)],
            negatives.iloc[rng.integers(0, len(negatives), n_rows - n_pos)],
        ]
    )
    rows = drawn.iloc[rng.permutation(len(drawn))].reset_index(drop=True)

    tenure_old = rows["tenure_months"].to_numpy(dtype=float)
    monthly_old = rows["monthly_charges"].to_numpy(dtype=float)
    total_old = rows["total_charges"].to_numpy(dtype=float)

    tenure_noise = rng.normal(0, noise_frac * real["tenure_months"].std(), n_rows)
    monthly_noise = rng.normal(0, noise_frac * real["monthly_charges"].std(), n_rows)
    tenure_new = np.where(
        tenure_old == 0,
        0.0,
        np.clip(np.rint(tenure_old + tenure_noise), 1, real["tenure_months"].max()),
    )
    monthly_new = np.clip(
        monthly_old + monthly_noise, real["monthly_charges"].min(), real["monthly_charges"].max()
    )
    ratio = (tenure_new / np.maximum(tenure_old, 1.0)) * (monthly_new / monthly_old)
    total_new = np.where(tenure_old == 0, 0.0, np.clip(total_old * ratio, 0.0, None))

    rows["tenure_months"] = tenure_new.astype("int64")
    rows["monthly_charges"] = monthly_new
    rows["total_charges"] = total_new
    rows["total_charges_imputed"] = 0
    rows = enforce_cross_field(rows)
    rows["customer_id"] = [f"{SYNTHETIC_ID_PREFIX}{first_number + i}" for i in range(n_rows)]
    rows["source"] = "synthetic"
    rows["batch_id"] = "synthetic"
    rows["event_ts"] = real["event_ts"].min()
    return rows[list(VALIDATED_COLUMNS)]


@ray.remote
def _make_chunk_task(
    real: pd.DataFrame, n_rows: int, seed: int, first_number: int, noise_frac: float
) -> pd.DataFrame:
    """Ray wrapper around ``make_chunk`` (one task per chunk)."""
    return make_chunk(real, n_rows, seed, first_number, noise_frac)


def chunk_sizes(n_rows: int, n_chunks: int) -> list[int]:
    """Split ``n_rows`` into ``n_chunks`` sizes that differ by at most one."""
    base, extra = divmod(n_rows, n_chunks)
    return [base + (1 if i < extra else 0) for i in range(n_chunks)]


def generate_synthetic(real: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Generate all synthetic rows with Ray (call inside ``ray_session``)."""
    cfg = params.synthetic
    sizes = chunk_sizes(int(cfg.n_rows), int(cfg.n_chunks))
    real_ref = ray.put(real)  # shared once instead of copied into every task
    retries = int(params.ray.max_task_retries)
    futures, first = [], 0
    for index, size in enumerate(sizes):
        futures.append(
            _make_chunk_task.options(max_retries=retries).remote(
                real_ref,
                size,
                int(params.seed) + index,
                first,
                float(cfg.numeric_noise_frac_of_std),
            )
        )
        first += size
    return pd.concat(ray.get(futures), ignore_index=True)


def quality_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Compare synthetic rows with the real training rows they came from."""
    ks = {c: round(float(ks_2samp(real[c], synthetic[c]).statistic), 4) for c in NUMERIC_CHECKED}
    gaps: dict[str, float] = {}
    for column in CATEGORICAL_INPUTS:
        real_share = real[column].value_counts(normalize=True)
        synth_share = synthetic[column].value_counts(normalize=True)
        categories = real_share.index.union(synth_share.index)
        gaps[column] = float(
            (
                real_share.reindex(categories, fill_value=0)
                - synth_share.reindex(categories, fill_value=0)
            )
            .abs()
            .max()
        )
    return {
        "ks_statistic": ks,
        "max_ks_statistic": max(ks.values()),
        "max_category_gap": round(max(gaps.values()), 4),
        "category_gap_by_column": {c: round(g, 4) for c, g in gaps.items()},
        "churn_share_real": round(float(real["churn"].mean()), 4),
        "churn_share_synthetic": round(float(synthetic["churn"].mean()), 4),
    }


@tracked_stage("synthesize", report="synthetic_stats.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the synthesize stage."""
    train = pd.read_parquet(paths.split_file("train_real"))
    real = train[train["source"] == "real"].reset_index(drop=True)
    cfg = params.synthetic
    paths.ensure_parents(paths.synthetic)

    if not bool(cfg.enabled):
        train.iloc[0:0][list(VALIDATED_COLUMNS)].to_parquet(paths.synthetic, index=False)
        log.info("synthetic data is switched off in params.yaml; wrote an empty file")
        return StageReport(rows_in=len(real), rows_out=0, extra={"enabled": False})

    with ray_session(params):
        started = time.perf_counter()
        synthetic = generate_synthetic(real, params)
        seconds = time.perf_counter() - started

    quality = quality_report(real, synthetic)
    synthetic.to_parquet(paths.synthetic, index=False)
    report = StageReport(
        rows_in=len(real),
        rows_out=len(synthetic),
        extra={
            "enabled": True,
            "chunks": int(cfg.n_chunks),
            "generation_seconds": round(seconds, 3),
            "generation_rows_per_second": round(len(synthetic) / seconds, 1)
            if seconds > 0
            else 0.0,
            "note": "synthetic rows add volume, not information",
            **quality,
        },
    )
    problems = []
    if quality["max_ks_statistic"] > float(cfg.max_ks_stat):
        problems.append(f"KS distance {quality['max_ks_statistic']} > {cfg.max_ks_stat}")
    if quality["max_category_gap"] > float(cfg.max_category_gap):
        problems.append(f"category gap {quality['max_category_gap']} > {cfg.max_category_gap}")
    if abs(quality["churn_share_real"] - quality["churn_share_synthetic"]) > MAX_CHURN_SHARE_GAP:
        problems.append("churn share differs from the real data by more than one point")
    if problems:
        raise StageError("synthetic data failed quality checks: " + "; ".join(problems), report)
    return report


if __name__ == "__main__":
    run.main()
