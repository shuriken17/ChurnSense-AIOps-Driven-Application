"""Stage 1: ingest the raw data with Ray Data.

Inputs:  data/raw/telco_churn.csv (7,043 real rows) and data/feedback/ (labelled batches).
Output:  data/interim/ingested.parquet, reports/ingest_stats.json.

What the stage guarantees (guardrails from the design document):

* the raw file must match the checksum and row count recorded in data/raw/SOURCE.md;
* every feedback batch must be listed in manifest.json and match its checksum and row
  count; a batch that does not is excluded ("quarantined") and listed in the report, and
  the stage fails if too many feedback rows are excluded;
* duplicate customers are removed (the latest ``event_ts`` wins), and the stage fails if more
  than ``ingest.max_duplicate_frac`` of rows were duplicates INSIDE one batch (the same customer
  in two different batches is a normal update, reported but not guarded);
* an empty feedback folder is perfectly valid.

Ray does the parsing and standardising in parallel blocks (``map_batches``); the result is
identical to a single-process run (tested).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import ray

from churnsense.config import Params, Paths
from churnsense.frames import to_native_dtypes
from churnsense.logging_utils import get_logger
from churnsense.raycfg import ray_session
from churnsense.schema import INGESTED_COLUMNS
from churnsense.standardise import standardise_batch
from churnsense.tracking import StageError, StageReport, tracked_stage

log = get_logger(__name__)

_SHA_PATTERN = re.compile(r"sha-?256\W+([0-9a-f]{64})", re.IGNORECASE)
MANIFEST = "manifest.json"


class IntegrityError(RuntimeError):
    """A file does not match the checksum or row count recorded for it."""


# --------------------------------------------------------------------------- integrity
def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256(source_md: Path) -> str:
    """Read the raw file's checksum from SOURCE.md (a line like ``SHA-256: <64 hex>``)."""
    try:
        text = source_md.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise IntegrityError(f"cannot find {source_md}") from exc
    match = _SHA_PATTERN.search(text)
    if not match:
        raise IntegrityError(f"{source_md} has no 'SHA-256: <64 hex digits>' line")
    return match.group(1).lower()


def verify_raw(csv_path: Path, source_md: Path) -> str:
    """Check the raw CSV against SOURCE.md; return its checksum.

    Raises:
        IntegrityError: if the file is missing or its checksum differs.
    """
    if not csv_path.is_file():
        raise IntegrityError(f"raw file not found: {csv_path} (run scripts/download_data.sh)")
    actual = sha256_of(csv_path)
    wanted = expected_sha256(source_md)
    if actual != wanted:
        raise IntegrityError(
            f"checksum mismatch for {csv_path.name}: "
            f"expected {wanted[:12]}..., got {actual[:12]}..."
        )
    return actual


@dataclass(frozen=True)
class FeedbackBatch:
    """One accepted feedback batch (a parquet file listed in the manifest)."""

    name: str
    path: Path
    rows: int
    scenario: str
    synthetic_labels: bool
    created_at: str


def read_manifest(feedback_dir: Path) -> list[dict[str, Any]]:
    """Return the list of batch entries of ``manifest.json`` (empty if there is none)."""
    manifest = feedback_dir / MANIFEST
    if not manifest.is_file():
        return []
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8")).get("batches", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise IntegrityError(f"{manifest} is not a valid manifest: {exc}") from exc
    if not isinstance(entries, list):
        raise IntegrityError(f"{manifest}: 'batches' must be a list")
    return entries


def verify_feedback(feedback_dir: Path) -> tuple[list[FeedbackBatch], list[dict[str, Any]]]:
    """Check every feedback file; return (accepted batches, quarantined entries).

    A parquet file that is not listed in the manifest is quarantined too: an unknown file
    must not silently become training data.
    """
    accepted: list[FeedbackBatch] = []
    quarantined: list[dict[str, Any]] = []
    listed: set[str] = set()

    for entry in read_manifest(feedback_dir):
        name = str(entry.get("file", ""))
        listed.add(name)
        path = feedback_dir / name
        rows_claimed = int(entry.get("rows", -1))
        if not path.is_file():
            quarantined.append({"file": name, "rows": rows_claimed, "reason": "file is missing"})
            continue
        if sha256_of(path) != str(entry.get("sha256", "")).lower():
            quarantined.append({"file": name, "rows": rows_claimed, "reason": "checksum mismatch"})
            continue
        try:
            rows_actual = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:  # noqa: BLE001 - any unreadable file is quarantined
            quarantined.append(
                {"file": name, "rows": rows_claimed, "reason": f"unreadable parquet: {exc}"}
            )
            continue
        if rows_actual != rows_claimed:
            quarantined.append(
                {
                    "file": name,
                    "rows": rows_actual,
                    "reason": f"row count {rows_actual} differs from manifest {rows_claimed}",
                }
            )
            continue
        accepted.append(
            FeedbackBatch(
                name=name,
                path=path,
                rows=rows_actual,
                scenario=str(entry.get("scenario", "")),
                synthetic_labels=bool(entry.get("synthetic_labels", False)),
                created_at=str(entry.get("created_at", "")),
            )
        )

    for path in sorted(feedback_dir.glob("*.parquet")) if feedback_dir.is_dir() else []:
        if path.name not in listed:
            quarantined.append(
                {
                    "file": path.name,
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "reason": "not listed in manifest.json",
                }
            )
    return accepted, quarantined


# --------------------------------------------------------------------------- Ray reading
def _read_raw_with_ray(csv_path: Path, params: Params) -> pd.DataFrame:
    """Read and standardise the raw CSV in parallel blocks."""
    # TotalCharges holds " " for new customers; read it as text and convert afterwards.
    options = pacsv.ConvertOptions(column_types={"TotalCharges": pa.string()})
    dataset = ray.data.read_csv(str(csv_path), convert_options=options)
    dataset = dataset.repartition(int(params.ray.num_cpus))
    dataset = dataset.map_batches(
        standardise_batch,
        batch_format="pandas",
        batch_size=int(params.ray.batch_size),
        fn_kwargs={"source": "real", "event_ts": "", "batch_id": "raw"},
    )
    return to_native_dtypes(dataset.to_pandas())


def _standardise_feedback(batch: pd.DataFrame, event_ts: str, batch_id: str) -> pd.DataFrame:
    """Ray adapter: force the lineage of feedback rows, then standardise."""
    kept = batch.drop(columns=[c for c in ("source", "batch_id") if c in batch.columns])
    return standardise_batch(kept, "feedback", event_ts, batch_id)


def _read_feedback_with_ray(batch: FeedbackBatch, params: Params) -> pd.DataFrame:
    """Read and standardise one accepted feedback batch."""
    dataset = ray.data.read_parquet(str(batch.path))
    dataset = dataset.map_batches(
        _standardise_feedback,
        batch_format="pandas",
        batch_size=int(params.ray.batch_size),
        fn_kwargs={"event_ts": batch.created_at, "batch_id": Path(batch.name).stem},
    )
    return to_native_dtypes(dataset.to_pandas())


# --------------------------------------------------------------------------- stage
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Keep one row per customer: the one with the latest ``event_ts`` (later row wins ties).

    Returns the de-duplicated frame in its original row order, the number of rows removed, and how
    many of those were duplicates INSIDE one batch (or inside the raw file).

    The two kinds mean different things. The same customer in two different batches is normal: a
    later report replaces an earlier one (that is why the latest wins). The same customer twice in
    ONE batch means the file was built wrongly, or ingested twice, so only that kind is guarded.
    """
    position = pd.Series(range(len(df)), index=df.index, name="_pos")
    ordered = pd.concat([df[["customer_id", "event_ts", "batch_id"]], position], axis=1)
    winners = ordered.sort_values(["event_ts", "_pos"], kind="stable").drop_duplicates(
        "customer_id", keep="last"
    )
    kept = df.iloc[sorted(winners["_pos"].tolist())].reset_index(drop=True)
    losers = ordered[~ordered["_pos"].isin(winners["_pos"])]
    winner_batch = winners.set_index("customer_id")["batch_id"]
    within = int((losers["batch_id"] == losers["customer_id"].map(winner_batch)).sum())
    return kept, len(df) - len(kept), within


@tracked_stage("ingest", report="ingest_stats.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the ingest stage and write ``data/interim/ingested.parquet``."""
    raw_sha = verify_raw(paths.raw_csv, paths.source_md)
    accepted, quarantined = verify_feedback(paths.feedback_dir)
    quarantined_rows = sum(max(int(q["rows"]), 0) for q in quarantined)
    feedback_rows = sum(b.rows for b in accepted)
    log.info(
        "integrity checks done",
        extra={"accepted_batches": len(accepted), "quarantined_batches": len(quarantined)},
    )

    with ray_session(params):
        started = time.perf_counter()  # measure the work, not Ray's start-up time
        parts = [_read_raw_with_ray(paths.raw_csv, params)]
        parts += [_read_feedback_with_ray(batch, params) for batch in accepted]
        ray_seconds = time.perf_counter() - started

    df = pd.concat(parts, ignore_index=True)
    rows_read = len(df)
    expected_raw = int(params.ingest.raw_expected_rows)
    raw_rows = len(parts[0])
    if raw_rows != expected_raw:
        raise StageError(f"raw file has {raw_rows} rows, expected {expected_raw}")

    df, removed, within_batch = remove_duplicates(df)
    extras = [c for c in df.columns if c not in INGESTED_COLUMNS]  # keep: validate reports drift
    df = df[[*INGESTED_COLUMNS, *extras]]

    paths.ensure_parents(paths.ingested)
    df.to_parquet(paths.ingested, index=False)

    by_source = {str(k): int(v) for k, v in df["source"].value_counts().items()}
    report = StageReport(
        rows_in=rows_read,
        rows_out=len(df),
        extra={
            "raw_sha256": raw_sha,
            "raw_rows": raw_rows,
            "feedback_rows_accepted": feedback_rows,
            "rows_by_source": by_source,
            "duplicates_removed": removed,
            "duplicates_within_batch": within_batch,
            "duplicates_across_batches": removed - within_batch,
            "feedback_batches_accepted": [b.name for b in accepted],
            "feedback_batches_quarantined": quarantined,
            "ray_seconds": round(ray_seconds, 3),
            "ray_rows_per_second": round(rows_read / ray_seconds, 1) if ray_seconds > 0 else 0.0,
            "ray_version": ray.__version__,
        },
    )

    if rows_read and within_batch / rows_read > float(params.ingest.max_duplicate_frac):
        raise StageError(
            f"{within_batch} of {rows_read} rows are duplicates inside one batch "
            f"(limit {params.ingest.max_duplicate_frac:.0%})",
            report,
        )
    total_feedback = feedback_rows + quarantined_rows
    if total_feedback and quarantined_rows / total_feedback > float(
        params.validation.max_quarantine_frac
    ):
        raise StageError(
            f"{quarantined_rows} of {total_feedback} feedback rows were quarantined "
            f"(limit {params.validation.max_quarantine_frac:.0%})",
            report,
        )
    return report


if __name__ == "__main__":
    run.main()
