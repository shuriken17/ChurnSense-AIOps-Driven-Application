"""Turn reported outcomes (feedback) into one labelled batch file for the data pipeline.

Each call collects the feedback received since the last call, writes
``data/feedback/batch_<time>.parquet`` (the 20 input fields plus ``churn``) and registers it in
``data/feedback/manifest.json`` with its checksum. The ingest stage verifies that checksum, so a
damaged or hand-edited batch is excluded instead of silently trained on.

Honesty rule: a batch whose labels came from the traffic simulator is marked
``synthetic_labels: true`` in the manifest. Real customers' outcomes are not simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from churnsense.config import Paths, default_paths
from churnsense.schema import INPUT_COLUMNS
from churnsense.store import read_recent

CURSOR_NAME = "feedback_cursor.json"
MAX_RECORDS = 1_000_000


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text("utf-8")) if path.is_file() else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_batch(
    paths: Paths, log_dir: Path, state_dir: Path, scenario: str = "live"
) -> dict[str, Any] | None:
    """Write a batch of the not-yet-batched feedback; return its manifest entry (or None)."""
    cursor_file = state_dir / CURSOR_NAME
    cursor = _read_json(cursor_file, {"after": ""})["after"]
    records = [r for r in read_recent(log_dir, "feedback", MAX_RECORDS) if r["ts"] > cursor]
    if not records:
        return None
    rows = []
    for r in records:
        who = f"live-{r['cid']}" if r.get("cid") else f"req-{r['request_id']}"
        rows.append({"customer_id": who, **r["inputs"], "churn": int(r["churned"]), "_ts": r["ts"]})
    frame = pd.DataFrame(rows).sort_values("_ts").drop_duplicates("customer_id", keep="last")
    frame = frame.drop(columns="_ts")
    if "total_charges" not in frame:
        frame["total_charges"] = float("nan")
    frame = frame[[*INPUT_COLUMNS, "churn"]].astype(
        {
            "senior_citizen": "int64",
            "tenure_months": "int64",
            "churn": "int64",
            "monthly_charges": "float64",
            "total_charges": "float64",
        }
    )
    frame = frame.reset_index(drop=True)

    created = datetime.now(UTC)
    paths.feedback_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = paths.feedback_dir / "manifest.json"
    manifest = _read_json(manifest_file, {"batches": []})
    # the sequence number keeps two batches made in the same second from sharing a file
    name = f"batch_{created:%Y%m%dT%H%M%S}_{len(manifest['batches']) + 1:03d}.parquet"
    target = paths.feedback_dir / name
    frame.to_parquet(target, index=False)
    entry = {
        "file": name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "scenario": scenario,
        "synthetic_labels": any(r.get("source") == "simulator" for r in records),
        "created_at": created.isoformat(timespec="seconds"),
    }
    manifest["batches"].append(entry)
    _write_json(manifest_file, manifest)
    _write_json(cursor_file, {"after": max(r["ts"] for r in records)})
    return entry


def repair_manifest(paths: Paths) -> list[dict[str, Any]]:
    """Re-register batch files that are missing from ``manifest.json``; return what was added.

    The manifest is the integrity record of the batches: a batch file it does not list is refused
    by the ingest stage. If the manifest is lost or overwritten, this command lists the files again
    with a fresh checksum. That is an OPERATOR decision (the checksum can no longer prove the file
    is unchanged), so every re-registered entry is marked ``recovered`` and, because its origin is
    no longer recorded, ``synthetic_labels: true`` (the conservative choice). Entries are ordered
    by creation time, because the newest batch feeds the recent test set.
    """
    manifest_file = paths.feedback_dir / "manifest.json"
    manifest = _read_json(manifest_file, {"batches": []})
    listed = {b["file"] for b in manifest["batches"]}
    added = []
    for path in sorted(paths.feedback_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime):
        if path.name in listed:
            continue
        added.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": int(pq.ParquetFile(path).metadata.num_rows),
                "scenario": "recovered",
                "synthetic_labels": True,
                "recovered": True,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
                    timespec="seconds"
                ),
            }
        )
    if added:
        manifest["batches"] = sorted([*manifest["batches"], *added], key=lambda b: b["created_at"])
        _write_json(manifest_file, manifest)
    return added


def main() -> None:
    """Command line: ``python -m churnsense.feedback.batcher [--scenario NAME]``."""
    parser = argparse.ArgumentParser(description="Make a labelled batch from new feedback.")
    parser.add_argument("--scenario", default="live")
    parser.add_argument(
        "--repair", action="store_true", help="re-register batch files missing from manifest.json"
    )
    args = parser.parse_args()
    if args.repair:
        added = repair_manifest(default_paths())
        print(json.dumps(added) if added else "nothing to repair: every batch file is listed")
        return
    entry = build_batch(
        default_paths(),
        Path(os.environ.get("INFERENCE_LOG_DIR", "/srv/churnsense/inference-logs")),
        Path(os.environ.get("STATE_DIR", "/srv/churnsense/state")),
        args.scenario,
    )
    print(json.dumps(entry) if entry else "no new feedback: nothing to batch")


if __name__ == "__main__":
    main()
