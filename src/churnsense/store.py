"""Inference log and feedback store: JSON lines, one file per day and kind.

``predictions-YYYY-MM-DD.jsonl`` holds one line per scored customer (the inputs, the answer, the
model). ``feedback-YYYY-MM-DD.jsonl`` holds one line per reported outcome, with the inputs and
the prediction copied in, so a feedback line is complete on its own (no join is ever needed).

Customer ids are stored only as a salted hash, so a log file does not reveal who the customers
are, while the same id still maps to the same hash for feedback.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INDEX_LIMIT = 20_000
MIN_LABELLED = 30


class UnknownPredictionError(LookupError):
    """Feedback refers to a prediction that was never made (or is too old)."""


def _day_files(directory: Path, kind: str) -> list[Path]:
    return sorted(directory.glob(f"{kind}-*.jsonl")) if directory.is_dir() else []


def read_recent(directory: Path, kind: str, n: int) -> list[dict[str, Any]]:
    """The ``n`` newest records of a kind, oldest first. Unreadable lines are skipped."""
    records: list[dict[str, Any]] = []
    for path in reversed(_day_files(directory, kind)):
        lines = path.read_text("utf-8").splitlines()
        for line in reversed(lines):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
            if len(records) >= n:
                return records[::-1]
    return records[::-1]


class LogStore:
    """Append-only store with a small in-memory index for feedback look-ups."""

    def __init__(self, directory: Path | str, salt: str = "dev-salt") -> None:
        self.directory = Path(directory)
        self.salt = salt
        self._lock = threading.Lock()
        self._by_request: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._by_cid: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def hash_id(self, customer_id: str | None) -> str | None:
        """Salted hash of a customer id (16 hex characters), or None for no id."""
        if not customer_id:
            return None
        return hashlib.sha256(f"{self.salt}:{customer_id}".encode()).hexdigest()[:16]

    def _append(self, kind: str, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{kind}-{datetime.now(UTC):%Y-%m-%d}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _remember(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._by_request[record["request_id"]] = record
            if record.get("cid"):
                self._by_cid[record["cid"]] = record
            while len(self._by_request) > INDEX_LIMIT:
                _, old = self._by_request.popitem(last=False)
                if old.get("cid") and self._by_cid.get(old["cid"]) is old:
                    del self._by_cid[old["cid"]]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for record in read_recent(self.directory, "predictions", INDEX_LIMIT):
            self._remember(record)

    def log_prediction(self, record: dict[str, Any]) -> None:
        """Append one prediction (``request_id`` is required) and index it."""
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **record}
        self._ensure_loaded()
        self._append("predictions", record)
        self._remember(record)

    def find(self, request_id: str | None = None, customer_id: str | None = None) -> dict | None:
        """The prediction a feedback refers to: by request id, else by (hashed) customer id."""
        self._ensure_loaded()
        with self._lock:
            if request_id and request_id in self._by_request:
                return self._by_request[request_id]
            cid = self.hash_id(customer_id)
            return self._by_cid.get(cid) if cid else None

    def log_feedback(
        self,
        *,
        churned: bool,
        request_id: str | None = None,
        customer_id: str | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        """Record the real outcome of a scored customer.

        Raises:
            UnknownPredictionError: if no matching prediction exists.
        """
        found = self.find(request_id, customer_id)
        if found is None:
            raise UnknownPredictionError("no prediction matches that request id or customer id")
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "cid": found.get("cid"),
            "request_id": found["request_id"],
            "churned": int(churned),
            "probability": found["probability"],
            "band": found["band"],
            "run_id": found["run_id"],
            "version": found.get("version"),
            "inputs": found["inputs"],
            "source": source,
        }
        self._append("feedback", record)
        return record

    def performance(self, last_n: int = 1000, min_samples: int = MIN_LABELLED) -> dict[str, Any]:
        """Accuracy, precision and recall of the HIGH band against reported outcomes.

        "Predicted churn" means the customer was put in the HIGH band. With fewer than
        ``min_samples`` labelled outcomes the numbers are None: too few to mean anything.
        """
        return performance_of(read_recent(self.directory, "feedback", last_n), min_samples)


def performance_of(
    records: list[dict[str, Any]], min_samples: int = MIN_LABELLED
) -> dict[str, Any]:
    """Confusion counts and rates for a list of feedback records."""
    tp = sum(1 for r in records if r["band"] == "HIGH" and r["churned"] == 1)
    fp = sum(1 for r in records if r["band"] == "HIGH" and r["churned"] == 0)
    fn = sum(1 for r in records if r["band"] != "HIGH" and r["churned"] == 1)
    tn = len(records) - tp - fp - fn
    enough = len(records) >= min_samples
    return {
        "n": len(records),
        "min_samples": min_samples,
        "enough_data": enough,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(records) if enough else None,
        "precision": tp / (tp + fp) if enough and (tp + fp) else None,
        "recall": tp / (tp + fn) if enough and (tp + fn) else None,
        "predicted_positive": "HIGH band",
    }
