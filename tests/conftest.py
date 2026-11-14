"""Shared fixtures for the ChurnSense tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from churnsense.config import Params, Paths, load_params

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CSV = REPO / "tests/fixtures/telco_500.csv"
FIXTURE_ROWS = 500


@pytest.fixture(scope="session")
def ray_cluster():
    """Start one small local Ray for the whole test session."""
    import ray

    from churnsense.raycfg import ensure_workers_can_import

    ensure_workers_can_import()  # pytest adds src/ only to this process, not to Ray workers
    ray.init(
        num_cpus=2,
        object_store_memory=200 * 1024 * 1024,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="WARNING",
    )
    yield
    ray.shutdown()


def make_params() -> Params:
    """The real params.yaml, adjusted so that a 500-row fixture passes every guardrail."""
    p = load_params(REPO / "params.yaml")
    p.ingest["raw_expected_rows"] = FIXTURE_ROWS
    p.validation["min_rows"] = 100
    p.ray["num_cpus"] = 2
    p.ray["object_store_mb"] = 200
    p.synthetic["n_rows"] = 6000
    p.synthetic["max_ks_stat"] = 0.10  # a 350-row real sample is a noisy yardstick
    return p


@pytest.fixture
def params() -> Params:
    """A fresh copy of the fixture-sized parameters for every test."""
    return make_params()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of some bytes."""
    return hashlib.sha256(data).hexdigest()


def make_workspace(root: Path) -> Paths:
    """Build a project folder holding the fixture CSV, its SOURCE.md and no feedback."""
    paths = Paths(root=root)
    paths.raw_csv.parent.mkdir(parents=True)
    shutil.copy(FIXTURE_CSV, paths.raw_csv)
    paths.source_md.write_text(
        f"# test data\n\nSHA-256: {sha256_bytes(paths.raw_csv.read_bytes())}\n", encoding="utf-8"
    )
    paths.feedback_dir.mkdir(parents=True)
    (paths.feedback_dir / "manifest.json").write_text(json.dumps({"batches": []}), "utf-8")
    return paths


@pytest.fixture
def workspace(tmp_path: Path) -> Paths:
    """A temporary project folder (one per test)."""
    return make_workspace(tmp_path)


def raw_frame() -> pd.DataFrame:
    """The fixture rows with the original IBM headers (all text)."""
    return pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=False)
