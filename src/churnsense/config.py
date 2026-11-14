"""Configuration: parameters from params.yaml and every file path of the pipeline.

Keeping paths in one dataclass means a stage never builds a path by hand, and tests can
point the whole pipeline at a temporary folder by passing a different ``Paths`` object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Top-level keys that params.yaml must contain (checked when the file is loaded).
REQUIRED_KEYS = ("seed", "ray", "ingest", "validation", "split", "synthetic", "features")


class ConfigError(RuntimeError):
    """Raised when params.yaml is missing, unreadable or incomplete."""


class Params(dict):
    """A dictionary whose keys can also be read as attributes (params.ray.num_cpus)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"params has no key {name!r}") from exc

    @classmethod
    def wrap(cls, value: Any) -> Any:
        """Recursively turn nested dictionaries into Params objects."""
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value


def repo_root() -> Path:
    """Return the repository root: $CHURNSENSE_ROOT if set, else the current directory."""
    return Path(os.environ.get("CHURNSENSE_ROOT", Path.cwd())).resolve()


def load_params(
    path: str | Path | None = None, required: tuple[str, ...] = REQUIRED_KEYS
) -> Params:
    """Read params.yaml and check that the expected sections exist.

    Args:
        path: file to read; defaults to ``params.yaml`` in the repository root.
        required: top-level keys that must be present.

    Raises:
        ConfigError: if the file is missing, not valid YAML, or lacks a required key.
    """
    file = Path(path) if path else repo_root() / "params.yaml"
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"params file not found: {file}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"params file is not valid YAML: {file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"params file must contain a mapping: {file}")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"params file {file} is missing sections: {missing}")
    return Params.wrap(raw)


@dataclass(frozen=True)
class Paths:
    """All file locations used by the data pipeline, relative to ``root``."""

    root: Path

    # ---- inputs (tracked by DVC) ----
    @property
    def raw_csv(self) -> Path:
        """The IBM Telco CSV."""
        return self.root / "data/raw/telco_churn.csv"

    @property
    def source_md(self) -> Path:
        """Document that records the raw file's checksum and origin."""
        return self.root / "data/raw/SOURCE.md"

    @property
    def feedback_dir(self) -> Path:
        """Folder of labelled feedback batches plus manifest.json."""
        return self.root / "data/feedback"

    # ---- intermediate data ----
    @property
    def interim(self) -> Path:
        """Folder for intermediate parquet files."""
        return self.root / "data/interim"

    @property
    def ingested(self) -> Path:
        """Output of the ingest stage."""
        return self.interim / "ingested.parquet"

    @property
    def validated(self) -> Path:
        """Output of the validate stage."""
        return self.interim / "validated.parquet"

    @property
    def splits_dir(self) -> Path:
        """Folder of the four split files."""
        return self.interim / "splits"

    def split_file(self, name: str) -> Path:
        """Return the parquet file of one split (train_real, val, test_golden, test_recent)."""
        return self.splits_dir / f"{name}.parquet"

    @property
    def synthetic(self) -> Path:
        """Output of the synthesize stage."""
        return self.root / "data/synthetic/train_synthetic.parquet"

    @property
    def quarantine(self) -> Path:
        """Rows rejected by validation, with the reason."""
        return self.root / "data/quarantine/rejected.parquet"

    @property
    def processed_dir(self) -> Path:
        """Folder of model-ready feature files."""
        return self.root / "data/processed"

    def processed_file(self, name: str) -> Path:
        """Return the featurized parquet file of one split (train, val, test_golden, ...)."""
        return self.processed_dir / f"{name}.parquet"

    # ---- reports and models ----
    @property
    def reports(self) -> Path:
        """Folder of JSON reports (small, committed to Git)."""
        return self.root / "reports"

    def report(self, name: str) -> Path:
        """Return the path of one JSON report."""
        return self.reports / name

    @property
    def eda_dir(self) -> Path:
        """Folder of EDA plots."""
        return self.reports / "eda"

    @property
    def models_dir(self) -> Path:
        """Folder of model files (the DVC-tracked candidate and its run information)."""
        return self.root / "models"

    @property
    def candidate(self) -> Path:
        """The newly trained MLflow pyfunc model (the challenger)."""
        return self.models_dir / "candidate"

    @property
    def run_info(self) -> Path:
        """Small JSON about the training run: run id, model URI, git commit."""
        return self.models_dir / "run_info.json"

    @property
    def plots_dir(self) -> Path:
        """Folder of training plots and importance tables."""
        return self.reports / "plots"

    @property
    def reference_profile(self) -> Path:
        """Reference distribution of the training data, used by drift detection."""
        return self.root / "models/reference_profile.json"

    def ensure_parents(self, *files: Path) -> None:
        """Create the parent folder of every given file."""
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)


def serving_dir() -> Path:
    """Folder of exported models: ``$SERVING_DIR`` or ``/srv/churnsense/serving``."""
    return Path(os.environ.get("SERVING_DIR", "/srv/churnsense/serving"))


def default_paths(root: Path | None = None) -> Paths:
    """Return the standard ``Paths`` for the repository."""
    return Paths(root=root or repo_root())
