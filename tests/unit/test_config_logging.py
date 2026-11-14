"""Unit tests for configuration and logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churnsense.config import ConfigError, Params, Paths, load_params, repo_root
from churnsense.logging_utils import get_logger, setup_logging
from tests.conftest import REPO

pytestmark = pytest.mark.unit


def test_real_params_file_loads_and_allows_attribute_access():
    params = load_params(REPO / "params.yaml")
    assert params.seed == 42
    assert params.ray.num_cpus == 3 and params.ray.object_store_mb == 800
    assert params.synthetic.n_rows == 200_000 and params.synthetic.n_chunks == 12
    assert params.gate.min_roc_auc == 0.80
    assert params.validation.churn_rate_bounds == [0.10, 0.45]


def test_missing_attribute_is_a_clear_error():
    with pytest.raises(AttributeError, match="nothing"):
        _ = Params({"a": 1}).nothing


def test_missing_file_bad_yaml_and_missing_sections_are_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_params(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: [1, 2", "utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_params(bad)
    partial = tmp_path / "partial.yaml"
    partial.write_text("seed: 1\n", "utf-8")
    with pytest.raises(ConfigError, match="missing sections"):
        load_params(partial)
    listy = tmp_path / "list.yaml"
    listy.write_text("- 1\n- 2\n", "utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_params(listy)


def test_repo_root_follows_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURNSENSE_ROOT", str(tmp_path))
    assert repo_root() == tmp_path.resolve()


def test_paths_live_under_the_root_and_create_their_folders(tmp_path):
    paths = Paths(root=tmp_path)
    for file in (
        paths.ingested,
        paths.validated,
        paths.synthetic,
        paths.quarantine,
        paths.split_file("val"),
        paths.processed_file("train"),
        paths.report("x.json"),
        paths.reference_profile,
    ):
        assert tmp_path in file.parents
    paths.ensure_parents(paths.processed_file("train"), paths.report("x.json"))
    assert paths.processed_dir.is_dir() and paths.reports.is_dir()


def test_logs_are_json_lines_with_extra_fields(capsys):
    setup_logging("INFO")
    setup_logging("INFO")  # calling twice must not duplicate output
    get_logger("churnsense.test").info("hello", extra={"stage": "ingest", "rows": 7})
    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["msg"] == "hello" and record["level"] == "INFO"
    assert record["stage"] == "ingest" and record["rows"] == 7 and "ts" in record


def test_exceptions_are_logged_with_a_traceback(capsys):
    setup_logging("INFO")
    try:
        raise ValueError("broken")
    except ValueError:
        get_logger("churnsense.test").exception("it failed")
    record = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert "ValueError: broken" in record["exception"]


def test_log_level_from_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    setup_logging()
    get_logger("churnsense.test").info("quiet")
    get_logger("churnsense.test").warning("loud")
    out = capsys.readouterr().err
    assert "quiet" not in out and "loud" in out
    setup_logging("INFO")


def test_paths_is_relative_to_current_folder_by_default():
    assert Path(Paths(root=Path(".")).raw_csv).as_posix() == "data/raw/telco_churn.csv"
