"""Fixtures for the serving tests: a real exported model folder (built once)."""

from __future__ import annotations

import shutil

import pytest

from churnsense.deploy import export
from tests.factories import export_test_model


@pytest.fixture(scope="session")
def model_root(tmp_path_factory):
    """``serving/`` with v1 (logreg) and v2 (lightgbm); ``current`` points to v1."""
    root = tmp_path_factory.mktemp("serving_base") / "serving"
    export_test_model(root, 1, "logreg", "run-v1")
    export_test_model(root, 2, "lightgbm", "run-v2")
    export.switch_current(root, 1)
    return root


@pytest.fixture
def serving(model_root, tmp_path):
    """A private copy of the serving folder that a test may change."""
    target = tmp_path / "serving"
    shutil.copytree(model_root, target, symlinks=True)
    return target
