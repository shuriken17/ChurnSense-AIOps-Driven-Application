"""Runs the browser-level tests of the web interface (Node and jsdom), where they are available.

They load the real page and scripts in a simulated browser, answer the API calls with stubs and
click through every page. Node is not in the Docker images, so there the tests are skipped; run
them with ``cd tests/ui && npm install && npm test``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from churnsense.schema import field_catalogue
from tests.conftest import REPO

pytestmark = pytest.mark.unit
UI = REPO / "tests" / "ui"


def jsdom_path() -> str | None:
    for candidate in (
        os.environ.get("NODE_PATH", ""),
        "/tmp/jsd/node_modules",
        str(UI / "node_modules"),
        str(REPO / "node_modules"),
    ):
        for part in filter(None, candidate.split(os.pathsep)):
            if (Path(part) / "jsdom").is_dir():
                return part
    return None


def test_the_field_fixture_of_the_browser_tests_matches_the_real_schema():
    fixture = json.loads((UI / "fixtures" / "fields.json").read_text("utf-8"))
    assert fixture == json.loads(json.dumps(field_catalogue(), sort_keys=True)), (
        "regenerate tests/ui/fixtures/fields.json from churnsense.schema.field_catalogue()"
    )


@pytest.mark.skipif(
    shutil.which("node") is None or jsdom_path() is None, reason="Node or jsdom is not installed"
)
def test_the_web_interface_passes_its_browser_tests():
    done = subprocess.run(
        ["node", "--test"],
        cwd=UI,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "NODE_PATH": jsdom_path()},
    )
    summary = [line for line in done.stdout.splitlines() if line.startswith("# ")]
    assert done.returncode == 0, "\n".join(summary) + "\n" + done.stdout[-1500:]
    assert any(line.startswith("# pass") and int(line.split()[-1]) >= 35 for line in summary)
