"""Unit tests for the live smoke test and the shell scripts."""

from __future__ import annotations

import json
import shutil
import subprocess

import httpx
import pytest

from churnsense.deploy import smoke
from tests.conftest import REPO

pytestmark = pytest.mark.unit
ROW = {"input": {"gender": "Male"}, "expected": {"churn_probability": 0.3, "risk_band": "MEDIUM"}}


@pytest.fixture
def smoke_file(tmp_path):
    path = tmp_path / "smoke_rows.json"
    path.write_text(json.dumps({"rows": [ROW, ROW]}))
    return path


def fake_backend(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(
        smoke.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def answer(probability=0.3, band="MEDIUM", run="r1", status=200):
    return lambda request: httpx.Response(
        status, json={"churn_probability": probability, "risk_band": band, "model": {"run_id": run}}
    )


def test_smoke_test_passes_when_every_row_matches(monkeypatch, smoke_file):
    fake_backend(monkeypatch, answer())
    assert smoke.smoke_test("http://b", smoke_file, "r1") == {"rows": 2, "all_match": True}


@pytest.mark.parametrize(
    ("handler", "text"),
    [
        (answer(probability=0.31), "probability differs"),
        (answer(band="HIGH"), "band HIGH"),
        (answer(run="old"), "another model"),
        (answer(status=500), "HTTP 500"),
    ],
)
def test_smoke_test_names_the_problem(monkeypatch, smoke_file, handler, text):
    fake_backend(monkeypatch, handler)
    with pytest.raises(smoke.DeploymentError, match=text):
        smoke.smoke_test("http://b", smoke_file, "r1")


def test_an_unreachable_backend_is_a_deployment_error(monkeypatch, smoke_file):
    def refuse(request):
        raise httpx.ConnectError("down")

    fake_backend(monkeypatch, refuse)
    with pytest.raises(smoke.DeploymentError, match="not reachable"):
        smoke.smoke_test("http://b", smoke_file, "r1")


def test_reload_checks_that_the_expected_run_is_now_served(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(200, json={"status": "reloaded", "model": {"run_id": "other"}})

    monkeypatch.setattr(smoke.httpx, "post", post)
    with pytest.raises(smoke.DeploymentError, match="expected"):
        smoke.reload_model_server("http://m", "t", "wanted")
    monkeypatch.setattr(
        smoke.httpx,
        "post",
        lambda url, **kw: httpx.Response(200, json={"model": {"run_id": "wanted"}}),
    )
    assert smoke.reload_model_server("http://m", "t", "wanted")["run_id"] == "wanted"
    monkeypatch.setattr(
        smoke.httpx, "post", lambda url, **kw: httpx.Response(500, json={"error": "corrupt"})
    )
    with pytest.raises(smoke.DeploymentError, match="refused"):
        smoke.reload_model_server("http://m", "t", "wanted")


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_every_shell_script_passes_shellcheck():
    scripts = sorted(str(p) for p in (REPO / "scripts").glob("*.sh"))
    assert subprocess.run(["shellcheck", "-x", *scripts], capture_output=True).returncode == 0


def test_the_certificate_script_makes_a_usable_certificate_once(tmp_path):
    env = {"TLS_DIR": str(tmp_path / "tls"), "PATH": "/usr/bin:/bin"}
    first = subprocess.run(
        ["bash", str(REPO / "scripts/make_cert.sh")], capture_output=True, text=True, env=env
    )
    again = subprocess.run(
        ["bash", str(REPO / "scripts/make_cert.sh")], capture_output=True, text=True, env=env
    )
    assert "created" in first.stdout and "already exists" in again.stdout
    text = subprocess.run(
        ["openssl", "x509", "-in", str(tmp_path / "tls/tls.crt"), "-noout", "-text"],
        capture_output=True,
        text=True,
    ).stdout
    assert "DNS:localhost" in text and "IP Address:127.0.0.1" in text
