"""Unit tests for the stage wrapper: reports, metrics push, exit codes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from churnsense import tracking
from churnsense.config import ConfigError
from churnsense.tracking import StageError, StageReport, tracked_stage

pytestmark = pytest.mark.unit


class Collector(BaseHTTPRequestHandler):
    """A tiny fake Pushgateway that remembers what it receives."""

    received: list[tuple[str, str, str]] = []

    def _remember(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        Collector.received.append((self.command, self.path, body))
        self.send_response(200)
        self.end_headers()

    do_POST = do_PUT = _remember  # noqa: N815 - names fixed by http.server

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def pushgateway(monkeypatch):
    Collector.received = []
    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PUSHGATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
    yield Collector
    server.shutdown()


def make_stage(behaviour):
    @tracked_stage("unit_demo", report="unit_demo_stats.json")
    def stage(params, paths):
        return behaviour()

    return stage


def read_report(paths):
    return json.loads(paths.report("unit_demo_stats.json").read_text("utf-8"))


def test_success_writes_a_report_with_speed_numbers(workspace, params, monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    stage = make_stage(lambda: StageReport(rows_in=10, rows_out=8, extra={"answer": 42}))
    stage(params, workspace)
    report = read_report(workspace)
    assert report["status"] == "ok" and report["stage"] == "unit_demo"
    assert (report["rows_in"], report["rows_out"], report["answer"]) == (10, 8, 42)
    assert report["duration_seconds"] >= 0 and "rows_per_second" in report


def test_failure_keeps_the_diagnostics_and_reraises(workspace, params):
    def fail():
        raise StageError("data is bad", StageReport(rows_in=5, extra={"culprit": "x"}))

    with pytest.raises(StageError):
        make_stage(fail)(params, workspace)
    report = read_report(workspace)
    assert report["status"] == "failed" and report["error"] == "data is bad"
    assert report["rows_in"] == 5 and report["culprit"] == "x"


def test_crash_is_reported_with_its_type(workspace, params):
    def crash():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        make_stage(crash)(params, workspace)
    assert "ValueError: boom" in read_report(workspace)["error"]


def test_metrics_are_pushed_with_pushadd_and_a_stage_label(workspace, params, pushgateway):
    make_stage(lambda: StageReport(rows_in=3, rows_out=3))(params, workspace)
    method, path, body = pushgateway.received[-1]
    assert method == "POST"  # POST (pushadd) keeps older metrics of the group
    assert "/job/churnsense_pipeline/stage/unit_demo" in path
    for name in (
        "churnsense_stage_duration_seconds",
        "churnsense_stage_rows_total",
        "churnsense_stage_rows_per_second",
        "churnsense_stage_last_success_timestamp",
    ):
        assert name in body
    assert "churnsense_stage_last_status 1.0" in body


def test_failed_run_pushes_status_zero_and_no_success_timestamp(workspace, params, pushgateway):
    def fail():
        raise StageError("bad")

    with pytest.raises(StageError):
        make_stage(fail)(params, workspace)
    body = pushgateway.received[-1][2]
    assert "churnsense_stage_last_status 0.0" in body
    assert "last_success_timestamp" not in body


def test_an_unreachable_pushgateway_never_breaks_the_stage(workspace, params, monkeypatch):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://127.0.0.1:9")  # nothing listens there
    stage = make_stage(lambda: StageReport(rows_in=1, rows_out=1))
    stage(params, workspace)  # must not raise
    assert read_report(workspace)["status"] == "ok"
    assert tracking.push_stage_metrics("x", ok=True, duration=1.0, rows=1) is False


def test_no_gateway_configured_means_no_push(monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    assert tracking.push_stage_metrics("x", ok=True, duration=1.0, rows=1) is False


def test_main_exit_codes(workspace, params, monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    monkeypatch.setattr(tracking, "load_params", lambda: params)
    monkeypatch.setattr(tracking, "default_paths", lambda: workspace)

    make_stage(lambda: StageReport()).main()  # success: returns normally

    def fail():
        raise StageError("nope")

    with pytest.raises(SystemExit) as failure:
        make_stage(fail).main()
    assert failure.value.code == 1

    def bad_config():
        raise ConfigError("no params")

    monkeypatch.setattr(tracking, "load_params", bad_config)
    with pytest.raises(SystemExit) as config:
        make_stage(lambda: StageReport()).main()
    assert config.value.code == 2
