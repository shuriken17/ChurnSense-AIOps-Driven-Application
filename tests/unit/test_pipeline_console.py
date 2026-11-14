"""Unit tests for the pipeline console: probes, stage and run endpoints, request queue."""

from __future__ import annotations

import json
import math
import os
import time

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from churnsense.api import backend as backend_module
from churnsense.api.backend import PIPELINE_EDGES, STAGE_REPORTS, create_app
from churnsense.api.components import ComponentProbe, targets_from_environment
from churnsense.controller import main as ctl
from churnsense.deploy import export
from churnsense.serving.model_server import create_app as model_app
from churnsense.store import LogStore
from tests.conftest import REPO, make_params

pytestmark = pytest.mark.unit
TOKEN = {"x-admin-token": "secret"}


# ------------------------------------------------------------------ component probes
def probe_client(down=()):
    def handler(request):
        if any(name in str(request.url) for name in down):
            raise httpx.ConnectError("down")
        if "broken" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_every_component_gets_a_health_url_that_can_be_overridden(monkeypatch):
    targets = targets_from_environment()
    assert set(targets) == {
        "model-server",
        "drift-monitor",
        "mlflow",
        "nginx",
        "prometheus",
        "grafana",
        "pushgateway",
    }
    assert (
        targets["mlflow"] == "http://mlflow:5000/health"
        and targets["nginx"] == "http://nginx/healthz"
    )
    monkeypatch.setenv("MLFLOW_URL", "http://elsewhere:9000/")
    assert targets_from_environment()["mlflow"] == "http://elsewhere:9000/health"


def test_probing_reports_up_down_and_server_errors(tmp_path):
    targets = {"a": "http://a/health", "b": "http://b/health", "broken": "http://broken/health"}
    probe = ComponentProbe(targets, tmp_path, probe_client(down=("://b/",)))
    assert probe.probe_once() == {"a": 1.0, "b": 0.0, "broken": 0.0}


def test_the_heartbeat_is_the_age_of_the_controller_status_file(tmp_path):
    probe = ComponentProbe({}, tmp_path)
    assert math.isnan(probe.heartbeat_age())
    status = tmp_path / "controller_status.json"
    status.write_text("{}")
    os.utime(status, (time.time() - 120, time.time() - 120))
    assert 119 <= probe.heartbeat_age() <= 125


def test_the_background_probe_starts_and_stops(tmp_path):
    probe = ComponentProbe({"a": "http://a/h"}, tmp_path, probe_client(), interval=0.05)
    probe.start()
    deadline = time.time() + 3
    while not probe.results and time.time() < deadline:
        time.sleep(0.02)
    probe.stop()
    assert probe.results == {"a": 1.0}


# ------------------------------------------------------------------ the endpoints
@pytest.fixture
def console(serving, tmp_path):
    reports, state = tmp_path / "reports", tmp_path / "state"
    reports.mkdir()
    (state / "runs").mkdir(parents=True)
    probe = ComponentProbe(
        {"model-server": "http://m/health", "mlflow": "http://mlflow/health"},
        state,
        probe_client(down=("mlflow",)),
    )
    with TestClient(model_app(serving, "secret")) as model:
        yield (
            TestClient(
                create_app(
                    model, "secret", LogStore(tmp_path / "logs"), reports, state, components=probe
                )
            ),
            reports,
            state,
        )


def test_components_endpoint_answers_even_before_the_first_background_round(console):
    client, _, state = console
    body = client.get("/api/v1/components").json()
    assert body["components"] == {"model-server": {"up": True}, "mlflow": {"up": False}}
    assert body["controller_heartbeat_age_seconds"] is None
    (state / "controller_status.json").write_text("{}")
    assert client.get("/api/v1/components").json()["controller_heartbeat_age_seconds"] < 5


def test_probe_results_are_published_as_metrics(console):
    client, _, _ = console
    client.get("/api/v1/components")
    text = client.get("/api/v1/metrics").text
    assert 'churnsense_dependency_up{name="model-server"} 1.0' in text
    assert 'churnsense_dependency_up{name="mlflow"} 0.0' in text
    assert "churnsense_controller_heartbeat_age_seconds" in text


def test_stages_list_all_nine_with_speed_and_say_when_one_has_not_run(console):
    client, reports, _ = console
    (reports / "ingest_stats.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "duration_seconds": 11.3,
                "rows_in": 7043,
                "rows_out": 7043,
                "rows_per_second": 624.7,
            }
        )
    )
    (reports / "gate_decision.json").write_text(json.dumps({"status": "ok", "decision": "PROMOTE"}))
    (reports / "train_metrics.json").write_text(
        json.dumps({"status": "failed", "error": "boom", "winner": "logreg"})
    )
    body = client.get("/api/v1/pipeline/stages").json()
    stages = {s["stage"]: s for s in body["stages"]}
    assert [s["stage"] for s in body["stages"]] == [name for name, _ in STAGE_REPORTS]
    assert stages["ingest"]["rows_per_second"] == 624.7 and stages["ingest"]["status"] == "ok"
    assert stages["gate"]["detail"] == "PROMOTE" and stages["train"]["status"] == "failed"
    assert stages["train"]["error"] == "boom" and stages["split"] == {
        "stage": "split",
        "status": "not run",
    }
    assert body["edges"] == [list(e) for e in PIPELINE_EDGES]


def test_the_graph_in_the_api_is_exactly_the_dvc_graph():
    dvc = yaml.safe_load((REPO / "dvc.yaml").read_text("utf-8"))["stages"]
    produced = {}
    for name, stage in dvc.items():
        for kind in ("outs", "metrics", "plots"):
            for out in stage.get(kind, []):
                produced[out if isinstance(out, str) else next(iter(out))] = name
    edges = set()
    for name, stage in dvc.items():
        for dep in stage["deps"]:
            owner = next(
                (
                    stage_name
                    for path, stage_name in produced.items()
                    if dep == path or dep.startswith(path + "/")
                ),
                None,
            )
            if owner and owner != name:
                edges.add((owner, name))
    assert edges == set(PIPELINE_EDGES)
    assert [n for n, _ in STAGE_REPORTS] == list(dvc)


def test_runs_come_newest_first_with_their_steps_and_waiting_requests(console):
    client, _, state = console
    for name, outcome in (("20260101T000000Z", "failed"), ("20260102T000000Z", "promoted")):
        (state / "runs" / f"{name}.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "outcome": outcome,
                    "steps": [{"step": "dvc_repro", "ok": outcome != "failed", "detail": "x"}],
                }
            )
        )
    (state / "runs" / "broken.json").write_text("{not json")
    (state / "requests").mkdir()
    (state / "requests" / "a.json").write_text(json.dumps({"action": "rollback"}))
    body = client.get("/api/v1/pipeline/runs").json()
    assert [r["outcome"] for r in body["runs"]] == ["promoted", "failed"]
    assert body["pending_requests"] == ["rollback"] and body["runs"][1]["steps"][0]["ok"] is False
    assert len(client.get("/api/v1/pipeline/runs?limit=1").json()["runs"]) == 1


def test_a_request_needs_the_admin_token_and_a_known_action(console):
    client, _, state = console
    assert (
        client.post("/api/v1/admin/pipeline/request", json={"action": "retrain"}).status_code == 401
    )
    assert (
        client.post(
            "/api/v1/admin/pipeline/request",
            json={"action": "retrain"},
            headers={"x-admin-token": "no"},
        ).status_code
        == 401
    )
    bad = client.post(
        "/api/v1/admin/pipeline/request", json={"action": "format-disk"}, headers=TOKEN
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"
    assert not (state / "requests").exists() or not list((state / "requests").glob("*.json"))


def test_an_accepted_request_is_queued_once_per_action(console):
    client, _, state = console
    first = client.post("/api/v1/admin/pipeline/request", json={"action": "retrain"}, headers=TOKEN)
    assert first.status_code == 202 and first.json()["status"] == "accepted"
    queued = list((state / "requests").glob("*.json"))
    assert len(queued) == 1 and json.loads(queued[0].read_text())["action"] == "retrain"
    again = client.post("/api/v1/admin/pipeline/request", json={"action": "retrain"}, headers=TOKEN)
    assert again.status_code == 409 and again.json()["error"]["code"] == "already_requested"
    other = client.post(
        "/api/v1/admin/pipeline/request", json={"action": "rollback"}, headers=TOKEN
    )
    assert other.status_code == 202 and len(list((state / "requests").glob("*.json"))) == 2
    waiting = client.get("/api/v1/pipeline/runs").json()["pending_requests"]
    assert sorted(waiting) == ["retrain", "rollback"]


def test_admin_requests_are_disabled_without_a_configured_token(serving, tmp_path):
    with TestClient(model_app(serving, "secret")) as model:
        client = TestClient(
            create_app(model, "", LogStore(tmp_path / "l"), tmp_path / "r", tmp_path / "s")
        )
        assert (
            client.post(
                "/api/v1/admin/pipeline/request", json={"action": "retrain"}, headers=TOKEN
            ).status_code
            == 503
        )


# ------------------------------------------------------------------ the controller side
@pytest.fixture
def queue(tmp_path):
    state = tmp_path / "state"
    (state / "requests").mkdir(parents=True)
    return state


def request(state, name, action):
    (state / "requests" / f"{name}.json").write_text(json.dumps({"action": action}))


def test_requests_are_taken_oldest_first_and_deleted_before_they_run(queue, monkeypatch, tmp_path):
    seen = []

    def fake_execute(paths, cfg, decision, log_dir, state_dir):
        seen.append((decision.trigger, sorted(p.name for p in (queue / "requests").glob("*.json"))))
        return {"outcome": "rejected", "trigger": decision.trigger}

    monkeypatch.setattr(ctl, "execute", fake_execute)
    monkeypatch.setattr(
        ctl,
        "run_rollback",
        lambda state_dir: seen.append(("rollback", None)) or {"outcome": "rolled_back"},
    )
    request(queue, "1-first", "retrain")
    request(queue, "2-second", "rollback")
    done = ctl.process_requests(None, make_params(), tmp_path / "logs", queue)
    assert [d["outcome"] for d in done] == ["rejected", "rolled_back"]
    assert [s[0] for s in seen] == ["manual-ui", "rollback"]
    assert seen[0][1] == ["2-second.json"]  # its own file was already gone while it ran
    assert not list((queue / "requests").glob("*.json"))


def test_a_busy_controller_records_a_skipped_request_instead_of_failing(
    queue, monkeypatch, tmp_path
):
    def busy(*args):
        raise ctl.BusyError("busy")

    monkeypatch.setattr(ctl, "execute", busy)
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    request(queue, "1", "retrain")
    (record,) = ctl.process_requests(None, make_params(), tmp_path / "logs", queue)
    assert record["outcome"] == "skipped" and "already running" in record["steps"][0]["detail"]
    assert ctl.read_state(queue)["runs"]["skipped"] == 1
    assert list((queue / "runs").glob("*.json"))


def test_garbage_and_unknown_requests_are_discarded_without_running_anything(
    queue, monkeypatch, tmp_path
):
    monkeypatch.setattr(ctl, "execute", lambda *a: pytest.fail("must not run"))
    monkeypatch.setattr(ctl, "run_rollback", lambda *a: pytest.fail("must not run"))
    (queue / "requests" / "1.json").write_text("{not json")
    request(queue, "2", "format-disk")
    assert ctl.process_requests(None, make_params(), tmp_path / "logs", queue) == []
    assert not list((queue / "requests").glob("*.json"))


def test_a_rollback_request_rolls_back_and_is_recorded(serving, tmp_path, monkeypatch):
    export.switch_current(serving, 2)
    monkeypatch.setenv("SERVING_DIR", str(serving))
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)

    class Registry:
        def __init__(self, tracking_uri):
            pass

        def set_registered_model_alias(self, *args):
            pass

    monkeypatch.setattr("mlflow.MlflowClient", Registry)
    state = tmp_path / "state"
    record = ctl.run_rollback(state)
    assert record["outcome"] == "rolled_back" and record["trigger"] == "manual-ui"
    assert "v2 -> v1" in record["steps"][0]["detail"] and export.current_version(serving) == 1
    assert ctl.read_state(state)["runs"]["rolled_back"] == 1


def test_a_failed_rollback_is_recorded_not_raised(serving, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SERVING_DIR", str(serving)
    )  # current is v1, the oldest: nothing to go back to
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    record = ctl.run_rollback(tmp_path / "state")
    assert record["outcome"] == "failed" and "oldest" in record["steps"][0]["detail"]


def test_a_rollback_never_runs_during_a_retraining(serving, tmp_path, monkeypatch):
    export.switch_current(serving, 2)
    monkeypatch.setenv("SERVING_DIR", str(serving))
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    state = tmp_path / "state"
    with ctl.retrain_lock(state):
        record = ctl.run_rollback(state)
    assert record["outcome"] == "skipped" and export.current_version(serving) == 2


def test_the_loop_serves_requests_every_round_but_checks_the_policy_once_a_minute(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    (state / "requests").mkdir(parents=True)
    calls = {"requests": 0, "gather": 0}
    monkeypatch.setattr(
        ctl,
        "process_requests",
        lambda *a: calls.__setitem__("requests", calls["requests"] + 1) or [],
    )
    monkeypatch.setattr(
        ctl,
        "gather",
        lambda *a: (
            calls.__setitem__("gather", calls["gather"] + 1)
            or ctl.Signals("ok", [], None, 0, time.time())
        ),
    )
    rounds = []

    def sleep(_):
        rounds.append(1)
        if len(rounds) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(ctl.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        ctl.loop(None, make_params(), "d", "b", tmp_path / "logs", state)
    assert calls == {"requests": 3, "gather": 1}
    assert (state / "controller_status.json").is_file()


def test_the_backend_module_exposes_the_documented_actions():
    assert backend_module.REQUEST_ACTIONS == ("retrain", "rollback")
