"""Unit tests for the retrain policy, the run sequence, the loop helpers and rollback."""

from __future__ import annotations

import fcntl
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from churnsense.controller import main as ctl
from churnsense.controller import runner
from churnsense.controller.policy import Signals, decide, fingerprint
from churnsense.deploy import export, rollback, smoke
from churnsense.store import LogStore
from tests.conftest import make_params, make_workspace
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit
NOW = 1_800_000_000.0
CFG = make_params().controller
WEAK = {"enough_data": True, "n": 150, "recall": 0.45}
GOOD = {"enough_data": True, "n": 150, "recall": 0.85}


def signals(**kw):
    base = {
        "drift_status": "ok",
        "drifted_features": [],
        "performance": GOOD,
        "new_feedback_rows": 500,
        "now": NOW,
    }
    return Signals(**{**base, **kw})


# ------------------------------------------------------------------ policy
def test_a_healthy_system_is_left_alone():
    d = decide(signals(), {}, CFG)
    assert d.trigger is None and d.reason.startswith("healthy")


def test_a_drift_alert_triggers_with_a_fingerprint_of_the_drifted_features():
    d = decide(
        signals(drift_status="alert", drifted_features=["contract", "tenure_months"]), {}, CFG
    )
    assert d.trigger == "drift" and "contract" in d.reason
    assert d.fingerprint == fingerprint(["tenure_months", "contract"], NOW)  # order does not matter


def test_weak_recall_triggers_only_with_enough_labelled_outcomes():
    assert decide(signals(performance=WEAK), {}, CFG).trigger == "performance"
    for perf in ({**WEAK, "n": 99}, {**WEAK, "enough_data": False}, {**WEAK, "recall": None}):
        assert decide(signals(performance=perf), {}, CFG).trigger is None
    assert (
        decide(signals(performance={**WEAK, "recall": 0.60}), {}, CFG).trigger is None
    )  # floor is strict


def test_drift_wins_over_performance_and_candidates_do_not_trigger():
    assert decide(signals(drift_status="alert", performance=WEAK), {}, CFG).trigger == "drift"
    for status in ("alert_candidate", "warn", "ok", "insufficient_data", None):
        assert decide(signals(drift_status=status), {}, CFG).trigger is None


def test_silent_monitors_are_no_signal_not_a_trigger():
    assert decide(signals(drift_status=None, performance=None), {}, CFG).trigger is None


def test_the_cooldown_blocks_a_second_run_and_then_expires():
    s = signals(drift_status="alert", drifted_features=["x"])
    recent = {"last_run_ts": NOW - 3600}
    d = decide(s, recent, CFG)
    assert d.trigger is None and d.reason.startswith("cooldown")
    assert decide(s, {"last_run_ts": NOW - 7 * 3600}, CFG).trigger == "drift"


def test_the_same_drift_episode_is_not_retrained_twice():
    s = signals(drift_status="alert", drifted_features=["contract"])
    seen = {"last_run_ts": NOW - 24 * 3600 + 1, "last_fingerprint": fingerprint(["contract"], NOW)}
    assert decide(s, seen, CFG).reason.startswith("already handled")
    other = {"last_run_ts": 0, "last_fingerprint": fingerprint(["tenure_months"], NOW)}
    assert decide(s, other, CFG).trigger == "drift"
    tomorrow = fingerprint(["contract"], NOW + 86400)
    assert tomorrow != fingerprint(["contract"], NOW)  # a new day is a new episode


def test_without_new_feedback_there_is_nothing_new_to_learn():
    d = decide(signals(drift_status="alert", new_feedback_rows=99), {}, CFG)
    assert d.trigger is None and d.reason.startswith("not enough new feedback")


def test_guards_apply_in_the_documented_order():
    s = signals(drift_status="alert", drifted_features=["a"], new_feedback_rows=0)
    state = {"last_run_ts": NOW - 60, "last_fingerprint": fingerprint(["a"], NOW)}
    assert decide(s, state, CFG).reason.startswith("cooldown")  # before 'already handled'
    assert decide(s, {"last_fingerprint": fingerprint(["a"], NOW)}, CFG).reason.startswith(
        "already handled"
    )


# ------------------------------------------------------------------ the run sequence
@pytest.fixture
def run_env(tmp_path):
    paths = make_workspace(tmp_path / "repo")
    store = LogStore(tmp_path / "logs", "salt")
    inputs = json.loads(
        canonical_rows(30, seed=2)[
            [
                c
                for c in canonical_rows(1).columns
                if c
                in (
                    "gender",
                    "senior_citizen",
                    "partner",
                    "dependents",
                    "tenure_months",
                    "phone_service",
                    "multiple_lines",
                    "internet_service",
                    "online_security",
                    "online_backup",
                    "device_protection",
                    "tech_support",
                    "streaming_tv",
                    "streaming_movies",
                    "contract",
                    "paperless_billing",
                    "payment_method",
                    "monthly_charges",
                    "total_charges",
                )
            ]
        ].to_json(orient="records")
    )
    for i, row in enumerate(inputs):
        store.log_prediction(
            {
                "request_id": f"r{i}",
                "cid": store.hash_id(f"C{i}"),
                "inputs": row,
                "probability": 0.5,
                "band": "HIGH",
                "run_id": "m",
                "version": "v1",
            }
        )
        store.log_feedback(churned=i % 2 == 0, customer_id=f"C{i}", source="simulator")
    return paths, store.directory, tmp_path / "state", make_params()


class FakeCommands:
    """Records commands; ``dvc repro`` writes a gate decision like the real pipeline would."""

    def __init__(self, paths, decision="PROMOTE", fail=None):
        self.calls, self.envs, self.paths, self.decision, self.fail = [], [], paths, decision, fail

    def __call__(self, cmd, env, timeout):
        self.calls.append(cmd)
        self.envs.append(env)
        key = " ".join(cmd[:2])
        if self.fail and self.fail in key:
            return 1, f"{key} exploded"
        if cmd[:2] == ["dvc", "repro"]:
            self.paths.report("gate_decision.json").parent.mkdir(parents=True, exist_ok=True)
            self.paths.report("gate_decision.json").write_text(
                json.dumps(
                    {
                        "decision": self.decision,
                        "reasons": [] if self.decision == "PROMOTE" else ["no recent gain"],
                    }
                )
            )
        return 0, ""


def go(run_env, fake, trigger="drift"):
    paths, logs, state, params = run_env
    return runner.retrain(paths, params, trigger, "because", "fp123", logs, state, command=fake)


def test_a_promoted_run_goes_through_every_step_in_order(run_env):
    fake = FakeCommands(run_env[0])
    record = go(run_env, fake)
    assert record["outcome"] == "promoted"
    assert [c[:2] for c in fake.calls] == [
        ["dvc", "add"],
        ["git", "add"],
        ["git", "commit"],
        ["dvc", "repro"],
        ["python", "-m"],
        ["dvc", "push"],
        ["git", "add"],
        ["git", "commit"],
    ]
    assert fake.calls[4][-1] == "churnsense.deploy.promote" and "data/feedback.dvc" in fake.calls[1]
    steps = {s["step"]: s for s in record["steps"]}
    assert "30 rows" in steps["batch"]["detail"] and steps["gate"]["detail"].startswith("PROMOTE")
    assert all(s["ok"] for s in record["steps"]) and record["duration_seconds"] >= 0


def test_the_pipeline_is_told_why_it_runs_and_git_has_an_identity(run_env):
    fake = FakeCommands(run_env[0])
    go(run_env, fake, "performance")
    env = fake.envs[3]  # dvc repro
    assert env["PIPELINE_TRIGGER"] == "performance" and env["DRIFT_ALERT_FINGERPRINT"] == "fp123"
    assert len(env["CONTROLLER_RUN_ID"]) == 16 and env["GIT_AUTHOR_NAME"] == "churnsense-controller"


def test_a_rejected_run_never_promotes_but_still_saves_results(run_env):
    fake = FakeCommands(run_env[0], decision="REJECT")
    record = go(run_env, fake)
    assert record["outcome"] == "rejected" and ["python", "-m"] not in [c[:2] for c in fake.calls]
    assert ["dvc", "push"] in [c[:2] for c in fake.calls]
    assert "no recent gain" in next(s for s in record["steps"] if s["step"] == "gate")["detail"]


@pytest.mark.parametrize("failing", ["dvc add", "git commit", "dvc repro", "python -m", "dvc push"])
def test_a_failing_step_stops_the_run_and_names_the_step(run_env, failing):
    fake = FakeCommands(run_env[0], fail=failing)
    record = go(run_env, fake)
    assert record["outcome"] == "failed" and record["steps"][-1]["ok"] is False
    assert "exploded" in record["steps"][-1]["detail"]
    if failing == "dvc repro":
        assert ["python", "-m"] not in [c[:2] for c in fake.calls]  # nothing was promoted


def test_nothing_to_commit_after_the_results_is_not_a_failure(run_env):
    class NothingToCommit(FakeCommands):
        def __call__(self, cmd, env, timeout):
            if cmd[:2] == ["git", "commit"] and "controller: promoted" in cmd[-1]:
                self.calls.append(cmd)
                return 1, "nothing to commit"
            return super().__call__(cmd, env, timeout)

    assert go(run_env, NothingToCommit(run_env[0]))["outcome"] == "promoted"


def test_without_new_feedback_the_batch_steps_are_skipped_but_the_pipeline_still_runs(run_env):
    paths, logs, state, params = run_env
    runner.batcher.build_batch(paths, logs, state)  # use up the feedback first
    fake = FakeCommands(paths)
    record = go(run_env, fake, "manual")
    assert [c[:2] for c in fake.calls][:2] == [["dvc", "repro"], ["python", "-m"]]
    assert next(s for s in record["steps"] if s["step"] == "batch")["detail"] == "no new feedback"


def test_a_missing_gate_decision_is_a_failure_not_a_promotion(run_env):
    class NoGate(FakeCommands):
        def __call__(self, cmd, env, timeout):
            self.calls.append(cmd)
            return 0, ""

    record = go(run_env, NoGate(run_env[0]))
    assert record["outcome"] == "failed" and record["steps"][-1]["step"] == "read_gate"


def test_run_command_reports_exit_code_tail_and_timeout(tmp_path):
    assert runner.run_command(["python", "-c", "print('hi')"], {}, 10, tmp_path) == (0, "hi")
    code, tail = runner.run_command(
        ["python", "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"], {}, 10
    )
    assert code == 3 and "bad" in tail
    assert runner.run_command(["python", "-c", "import time; time.sleep(5)"], {}, 0.2)[0] == 124


# ------------------------------------------------------------------ loop helpers
def test_only_one_retraining_can_hold_the_lock(tmp_path):
    with ctl.retrain_lock(tmp_path), pytest.raises(ctl.BusyError), ctl.retrain_lock(tmp_path):
        pass
    with ctl.retrain_lock(tmp_path):  # released again
        pass


def test_state_defaults_and_atomic_writes(tmp_path):
    state = ctl.read_state(tmp_path)
    assert state["last_run_ts"] == 0 and state["runs"] == {
        "failed": 0,
        "rejected": 0,
        "promoted": 0,
        "skipped": 0,
        "rolled_back": 0,
    }
    ctl.write_json(tmp_path / "x" / "s.json", {"a": 1})
    assert json.loads((tmp_path / "x" / "s.json").read_text()) == {"a": 1}
    assert not list((tmp_path / "x").glob("*.tmp"))


def test_old_log_files_are_deleted_and_recent_ones_kept(tmp_path):
    today = datetime(2026, 11, 20, tzinfo=UTC)
    for days in (0, 5, 14, 15, 40):
        day = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        (tmp_path / f"predictions-{day}.jsonl").write_text("{}\n")
    (tmp_path / "notes.txt").write_text("keep")
    removed = ctl.prune_logs(tmp_path, 14, now=today)
    assert sorted(removed) == ["predictions-2026-10-11.jsonl", "predictions-2026-11-05.jsonl"]
    assert (tmp_path / "notes.txt").exists() and len(list(tmp_path.glob("predictions-*"))) == 3
    assert ctl.prune_logs(tmp_path / "missing", 14) == []


def test_new_feedback_is_counted_after_the_batch_cursor(run_env):
    paths, logs, state, _ = run_env
    assert ctl.count_new_feedback(logs, state) == 30
    runner.batcher.build_batch(paths, logs, state)
    assert ctl.count_new_feedback(logs, state) == 0


def test_gather_treats_a_silent_monitor_as_no_signal(monkeypatch, run_env):
    _, logs, state, _ = run_env

    def fake_get(url, timeout):
        if "8002" in url:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"n": 150, "recall": 0.4, "enough_data": True})

    monkeypatch.setattr(ctl.httpx, "get", fake_get)
    got = ctl.gather("http://drift:8002", "http://backend:8000", logs, state)
    assert (
        got.drift_status is None
        and got.performance["recall"] == 0.4
        and got.new_feedback_rows == 30
    )
    monkeypatch.setattr(
        ctl.httpx,
        "get",
        lambda url, timeout: (
            httpx.Response(200, json={"status": "alert", "drifted_features": ["contract"]})
            if "8002" in url
            else httpx.Response(200, text="not json")
        ),
    )
    got = ctl.gather("http://drift:8002", "http://backend:8000", logs, state)
    assert (
        got.drift_status == "alert"
        and got.drifted_features == ["contract"]
        and got.performance is None
    )


class Gateway(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802 - name fixed by http.server
        Gateway.seen.append(
            (self.path, self.rfile.read(int(self.headers["Content-Length"])).decode())
        )
        self.send_response(200)
        self.end_headers()

    do_PUT = do_POST  # noqa: N815

    def log_message(self, *args):
        pass


def test_the_outcome_is_pushed_to_the_gateway_and_a_dead_gateway_is_harmless(monkeypatch):
    Gateway.seen = []
    server = HTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PUSHGATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
    state = {"runs": {"failed": 1, "rejected": 2, "promoted": 3, "skipped": 0}}
    ctl.push_metrics({"outcome": "promoted", "duration_seconds": 12.5}, state)
    server.shutdown()
    path, body = Gateway.seen[-1]
    assert "/job/churnsense_controller" in path
    assert (
        "churnsense_controller_last_outcome 2.0" in body
        and 'churnsense_controller_runs_total{outcome="promoted"} 3.0' in body
    )
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://127.0.0.1:9")
    ctl.push_metrics({"outcome": "failed", "duration_seconds": 1}, state)  # must not raise
    monkeypatch.delenv("PUSHGATEWAY_URL")
    ctl.push_metrics({"outcome": "failed"}, state)


def test_execute_records_the_run_updates_the_state_and_cools_down(run_env, monkeypatch):
    paths, logs, state, params = run_env
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    monkeypatch.setattr(
        ctl, "retrain", lambda *a, **k: {"run_id": "R1", "outcome": "promoted", "trigger": "drift"}
    )
    decision = decide(signals(drift_status="alert", drifted_features=["contract"]), {}, CFG)
    record = ctl.execute(paths, params, decision, logs, state)
    saved = ctl.read_state(state)
    assert record["outcome"] == "promoted" and saved["runs"]["promoted"] == 1
    assert (
        saved["last_fingerprint"] == decision.fingerprint and saved["last_run_ts"] > time.time() - 5
    )
    assert (state / "runs" / "R1.json").is_file()
    again = decide(
        signals(drift_status="alert", drifted_features=["contract"], now=time.time()), saved, CFG
    )
    assert again.trigger is None  # cooldown now applies


def test_the_loop_writes_its_status_and_runs_when_the_policy_says_so(run_env, monkeypatch):
    paths, logs, state, params = run_env
    ran = []
    monkeypatch.setattr(
        ctl,
        "gather",
        lambda *a: signals(drift_status="alert", drifted_features=["contract"], now=time.time()),
    )
    monkeypatch.setattr(ctl, "execute", lambda *a: ran.append(a[2].trigger) or {})

    def stop(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(ctl.time, "sleep", stop)
    with pytest.raises(KeyboardInterrupt):
        ctl.loop(paths, params, "d", "b", logs, state)
    status = json.loads((state / ctl.STATUS_FILE).read_text())
    assert ran == ["drift"] and status["decision"] == "drift" and status["new_feedback_rows"] == 500


def test_a_failing_round_is_logged_and_the_loop_keeps_going(run_env, monkeypatch):
    paths, logs, state, params = run_env
    rounds = []
    monkeypatch.setattr(ctl, "gather", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))

    def sleep(_):
        rounds.append(1)
        if len(rounds) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(ctl.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        ctl.loop(paths, params, "d", "b", logs, state)
    assert len(rounds) == 2  # it survived the first failure


@pytest.mark.parametrize(("outcome", "code"), [("promoted", 0), ("rejected", 0), ("failed", 1)])
def test_once_mode_exit_codes(monkeypatch, capsys, outcome, code, tmp_path):
    monkeypatch.setattr("sys.argv", ["controller", "--once"])
    monkeypatch.setattr(
        ctl,
        "execute",
        lambda *a: {
            "run_id": "r",
            "trigger": "manual",
            "outcome": outcome,
            "note": "n",
            "duration_seconds": 1,
        },
    )
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as stop:
        ctl.main()
    assert stop.value.code == code and outcome in capsys.readouterr().out


def test_once_mode_refuses_when_another_run_holds_the_lock(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["controller", "--once"])
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    with (tmp_path / ctl.LOCK_FILE).open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        monkeypatch.setattr(ctl, "retrain", lambda *a, **k: pytest.fail("must not run"))
        with pytest.raises(SystemExit) as stop:
            ctl.main()
    assert stop.value.code == 3 and "REFUSED" in capsys.readouterr().out


# ------------------------------------------------------------------ rollback
def test_rollback_switches_to_the_previous_version_and_reports_it(serving, monkeypatch):
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    export.switch_current(serving, 2)
    summary = rollback.rollback(serving, aliases=False)
    assert summary["from"] == 2 and summary["to"] == 1 and export.current_version(serving) == 1
    assert summary["live"].startswith("skipped")


def test_rollback_refuses_when_there_is_nothing_older_or_nothing_current(serving, tmp_path):
    with pytest.raises(rollback.RollbackError, match="oldest"):
        rollback.rollback(serving, live=False, aliases=False)  # current is v1, the oldest
    with pytest.raises(rollback.RollbackError, match="no current"):
        rollback.rollback(tmp_path / "empty", live=False, aliases=False)


def test_rollback_reloads_the_server_and_checks_health(serving, monkeypatch):
    export.switch_current(serving, 2)
    seen = {}
    monkeypatch.setenv("MODEL_SERVER_URL", "http://m")
    monkeypatch.setattr(
        smoke, "reload_model_server", lambda url, token, run: seen.update(run=run) or {}
    )
    monkeypatch.setattr(rollback.httpx, "get", lambda url, timeout: httpx.Response(200))
    summary = rollback.rollback(serving, aliases=False)
    assert (
        seen["run"] == export.read_meta(serving / "v1").get("run_id", "")
        and summary["live"] == "reloaded and healthy"
    )


@pytest.mark.parametrize("failure", ["reload", "degraded"])
def test_a_failed_live_check_puts_the_newer_model_back(serving, monkeypatch, failure):
    export.switch_current(serving, 2)
    monkeypatch.setenv("MODEL_SERVER_URL", "http://m")

    def reload(url, token, run):
        if failure == "reload":
            raise smoke.DeploymentError("server refused")
        return {}

    monkeypatch.setattr(smoke, "reload_model_server", reload)
    monkeypatch.setattr(rollback.httpx, "get", lambda url, timeout: httpx.Response(503))
    with pytest.raises(rollback.RollbackError, match="v2 is current again"):
        rollback.rollback(serving, aliases=False)
    assert export.current_version(serving) == 2


def test_rollback_moves_the_registry_aliases(serving, monkeypatch):
    export.switch_current(serving, 2)
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def set_registered_model_alias(self, name, alias, version):
            calls.append((alias, version))

    monkeypatch.setattr("mlflow.MlflowClient", FakeClient)
    rollback.rollback(serving)
    assert calls == [("champion", "1"), ("rolled_back", "2")]


def test_the_backend_shows_what_the_controller_is_doing(serving, tmp_path):
    from fastapi.testclient import TestClient

    from churnsense.api.backend import create_app
    from churnsense.serving.model_server import create_app as model_app

    state = tmp_path / "state"
    (state / "runs").mkdir(parents=True)
    (state / "controller_status.json").write_text(
        json.dumps({"decision": "no retraining", "reason": "healthy"})
    )
    (state / "controller_state.json").write_text(json.dumps({"runs": {"promoted": 2}}))
    (state / "runs" / "20260101T000000Z.json").write_text(
        json.dumps({"run_id": "old", "outcome": "failed"})
    )
    (state / "runs" / "20260102T000000Z.json").write_text(
        json.dumps(
            {"run_id": "new", "outcome": "promoted", "trigger": "drift", "secret_detail": "x"}
        )
    )
    with TestClient(model_app(serving, "t")) as model:
        client = TestClient(
            create_app(model, "t", LogStore(tmp_path / "logs"), tmp_path / "reports", state)
        )
        view = client.get("/api/v1/pipeline").json()["controller"]
        assert view["watching"]["decision"] == "no retraining" and view["runs"] == {"promoted": 2}
        assert view["last_run"]["run_id"] == "new" and view["last_run"]["outcome"] == "promoted"
        assert "secret_detail" not in view["last_run"]  # only the listed fields are shown
        empty = TestClient(
            create_app(model, "t", LogStore(tmp_path / "l2"), tmp_path / "r2", tmp_path / "none")
        )
        assert empty.get("/api/v1/pipeline").json()["controller"] == {
            "watching": {},
            "runs": None,
            "last_run": None,
        }


def test_a_registry_that_is_down_does_not_fail_the_rollback(serving, monkeypatch):
    export.switch_current(serving, 2)
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)

    class DownClient:
        def __init__(self, tracking_uri):
            pass

        def set_registered_model_alias(self, *args):
            raise ConnectionError("mlflow is unhealthy")

    monkeypatch.setattr("mlflow.MlflowClient", DownClient)
    summary = rollback.rollback(serving)
    assert export.current_version(serving) == 1  # the switch itself happened
    assert summary["aliases"].startswith("skipped: MLflow did not accept")


def test_make_rollback_does_not_start_or_wait_for_mlflow():
    from tests.conftest import REPO

    line = next(
        ln for ln in (REPO / "Makefile").read_text().splitlines() if "deploy.rollback" in ln
    )
    assert "--no-deps" in line
