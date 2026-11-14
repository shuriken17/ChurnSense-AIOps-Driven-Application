"""Unit tests for the simulator, the feedback batcher and the new backend endpoints."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnsense.api.backend import create_app as backend_app
from churnsense.feedback import batcher
from churnsense.pipeline.ingest import verify_feedback
from churnsense.schema import INPUT_COLUMNS, cross_field_violations
from churnsense.serving.model_server import SELF_CHECK_ROW
from churnsense.serving.model_server import create_app as model_app
from churnsense.simulate import traffic
from churnsense.store import LogStore
from tests.conftest import make_workspace
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit
GOOD = {k: v for k, v in SELF_CHECK_ROW.items() if k != "customer_id"}
BASE = canonical_rows(400, seed=2)[traffic.FEATURES]


# ------------------------------------------------------------------ scenarios
def test_normal_traffic_is_the_real_data_unchanged():
    out = traffic.apply_scenario(BASE, "normal", np.random.default_rng(1))
    pd.testing.assert_frame_equal(out, BASE.reset_index(drop=True))


@pytest.mark.parametrize("scenario", ["drift_mild", "drift_strong"])
def test_drifted_traffic_shifts_the_inputs_but_never_breaks_the_rules(scenario):
    out = traffic.apply_scenario(BASE, scenario, np.random.default_rng(1))
    assert (out["contract"] == "Month-to-month").mean() > (
        BASE["contract"] == "Month-to-month"
    ).mean()
    assert out["monthly_charges"].mean() > BASE["monthly_charges"].mean()
    assert (cross_field_violations(out.assign(customer_id="x")) == "").all()
    assert out["monthly_charges"].between(0.01, 118.75).all() and (out["total_charges"] >= 0).all()
    np.testing.assert_allclose(
        out["total_charges"], (out["tenure_months"] * out["monthly_charges"]).round(2)
    )


def test_the_strong_scenario_also_shortens_tenure_and_adds_fiber():
    out = traffic.apply_scenario(BASE, "drift_strong", np.random.default_rng(1))
    assert out["tenure_months"].mean() < BASE["tenure_months"].mean() * 0.8
    assert (out["internet_service"] == "Fiber optic").mean() > 0.7


def test_scenarios_are_repeatable_and_unknown_ones_are_refused():
    a = traffic.apply_scenario(BASE, "drift_strong", np.random.default_rng(5))
    b = traffic.apply_scenario(BASE, "drift_strong", np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="unknown scenario"):
        traffic.apply_scenario(BASE, "chaos", np.random.default_rng(1))


def test_the_synthetic_ground_truth_is_a_probability_and_drifts_in_the_strong_scenario():
    normal = traffic.true_churn_probability(BASE, "normal")
    assert ((normal > 0) & (normal < 1)).all() and 0.15 < normal.mean() < 0.45
    strong_inputs = traffic.apply_scenario(BASE, "drift_strong", np.random.default_rng(1))
    same_inputs = traffic.true_churn_probability(strong_inputs, "normal")
    concept = traffic.true_churn_probability(strong_inputs, "drift_strong")
    assert concept.mean() > same_inputs.mean()  # the rule itself moved, not only the inputs


# ------------------------------------------------------------------ simulator against the API
@pytest.fixture
def system(serving, tmp_path):
    store = LogStore(tmp_path / "logs", "salt")
    with TestClient(model_app(serving, "secret")) as model:
        yield model, store, tmp_path


def sim(system, scenario="normal", n=40, feedback=True):
    model, store, tmp = system
    client = TestClient(backend_app(model, "secret", store, tmp / "reports"))
    return traffic.run("http://x", scenario, n, 7, feedback, BASE, client=client)


def test_the_simulator_sends_customers_and_synthetic_outcomes(system):
    _, store, _ = system
    summary = sim(system, "drift_strong", 40)
    assert summary["scored"] == 40 and summary["rejected"] == 0 and summary["feedback_sent"] == 40
    assert sum(summary["bands"].values()) == 40 and summary["synthetic_labels"] is True
    assert store.performance(min_samples=30)["n"] == 40
    assert {r["source"] for r in store_records(store, "feedback")} == {"simulator"}


def store_records(store, kind):
    from churnsense.store import read_recent

    return read_recent(store.directory, kind, 10_000)


def test_without_feedback_only_predictions_are_logged(system):
    _, store, _ = system
    assert sim(system, "normal", 10, feedback=False)["feedback_sent"] == 0
    assert len(store_records(store, "predictions")) == 10 and store_records(store, "feedback") == []


# ------------------------------------------------------------------ batcher
def make_batch(system, **kw):
    _, store, tmp = system
    paths = make_workspace(tmp / "repo")
    return paths, tmp / "state", batcher.build_batch(paths, store.directory, tmp / "state", **kw)


def test_a_batch_has_the_ingest_columns_a_checksum_and_the_synthetic_flag(system):
    sim(system, "normal", 40)
    paths, state, entry = make_batch(system, scenario="normal")
    frame = pd.read_parquet(paths.feedback_dir / entry["file"])
    assert list(frame.columns) == [*INPUT_COLUMNS, "churn"] and len(frame) == 40
    assert frame["customer_id"].str.startswith("live-").all() and frame["customer_id"].is_unique
    assert frame["churn"].isin([0, 1]).all() and str(frame["tenure_months"].dtype) == "int64"
    assert (
        entry["sha256"]
        == hashlib.sha256((paths.feedback_dir / entry["file"]).read_bytes()).hexdigest()
    )
    assert (
        entry["rows"] == 40 and entry["scenario"] == "normal" and entry["synthetic_labels"] is True
    )
    manifest = json.loads((paths.feedback_dir / "manifest.json").read_text())
    assert manifest["batches"] == [entry]
    accepted, quarantined = verify_feedback(paths.feedback_dir)  # what the ingest stage will do
    assert [a.name for a in accepted] == [entry["file"]] and quarantined == []


def test_each_feedback_goes_into_exactly_one_batch(system):
    _, store, tmp = system
    sim(system, "normal", 20)
    paths, state, first = make_batch(system)
    assert batcher.build_batch(paths, store.directory, state) is None  # nothing new
    model = system[0]
    sim((model, store, tmp), "drift_mild", 15)
    second = batcher.build_batch(paths, store.directory, state)
    assert second["rows"] == 15 and second["file"] != first["file"]  # even within one second
    assert len(json.loads((paths.feedback_dir / "manifest.json").read_text())["batches"]) == 2


def test_a_customer_reported_twice_appears_once_with_the_latest_outcome(system):
    model, store, tmp = system
    client = TestClient(backend_app(model, "secret", store, tmp / "reports"))
    client.post("/api/v1/predict", json={**GOOD, "customer_id": "C-9"})
    client.post("/api/v1/feedback", json={"customer_id": "C-9", "churned": False})
    client.post("/api/v1/feedback", json={"customer_id": "C-9", "churned": True})
    paths, _, entry = make_batch(system)
    frame = pd.read_parquet(paths.feedback_dir / entry["file"])
    assert len(frame) == 1 and frame["churn"].iloc[0] == 1 and entry["synthetic_labels"] is False


def test_feedback_without_a_customer_id_still_makes_a_row(system):
    model, store, tmp = system
    client = TestClient(backend_app(model, "secret", store, tmp / "reports"))
    request_id = client.post("/api/v1/predict", json=GOOD).json()["request_id"]
    client.post("/api/v1/feedback", json={"request_id": request_id, "churned": False})
    paths, _, entry = make_batch(system)
    assert (
        pd.read_parquet(paths.feedback_dir / entry["file"])["customer_id"].iloc[0]
        == f"req-{request_id}"
    )


# ------------------------------------------------------------------ new backend endpoints
@pytest.fixture
def api(system):
    model, store, tmp = system
    reports = tmp / "reports"
    reports.mkdir()
    return TestClient(backend_app(model, "secret", store, reports)), store, reports


def test_batch_scoring_scores_good_rows_and_explains_bad_ones(api):
    client, store, _ = api
    body = client.post(
        "/api/v1/predict/batch",
        json={
            "customers": [
                {**GOOD, "customer_id": "A"},
                {"contract": "x"},
                {**GOOD, "tenure_months": 70},
            ]
        },
    ).json()
    assert (
        body["summary"]["total"] == 3
        and body["summary"]["scored"] == 2
        and body["summary"]["rejected"] == 1
    )
    assert [r["index"] for r in body["results"]] == [0, 1, 2]
    assert (
        body["results"][1]["error"]["code"] == "invalid_customer"
        and "risk_band" in body["results"][0]
    )
    assert sum(body["summary"]["bands"].values()) == 2 and body["model"]["version"] == "v1"
    assert len(store_records(store, "predictions")) == 2  # only scored rows are logged


@pytest.mark.parametrize(
    "body", [{}, {"customers": []}, {"customers": "x"}, {"customers": [GOOD] * 1001}]
)
def test_a_malformed_batch_is_one_422(api, body):
    client, _, _ = api
    answer = client.post("/api/v1/predict/batch", json=body)
    assert answer.status_code == 422 and answer.json()["error"]["code"] == "invalid_request"


def test_a_batch_of_only_bad_rows_does_not_call_the_model(api):
    client, _, _ = api
    body = client.post("/api/v1/predict/batch", json={"customers": [{}, {"x": 1}]}).json()
    assert body["summary"]["scored"] == 0 and body["model"] is None


def test_feedback_endpoint_all_cases(api):
    client, _, _ = api
    rid = client.post("/api/v1/predict", json={**GOOD, "customer_id": "A"}).json()["request_id"]
    ok = client.post("/api/v1/feedback", json={"request_id": rid, "churned": True}).json()
    assert ok["status"] == "recorded" and ok["counted_as"] in ("true_positive", "false_negative")
    assert ok["matched"]["model"]["version"] == "v1"
    assert (
        client.post("/api/v1/feedback", json={"customer_id": "A", "churned": 0}).status_code == 200
    )
    unknown = client.post("/api/v1/feedback", json={"customer_id": "nobody", "churned": True})
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "unknown_prediction"
    for bad in (
        {"churned": True},
        {"customer_id": "A"},
        {"customer_id": "A", "churned": "maybe"},
        {"customer_id": "A", "request_id": rid, "churned": True},
        {"customer_id": "A", "churned": True, "source": "robot"},
        {"customer_id": 5, "churned": True},
    ):
        answer = client.post("/api/v1/feedback", json=bad)
        assert answer.status_code == 422 and answer.json()["error"]["code"] == "invalid_feedback", (
            bad
        )


def test_performance_stays_empty_until_there_are_enough_outcomes_then_reports_numbers(api):
    client, _, _ = api
    assert client.get("/api/v1/performance").json()["accuracy"] is None
    for i in range(35):
        client.post("/api/v1/predict", json={**GOOD, "customer_id": f"P-{i}"})
        client.post("/api/v1/feedback", json={"customer_id": f"P-{i}", "churned": i % 3 == 0})
    perf = client.get("/api/v1/performance").json()
    assert perf["n"] == 35 and perf["enough_data"] is True and 0 <= perf["accuracy"] <= 1
    assert perf["tp"] + perf["fp"] + perf["fn"] + perf["tn"] == 35
    text = client.get("/api/v1/metrics").text
    assert "churnsense_live_accuracy " in text and "churnsense_live_feedback_rows 35.0" in text
    assert 'churnsense_feedback_total{outcome="churned"} 12.0' in text


def test_stats_summarise_the_last_day(api):
    client, _, _ = api
    empty = client.get("/api/v1/stats").json()
    assert empty["predictions"] == 0 and empty["mean_probability"] is None
    for _ in range(4):
        client.post("/api/v1/predict", json=GOOD)
    stats = client.get("/api/v1/stats").json()
    assert (
        stats["predictions"] == 4
        and sum(stats["bands"].values()) == 4
        and stats["last_prediction_at"]
    )


def test_pipeline_status_reads_the_report_files_and_copes_with_none(api):
    client, _, reports = api
    assert client.get("/api/v1/pipeline").json()["gate"]["decision"] is None
    (reports / "gate_decision.json").write_text(
        json.dumps({"decision": "REJECT", "reasons": ["why"]})
    )
    (reports / "eval_metrics.json").write_text(
        json.dumps(
            {
                "challenger": {
                    "golden": {
                        "n": 9,
                        "roc_auc": 0.8,
                        "pr_auc": 0.6,
                        "recall_at_min_precision": 0.9,
                        "x": 1,
                    }
                }
            }
        )
    )
    body = client.get("/api/v1/pipeline").json()
    assert body["gate"] == {"decision": "REJECT", "reasons": ["why"]}
    assert body["golden"] == {"n": 9, "roc_auc": 0.8, "pr_auc": 0.6, "recall_at_min_precision": 0.9}


def test_a_failing_log_never_fails_the_prediction(system):
    model, _, tmp = system

    class Broken(LogStore):
        def log_prediction(self, record):
            raise OSError("disk full")

    client = TestClient(backend_app(model, "secret", Broken(tmp / "x"), tmp / "reports"))
    assert client.post("/api/v1/predict", json=GOOD).status_code == 200
    assert "churnsense_api_log_errors_total 1.0" in client.get("/api/v1/metrics").text


# ------------------------------------------------------------------ the lost-manifest incident
def test_a_manifest_overwritten_by_an_archive_is_caught_and_can_be_repaired(system):
    """Reproduces the real incident: an archive replaced manifest.json with an empty one."""
    sim(system, "normal", 40)
    paths, _, entry = make_batch(system, scenario="normal")
    manifest_file = paths.feedback_dir / "manifest.json"
    manifest_file.write_text(json.dumps({"batches": []}))  # what the archive did

    accepted, quarantined = verify_feedback(paths.feedback_dir)
    assert accepted == [] and quarantined[0]["reason"] == "not listed in manifest.json"
    assert (
        quarantined[0]["rows"] == 40
    )  # the guard failed closed: nothing unverified was trained on

    added = batcher.repair_manifest(paths)
    assert [a["file"] for a in added] == [entry["file"]]
    assert added[0]["recovered"] is True and added[0]["scenario"] == "recovered"
    assert added[0]["synthetic_labels"] is True and added[0]["rows"] == 40
    accepted, quarantined = verify_feedback(paths.feedback_dir)
    assert [a.name for a in accepted] == [entry["file"]] and quarantined == []
    assert batcher.repair_manifest(paths) == []  # repeating it changes nothing


def test_repair_keeps_existing_entries_and_orders_batches_by_time(system):
    import os

    sim(system, "normal", 20)
    paths, _, first = make_batch(system)
    manifest_file = paths.feedback_dir / "manifest.json"
    older = paths.feedback_dir / "batch_old.parquet"
    pd.read_parquet(paths.feedback_dir / first["file"]).to_parquet(older)
    os.utime(older, (1_700_000_000, 1_700_000_000))  # an old file the manifest does not know
    added = batcher.repair_manifest(paths)
    manifest = json.loads(manifest_file.read_text())
    assert [a["file"] for a in added] == ["batch_old.parquet"]
    assert [b["file"] for b in manifest["batches"]] == ["batch_old.parquet", first["file"]]
    assert manifest["batches"][1] == first  # the original entry is untouched


def test_the_repair_command_line_reports_nothing_to_do(monkeypatch, capsys, system):
    paths = make_workspace(system[2] / "clean")
    monkeypatch.setattr("sys.argv", ["batcher", "--repair"])
    monkeypatch.setattr(batcher, "default_paths", lambda: paths)
    batcher.main()
    assert "nothing to repair" in capsys.readouterr().out


def test_release_archives_never_contain_data_files():
    from tests.conftest import REPO

    script = (
        (REPO.parent / "package_batch.sh") if (REPO.parent / "package_batch.sh").exists() else None
    )
    if script is None:
        pytest.skip("the packaging script lives outside the repository")
    assert "data/feedback/*" in script.read_text()


def test_repeating_a_simulation_creates_new_customers_not_the_same_ones(system):
    _, store, _ = system
    sim(system, "normal", 20)
    sim(system, "normal", 20)  # same scenario, same seed (7), run again
    ids = [r["cid"] for r in store_records(store, "predictions")]
    assert len(ids) == 40 and len(set(ids)) == 40  # no customer appears twice


def test_a_fixed_run_tag_makes_the_ids_predictable(system):
    model, store, tmp = system
    client = TestClient(backend_app(model, "secret", store, tmp / "reports"))
    traffic.run("http://x", "normal", 3, 7, False, BASE, client=client, run_tag="abc")
    assert store.find(customer_id="sim-normal-7-abc-0") is not None


def test_the_drift_endpoint_passes_the_monitor_status_through_and_copes_when_it_is_down(system):
    import httpx

    model, store, tmp = system
    status = {"status": "alert", "max_psi": 1.2, "psi_by_feature": {"contract": 1.2}, "history": []}

    def monitor(request):
        return (
            httpx.Response(200, json=status)
            if request.url.path == "/status"
            else httpx.Response(404)
        )

    drift_client = httpx.Client(base_url="http://drift", transport=httpx.MockTransport(monitor))
    api = TestClient(backend_app(model, "secret", store, tmp / "r", tmp / "s", drift_client))
    assert api.get("/api/v1/drift").json() == status

    def down(request):
        raise httpx.ConnectError("down")

    broken = httpx.Client(base_url="http://drift", transport=httpx.MockTransport(down))
    api = TestClient(backend_app(model, "secret", store, tmp / "r", tmp / "s", broken))
    answer = api.get("/api/v1/drift")
    assert answer.status_code == 503 and "drift monitor" in answer.json()["error"]["message"]
    wrong = httpx.Client(
        base_url="http://drift", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    api = TestClient(backend_app(model, "secret", store, tmp / "r", tmp / "s", wrong))
    assert api.get("/api/v1/drift").status_code == 503
