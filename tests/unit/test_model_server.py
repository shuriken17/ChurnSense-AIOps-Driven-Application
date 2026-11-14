"""Unit tests for the model server (real small models, no Docker)."""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from churnsense.deploy import export
from churnsense.serving.model_server import SELF_CHECK_ROW, create_app

pytestmark = pytest.mark.unit
TOKEN = {"x-admin-token": "secret"}


@pytest.fixture
def client(serving):
    with TestClient(create_app(serving, "secret")) as c:
        yield c


def customer(**changes):
    return {**SELF_CHECK_ROW, **changes}


def test_health_and_model_info_describe_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"]["version"] == "v1"
    assert body["model"]["run_id"] == "run-v1" and body["model"]["family"] == "logreg"
    assert 0 < body["model"]["thresholds"]["t_medium"] <= body["model"]["thresholds"]["t_high"] < 1
    assert client.get("/model").json()["run_id"] == "run-v1"


def test_without_any_model_the_server_starts_and_says_so(tmp_path):
    with TestClient(create_app(tmp_path / "empty", "secret")) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/health").json() == {"status": "no_model"}
        assert c.post("/invocations", json={"instances": [customer()]}).status_code == 503
        assert c.get("/model").status_code == 503
        failed = c.post("/admin/reload", headers=TOKEN)
        assert failed.status_code == 500 and "no model has been promoted" in failed.json()["error"]


def test_predictions_have_probability_band_and_three_factors(client):
    body = client.post(
        "/invocations", json={"instances": [customer(), customer(tenure_months=60)]}
    ).json()
    assert body["model"] == {"version": "v1", "run_id": "run-v1"}
    assert len(body["predictions"]) == 2
    for row in body["predictions"]:
        assert 0 <= row["churn_probability"] <= 1 and row["risk_band"] in ("HIGH", "MEDIUM", "LOW")
        assert len(row["top_factors"]) == 3 and row["model_run_id"] == "run-v1"


def test_optional_fields_may_be_left_out(client):
    bare = customer()
    del bare["customer_id"], bare["total_charges"]
    assert client.post("/invocations", json={"instances": [bare]}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"instances": []},
        {"instances": "x"},
        {"instances": [{}]},
    ],
)
def test_bad_requests_are_422_not_500(client, body):
    assert client.post("/invocations", json=body).status_code == 422


def test_too_many_rows_are_refused(client):
    assert client.post("/invocations", json={"instances": [customer()] * 1001}).status_code == 422


def test_reload_needs_the_right_token(client, serving):
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"x-admin-token": "wrong"}).status_code == 401
    with TestClient(create_app(serving, "")) as disabled:
        assert disabled.post("/admin/reload", headers=TOKEN).status_code == 503


def test_reload_switches_to_the_new_model_without_a_restart(client, serving):
    export.switch_current(serving, 2)
    body = client.post("/admin/reload", headers=TOKEN).json()
    assert body["status"] == "reloaded" and body["model"]["run_id"] == "run-v2"
    assert body["previous"]["run_id"] == "run-v1"
    assert client.get("/health").json()["model"]["family"] == "lightgbm"
    answer = client.post("/invocations", json={"instances": [customer()]}).json()
    assert answer["model"]["run_id"] == "run-v2"


def test_a_broken_new_model_is_refused_and_the_old_one_keeps_serving(client, serving):
    broken = serving / "v3"
    shutil.copytree(serving / "v2", broken)
    (broken / "artifacts" / "bundle.joblib").write_bytes(b"corrupt")
    export.switch_current(serving, 3)
    answer = client.post("/admin/reload", headers=TOKEN)
    assert answer.status_code == 500 and answer.json()["status"] == "failed"
    assert answer.json()["serving"]["run_id"] == "run-v1"
    assert (
        client.post("/invocations", json={"instances": [customer()]}).json()["model"]["run_id"]
        == "run-v1"
    )


def test_metrics_count_requests_bands_and_the_model_in_use(client, serving):
    client.post("/invocations", json={"instances": [customer()]})
    client.post("/invocations", json={"instances": []})
    text = client.get("/metrics").text
    assert 'churnsense_model_requests_total{outcome="ok"} 1.0' in text
    assert 'churnsense_model_requests_total{outcome="invalid"} 1.0' in text
    assert (
        "churnsense_model_predictions_total{band=" in text
        and "churnsense_model_probability_bucket" in text
    )
    assert text.count("churnsense_model_info{") == 1 and 'run_id="run-v1"' in text
    export.switch_current(serving, 2)
    client.post("/admin/reload", headers=TOKEN)
    text = client.get("/metrics").text
    assert text.count("churnsense_model_info{") == 1 and 'run_id="run-v2"' in text
    assert 'churnsense_model_reloads_total{result="ok"} 2.0' in text


# ------------------------------------------------------------------ MLflow's own scoring API
ROW = {k: v for k, v in SELF_CHECK_ROW.items()}


def test_mlflow_scoring_api_is_mounted_and_answers_in_mlflow_format(client):
    assert client.get("/mlflow/ping").status_code == 200
    answer = client.post("/mlflow/invocations", json={"dataframe_records": [ROW]})
    assert answer.status_code == 200
    prediction = answer.json()["predictions"][0]
    assert set(prediction) == {"churn_probability", "risk_band", "top_factors", "model_run_id"}
    ours = client.post("/invocations", json={"instances": [ROW]}).json()["predictions"][0]
    assert prediction["churn_probability"] == pytest.approx(
        ours["churn_probability"]
    )  # the same model


def test_mlflow_scoring_api_enforces_the_model_signature(client):
    bad = client.post(
        "/mlflow/invocations", json={"dataframe_records": [{"contract": "Month-to-month"}]}
    )
    assert bad.status_code in (400, 422)


def test_mlflow_scoring_api_follows_a_reload_and_is_unavailable_without_a_model(
    client, serving, tmp_path
):
    export.switch_current(serving, 2)
    client.post("/admin/reload", headers=TOKEN)
    after = client.post("/mlflow/invocations", json={"dataframe_records": [ROW]}).json()[
        "predictions"
    ][0]
    assert after["model_run_id"] == "run-v2"
    with TestClient(create_app(tmp_path / "empty", "secret")) as empty:
        assert (
            empty.post("/mlflow/invocations", json={"dataframe_records": [ROW]}).status_code == 503
        )
