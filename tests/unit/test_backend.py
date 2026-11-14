"""Unit tests for customer validation and the backend API (model server in the same process)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from churnsense.api.backend import create_app as backend_app
from churnsense.api.validation import validate_customer
from churnsense.schema import INPUT_FIELDS
from churnsense.serving.model_server import SELF_CHECK_ROW
from churnsense.serving.model_server import create_app as model_app
from churnsense.store import LogStore

pytestmark = pytest.mark.unit
LIMIT = {f.name: f.max_value for f in INPUT_FIELDS}
GOOD = {k: v for k, v in SELF_CHECK_ROW.items() if k != "customer_id"}


def problems(**changes):
    _, found = validate_customer({**GOOD, **changes})
    return {p["field"]: p["message"] for p in found}


def test_a_good_customer_passes_and_whole_numbers_become_ints():
    clean, found = validate_customer({**GOOD, "tenure_months": 12.0, "senior_citizen": 1.0})
    assert found == [] and clean["tenure_months"] == 12 and isinstance(clean["tenure_months"], int)


@pytest.mark.parametrize(
    ("changes", "field", "text"),
    [
        ({"contract": "Weekly"}, "contract", "must be one of: Month-to-month"),
        ({"gender": "other"}, "gender", "must be one of"),
        ({"tenure_months": -1}, "tenure_months", "at least 0"),
        ({"tenure_months": LIMIT["tenure_months"] + 1}, "tenure_months", "at most"),
        ({"tenure_months": 5.5}, "tenure_months", "whole number"),
        ({"tenure_months": "12"}, "tenure_months", "must be a number"),
        ({"tenure_months": True}, "tenure_months", "must be a number"),
        ({"senior_citizen": 2}, "senior_citizen", "0 or 1"),
        ({"monthly_charges": 0}, "monthly_charges", "above 0"),
        ({"monthly_charges": LIMIT["monthly_charges"] + 1}, "monthly_charges", "at most"),
        ({"total_charges": -1}, "total_charges", "at least"),
        ({"customer_id": "x" * 65}, "customer_id", "at most 64"),
        ({"customer_id": 5}, "customer_id", "text"),
        ({"mystery": 1}, "mystery", "not a known field"),
        ({"gender": None}, "gender", "is required"),
    ],
)
def test_each_rule_names_the_field_and_says_what_is_wrong(changes, field, text):
    assert text in problems(**changes)[field]


def test_missing_required_fields_are_listed_but_optional_ones_are_not():
    _, found = validate_customer({"contract": "Month-to-month"})
    names = {p["field"] for p in found}
    assert "gender" in names and "customer_id" not in names and "total_charges" not in names
    assert validate_customer({**GOOD, "total_charges": None})[1] == []


def test_relationships_between_fields_are_checked_only_on_otherwise_valid_data():
    assert "(combination)" in problems(phone_service="No", multiple_lines="Yes")
    assert "(combination)" in problems(internet_service="No", online_security="Yes")
    assert "(combination)" not in problems(
        contract="Weekly", phone_service="No", multiple_lines="Yes"
    )


def test_a_body_that_is_not_an_object_is_one_clear_problem():
    assert validate_customer([1, 2])[1][0]["field"] == "(body)"


# ------------------------------------------------------------------ the API
@pytest.fixture
def api(serving, tmp_path):
    with TestClient(model_app(serving, "secret")) as model:
        store = LogStore(tmp_path / "logs")
        yield TestClient(backend_app(model, "secret", store, tmp_path / "reports")), model


def test_health_fields_and_model_endpoints(api):
    client, _ = api
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok" and health["model"]["version"] == "v1"
    fields = client.get("/api/v1/fields").json()
    assert len(fields) == 20 and {"name", "label", "kind", "required", "allowed"} <= set(fields[0])
    assert client.get("/api/v1/model").json()["run_id"] == "run-v1"


def test_predict_returns_the_documented_answer(api):
    client, _ = api
    body = client.post(
        "/api/v1/predict", json={**GOOD, "customer_id": "A-1"}, headers={"x-request-id": "req-7"}
    ).json()
    assert set(body) == {
        "customer_id",
        "churn_probability",
        "risk_band",
        "top_factors",
        "model",
        "request_id",
    }
    assert body["customer_id"] == "A-1" and body["request_id"] == "req-7"
    assert body["model"]["version"] == "v1" and len(body["top_factors"]) == 3


def test_invalid_customers_get_422_in_the_one_error_shape(api):
    client, _ = api
    answer = client.post("/api/v1/predict", json={**GOOD, "contract": "Weekly"})
    error = answer.json()["error"]
    assert answer.status_code == 422 and error["code"] == "invalid_customer"
    assert error["details"] == [{"field": "contract", "message": error["details"][0]["message"]}]
    assert client.post("/api/v1/predict", content=b"not json").status_code == 422
    assert client.post("/api/v1/predict", json=[1]).status_code == 422


def test_unknown_routes_use_the_same_error_shape(api):
    client, _ = api
    answer = client.get("/api/v1/nothing")
    assert answer.status_code == 404 and answer.json()["error"]["code"] == "http_404"


def test_every_response_has_a_request_id(api):
    client, _ = api
    assert len(client.get("/api/v1/health").headers["x-request-id"]) >= 8
    assert (
        client.get("/api/v1/health", headers={"x-request-id": "mine"}).headers["x-request-id"]
        == "mine"
    )


def broken_client() -> httpx.Client:
    def refuse(request):
        raise httpx.ConnectError("down")

    return httpx.Client(base_url="http://model-server:8001", transport=httpx.MockTransport(refuse))


def test_a_down_model_server_gives_503_not_a_crash():
    client = TestClient(backend_app(broken_client(), "secret"))
    health = client.get("/api/v1/health")
    assert health.status_code == 503 and health.json()["model_server"] == "down"
    answer = client.post("/api/v1/predict", json=GOOD)
    assert answer.status_code == 503 and "not reachable" in answer.json()["error"]["message"]
    assert client.get("/api/v1/fields").status_code == 200  # the backend itself is fine


def test_a_model_server_without_a_model_gives_503(tmp_path):
    with TestClient(model_app(tmp_path / "none", "secret")) as model:
        client = TestClient(backend_app(model, "secret"))
        assert client.post("/api/v1/predict", json=GOOD).status_code == 503
        assert client.get("/api/v1/model").status_code == 503


def test_admin_reload_needs_the_token_and_passes_through(api):
    client, _ = api
    assert client.post("/api/v1/admin/reload-model").status_code == 401
    assert (
        client.post("/api/v1/admin/reload-model", headers={"x-admin-token": "no"}).status_code
        == 401
    )
    ok = client.post("/api/v1/admin/reload-model", headers={"x-admin-token": "secret"})
    assert ok.status_code == 200 and ok.json()["status"] == "reloaded"
    assert (
        TestClient(backend_app(broken_client(), "")).post("/api/v1/admin/reload-model").status_code
        == 503
    )


def test_metrics_count_requests_and_rejections(api):
    client, _ = api
    client.post("/api/v1/predict", json=GOOD)
    client.post("/api/v1/predict", json={**GOOD, "contract": "x"})
    text = client.get("/api/v1/metrics").text
    assert (
        'churnsense_api_requests_total{method="POST",path="/api/v1/predict",status="200"} 1.0'
        in text
    )
    assert "churnsense_api_validation_errors_total 1.0" in text
