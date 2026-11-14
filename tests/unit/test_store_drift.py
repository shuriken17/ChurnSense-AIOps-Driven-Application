"""Unit tests for the inference log and feedback store, PSI and the drift monitor."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnsense import store as store_module
from churnsense.drift import monitor as monitor_module
from churnsense.drift.psi import categorical_shares, compute_drift, numeric_shares, psi
from churnsense.features.derive import derive_features
from churnsense.features.profile import build_reference_profile
from churnsense.schema import INPUT_COLUMNS
from churnsense.store import LogStore, UnknownPredictionError, performance_of, read_recent
from tests.conftest import make_params
from tests.factories import canonical_rows

pytestmark = pytest.mark.unit
FEATURES = [c for c in INPUT_COLUMNS if c != "customer_id"]


def record(request_id="r1", customer="A-1", band="HIGH", probability=0.7, store=None):
    cid = store.hash_id(customer) if store else None
    return {
        "request_id": request_id,
        "cid": cid,
        "inputs": {"tenure_months": 3},
        "probability": probability,
        "band": band,
        "run_id": "run-1",
        "version": "v1",
    }


# ------------------------------------------------------------------ store
def test_ids_are_hashed_with_the_salt_and_missing_ids_stay_missing(tmp_path):
    a, b = LogStore(tmp_path, "salt-a"), LogStore(tmp_path, "salt-b")
    assert a.hash_id("X") == a.hash_id("X") != b.hash_id("X") and len(a.hash_id("X")) == 16
    assert a.hash_id(None) is None and a.hash_id("") is None


def test_predictions_are_appended_one_json_line_each_and_never_hold_the_raw_id(tmp_path):
    store = LogStore(tmp_path)
    store.log_prediction(record("r1", "A-1", store=store))
    store.log_prediction(record("r2", "B-2", store=store))
    (file,) = tmp_path.glob("predictions-*.jsonl")
    lines = [json.loads(line) for line in file.read_text().splitlines()]
    assert [r["request_id"] for r in lines] == ["r1", "r2"] and "ts" in lines[0]
    assert "A-1" not in file.read_text() and lines[0]["cid"] == store.hash_id("A-1")


def test_feedback_finds_the_prediction_by_request_id_or_by_customer_id(tmp_path):
    store = LogStore(tmp_path)
    store.log_prediction(record("r1", "A-1", "HIGH", 0.7, store))
    by_request = store.log_feedback(churned=True, request_id="r1")
    by_customer = store.log_feedback(churned=False, customer_id="A-1", source="simulator")
    assert by_request["band"] == "HIGH" and by_request["churned"] == 1 and by_request["inputs"]
    assert by_customer["churned"] == 0 and by_customer["source"] == "simulator"
    assert len(read_recent(tmp_path, "feedback", 10)) == 2


def test_feedback_for_an_unknown_prediction_is_an_error(tmp_path):
    store = LogStore(tmp_path)
    with pytest.raises(UnknownPredictionError):
        store.log_feedback(churned=True, request_id="nope")
    with pytest.raises(UnknownPredictionError):
        store.log_feedback(churned=True, customer_id="nobody")
    assert not list(tmp_path.glob("feedback-*"))  # nothing was written


def test_a_restarted_store_still_finds_earlier_predictions(tmp_path):
    first = LogStore(tmp_path)
    first.log_prediction(record("r1", "A-1", store=first))
    again = LogStore(tmp_path)  # a new process: the index is rebuilt from the files
    assert again.log_feedback(churned=True, customer_id="A-1")["request_id"] == "r1"


def test_the_index_is_bounded_and_drops_the_oldest(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "INDEX_LIMIT", 3)
    store = LogStore(tmp_path)
    for i in range(5):
        store.log_prediction(record(f"r{i}", f"C-{i}", store=store))
    assert store.find(request_id="r0") is None and store.find(request_id="r4") is not None
    assert store.find(customer_id="C-0") is None and store.find(customer_id="C-4") is not None


def test_read_recent_returns_the_newest_n_oldest_first_across_days_and_skips_bad_lines(tmp_path):
    for day, n in ((2, 3), (3, 3)):
        lines = [json.dumps({"request_id": f"d{day}-{i}"}) for i in range(n)]
        (tmp_path / f"predictions-2026-01-0{day}.jsonl").write_text(
            "\n".join(lines) + "\n{broken\n"
        )
    got = [r["request_id"] for r in read_recent(tmp_path, "predictions", 4)]
    assert got == ["d2-2", "d3-0", "d3-1", "d3-2"]
    assert len(read_recent(tmp_path, "predictions", 100)) == 6
    assert read_recent(tmp_path / "missing", "predictions", 5) == []


def test_performance_counts_high_band_as_predicted_churn():
    rows = (
        [{"band": "HIGH", "churned": 1}] * 8
        + [{"band": "HIGH", "churned": 0}] * 2
        + [{"band": "LOW", "churned": 1}] * 4
        + [{"band": "MEDIUM", "churned": 0}] * 16
    )
    out = performance_of(rows, min_samples=30)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (8, 2, 4, 16) and out["n"] == 30
    assert out["accuracy"] == pytest.approx(24 / 30)
    assert out["precision"] == pytest.approx(0.8) and out["recall"] == pytest.approx(8 / 12)


def test_performance_with_too_little_or_degenerate_data_is_none_not_a_crash():
    few = performance_of([{"band": "HIGH", "churned": 1}] * 5, min_samples=30)
    assert few["enough_data"] is False and few["accuracy"] is None and few["n"] == 5
    none_high = performance_of([{"band": "LOW", "churned": 0}] * 40, min_samples=30)
    assert none_high["precision"] is None and none_high["recall"] is None
    assert none_high["accuracy"] == 1.0
    assert performance_of([], min_samples=1)["accuracy"] is None


# ------------------------------------------------------------------ PSI
def test_psi_is_zero_for_equal_shares_and_matches_the_formula():
    assert psi([0.25] * 4, [0.25] * 4) == pytest.approx(0.0, abs=1e-12)
    expected, actual = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert psi(expected, actual) == pytest.approx(
        float(np.sum((actual - expected) * np.log(actual / expected)))
    )
    assert psi(expected, actual) == pytest.approx(psi(actual, expected))  # symmetric in this form


def test_psi_survives_empty_bins():
    assert (
        np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))
        and psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]) > 1
    )


def test_numeric_bins_are_left_closed_like_the_profile_and_cover_everything():
    edges = [10.0, 20.0]
    shares = numeric_shares(pd.Series([5, 10, 10.5, 20, 25, np.nan]), edges)
    assert shares == pytest.approx([2 / 5, 2 / 5, 1 / 5])  # x <= 10 | 10 < x <= 20 | x > 20


def test_categorical_shares_put_unseen_and_missing_values_in_other():
    shares = categorical_shares(pd.Series(["a", "a", "b", "z", None]), ["a", "b"])
    assert shares == pytest.approx([0.4, 0.2, 0.4])


@pytest.fixture(scope="module")
def profile():
    return build_reference_profile(derive_features(canonical_rows(500, seed=3)))


def shifted(n=300):
    frame = canonical_rows(n, seed=4)
    frame["contract"] = "Month-to-month"
    frame["tenure_months"] = (frame["tenure_months"] // 6).astype(int)
    return frame


def test_compute_drift_is_quiet_for_the_same_distribution_and_loud_for_a_shifted_one(profile):
    same = compute_drift(
        profile, derive_features(canonical_rows(300, seed=4)), min_rows=200, psi_alert=0.25
    )
    assert same.max_psi < 0.10 and same.drifted == [] and same.enough_data
    moved = compute_drift(profile, derive_features(shifted()), min_rows=200, psi_alert=0.25)
    assert moved.max_psi > 1 and {"contract", "tenure_months"} <= set(moved.drifted)
    values = [moved.psi_by_feature[f] for f in moved.drifted]
    assert values == sorted(values, reverse=True) and moved.worst_feature == moved.drifted[0]


def test_compute_drift_gives_no_verdict_without_enough_rows(profile):
    report = compute_drift(
        profile, derive_features(canonical_rows(50, seed=4)), min_rows=200, psi_alert=0.25
    )
    assert report.enough_data is False and report.psi_by_feature == {}


# ------------------------------------------------------------------ monitor
def log_rows(store, frame, prefix):
    inputs = json.loads(frame[FEATURES].to_json(orient="records"))
    for i, row in enumerate(inputs):
        store.log_prediction(
            {
                "request_id": f"{prefix}-{i}",
                "cid": None,
                "inputs": row,
                "probability": 0.3,
                "band": "LOW",
                "run_id": "run-1",
                "version": "v1",
            }
        )


@pytest.fixture
def drift_setup(tmp_path, profile):
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile))
    return LogStore(tmp_path / "logs"), profile_file, make_params()


def test_the_monitor_needs_enough_rows_then_reports_ok_for_normal_traffic(drift_setup):
    store, profile_file, params = drift_setup
    mon = monitor_module.DriftMonitor(store.directory, profile_file, params)
    assert mon.check()["status"] == "insufficient_data"
    log_rows(store, canonical_rows(300, seed=4), "n")
    result = mon.check()
    assert result["status"] == "ok" and result["n_rows"] == 300 and result["streak"] == 0
    assert result["mean_probability"] == pytest.approx(0.3) and result["latest_run_id"] == "run-1"


def test_an_alert_needs_three_checks_in_a_row_and_a_quiet_check_resets_the_streak(drift_setup):
    store, profile_file, params = drift_setup
    log_rows(store, shifted(), "s")
    mon = monitor_module.DriftMonitor(store.directory, profile_file, params)
    assert [mon.check()["status"] for _ in range(3)] == [
        "alert_candidate",
        "alert_candidate",
        "alert",
    ]
    assert mon.latest["streak"] == 3 and mon.latest["drifted_features"]
    log_rows(store, canonical_rows(500, seed=5), "later")  # the window is now mostly normal again
    assert mon.check()["status"] in ("ok", "warn") and mon.streak == 0


def test_the_warn_level_sits_between_ok_and_alert(drift_setup):
    store, profile_file, params = drift_setup
    log_rows(store, canonical_rows(300, seed=4), "n")
    params.drift["psi_warn"] = 0.0
    params.drift["psi_alert"] = 999
    assert (
        monitor_module.DriftMonitor(store.directory, profile_file, params).check()["status"]
        == "warn"
    )


def test_optional_inputs_missing_from_the_log_do_not_break_the_check(drift_setup):
    store, profile_file, params = drift_setup
    frame = canonical_rows(300, seed=4).drop(columns=["total_charges"])
    log_rows(store, frame.assign(total_charges=np.nan), "n")
    assert (
        monitor_module.DriftMonitor(store.directory, profile_file, params).check()["status"]
        != "insufficient_data"
    )


def test_the_monitor_service_exposes_status_check_health_and_metrics(drift_setup):
    store, profile_file, params = drift_setup
    log_rows(store, shifted(), "s")
    app = monitor_module.create_app(store.directory, profile_file, params, start_loop=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/status").json()["status"] == "insufficient_data"  # no check yet
        assert client.post("/check").json()["status"] == "alert_candidate"
        assert client.get("/status").json()["history"][-1]["status"] == "alert_candidate"
        text = client.get("/metrics").text
        assert (
            "churnsense_drift_status 2.0" in text
            and 'churnsense_drift_psi{feature="contract"}' in text
        )
        assert "churnsense_drift_window_rows 300.0" in text and "churnsense_drift_max_psi" in text


def test_a_check_before_any_traffic_does_not_crash_the_service(tmp_path, profile):
    (tmp_path / "p.json").write_text(json.dumps(profile))
    app = monitor_module.create_app(
        tmp_path / "none", tmp_path / "p.json", make_params(), start_loop=False
    )
    with TestClient(app) as client:
        assert client.post("/check").json()["status"] == "insufficient_data"
        assert client.get("/metrics").status_code == 200


def test_the_recorded_timestamps_are_utc_isoformat(tmp_path):
    store = LogStore(tmp_path)
    store.log_prediction(record("r1", store=store))
    stamp = read_recent(tmp_path, "predictions", 1)[0]["ts"]
    assert abs(datetime.fromisoformat(stamp) - datetime.now(UTC)) < timedelta(minutes=1)
