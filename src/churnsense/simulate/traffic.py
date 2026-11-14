"""Traffic simulator: sends realistic and drifted customers to the running API.

For demos, tests and the drift-to-retrain loop. Scenarios:

* ``normal``       customers resampled from the real data (no drift)
* ``drift_mild``   a moderate shift: more month-to-month, higher bills
* ``drift_strong`` a large shift in who the customers are AND in what makes them leave
                   (concept drift: fiber and senior customers become much likelier to churn)

The outcome labels sent as feedback are SYNTHETIC: drawn from a stated rule, not from real
customers, and marked ``source: simulator`` so every batch built from them is labelled
``synthetic_labels: true``. The rule is deliberately independent of the model.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Any

import httpx
import numpy as np
import pandas as pd

from churnsense.config import default_paths
from churnsense.schema import INPUT_COLUMNS, enforce_cross_field

SCENARIOS = ("normal", "drift_mild", "drift_strong")
FEATURES = [c for c in INPUT_COLUMNS if c != "customer_id"]


def apply_scenario(frame: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``frame`` shifted according to the scenario (rules stay valid)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; use one of {SCENARIOS}")
    out = frame.copy().reset_index(drop=True)
    if scenario == "normal":
        return out
    share = {"drift_mild": 0.4, "drift_strong": 0.85}[scenario]
    bill = {"drift_mild": 1.15, "drift_strong": 1.35}[scenario]
    pick = lambda p: rng.random(len(out)) < p  # noqa: E731 - tiny local helper
    out.loc[pick(share), "contract"] = "Month-to-month"
    out.loc[pick(share), "payment_method"] = "Electronic check"
    out["monthly_charges"] = (out["monthly_charges"] * bill).clip(upper=118.75).round(2)
    if scenario == "drift_strong":
        young = pick(0.7)
        out.loc[young, "tenure_months"] = (out.loc[young, "tenure_months"] // 6).astype(int)
        fiber = pick(0.75)
        out.loc[fiber & (out["internet_service"] == "No"), "phone_service"] = "Yes"
        out.loc[fiber, "internet_service"] = "Fiber optic"
    out = enforce_cross_field(out)
    out["total_charges"] = (out["tenure_months"] * out["monthly_charges"]).round(2)
    return out


def true_churn_probability(frame: pd.DataFrame, scenario: str) -> np.ndarray:
    """The synthetic ground truth: a stated logistic rule, independent of the model."""
    logit = (
        -1.6  # intercept calibrated: normal traffic churns ~26.5% like the real data
        + 1.35 * (frame["contract"] == "Month-to-month")
        - 0.9 * (frame["contract"] == "Two year")
        - 0.03 * frame["tenure_months"]
        + 0.6 * (frame["internet_service"] == "Fiber optic")
        + 0.4 * (frame["payment_method"] == "Electronic check")
        + 0.006 * (frame["monthly_charges"] - 65.0)
    )
    if scenario == "drift_strong":  # concept drift: the old relationship no longer holds
        logit = (
            logit
            + 0.9 * (frame["internet_service"] == "Fiber optic")
            + 0.8 * frame["senior_citizen"]
        )
    return (1.0 / (1.0 + np.exp(-logit))).to_numpy()


def base_customers(validated_file: Any) -> pd.DataFrame:
    """Real customers (without id) to resample from."""
    real = pd.read_parquet(validated_file)
    real = real[real["source"] == "real"]
    return real[FEATURES].reset_index(drop=True)


def run(
    backend_url: str,
    scenario: str,
    n: int,
    seed: int,
    with_feedback: bool,
    base: pd.DataFrame,
    client: httpx.Client | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    """Send ``n`` customers (and, if asked, their synthetic outcomes) to the backend."""
    rng = np.random.default_rng(seed)
    tag = run_tag or uuid.uuid4().hex[:6]  # a repeated run is NEW customers, not the same ones
    sample = base.sample(n=n, replace=True, random_state=seed)
    customers = apply_scenario(sample, scenario, rng)
    p_true = true_churn_probability(customers, scenario)
    labels = (rng.random(n) < p_true).astype(int)
    bands = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    scored = rejected = fed = 0
    http = client or httpx.Client(base_url=backend_url.rstrip("/"), timeout=20.0)
    with http as client:
        for i, row in customers.iterrows():
            customer_id = f"sim-{scenario}-{seed}-{tag}-{i}"
            payload = {"customer_id": customer_id, **json.loads(row.to_json())}
            answer = client.post("/api/v1/predict", json=payload)
            if answer.status_code != 200:
                rejected += 1
                continue
            scored += 1
            bands[answer.json()["risk_band"]] += 1
            if with_feedback:
                sent = client.post(
                    "/api/v1/feedback",
                    json={
                        "customer_id": customer_id,
                        "churned": bool(labels[i]),
                        "source": "simulator",
                    },
                )
                fed += sent.status_code == 200
    return {
        "scenario": scenario,
        "sent": n,
        "scored": scored,
        "rejected": rejected,
        "feedback_sent": int(fed),
        "bands": bands,
        "true_churn_rate": round(float(labels.mean()), 4),
        "synthetic_labels": with_feedback,
    }


def main() -> None:
    """Command line: ``python -m churnsense.simulate.traffic --scenario drift_strong --n 600``."""
    parser = argparse.ArgumentParser(description="Send simulated customers to the API.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument(
        "--backend-url", default=os.environ.get("BACKEND_URL", "http://backend:8000")
    )
    args = parser.parse_args()
    base = base_customers(default_paths().validated)
    print(
        json.dumps(
            run(args.backend_url, args.scenario, args.n, args.seed, not args.no_feedback, base)
        )
    )


if __name__ == "__main__":
    main()
