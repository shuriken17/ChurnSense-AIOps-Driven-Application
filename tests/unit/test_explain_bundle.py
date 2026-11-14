"""Unit tests for SHAP explanations (T-TR-03), the model bundle and the MLflow signature."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import shap

from churnsense.features.derive import derive_features
from churnsense.model.bundle import OUTPUT_COLUMNS
from churnsense.model.explain import additivity_error, top_factors, transformed_owners
from churnsense.model.pyfunc import model_inputs, model_signature
from churnsense.schema import FEATURE_COLUMNS, INPUT_FIELDS
from tests.factories import fitted_bundle

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", params=["lightgbm", "logreg"])
def bundle_and_rows(request):
    return fitted_bundle(request.param)


def test_t_tr_03_explanations_add_up_to_the_model_score(bundle_and_rows):
    bundle, rows = bundle_and_rows
    features = derive_features(rows)
    assert additivity_error(bundle.explainer, features) < 1e-9
    explanation = bundle.explainer.explain(features)
    assert explanation.values.shape == (len(rows), len(FEATURE_COLUMNS))
    probability = 1 / (1 + np.exp(-explanation.margin))
    np.testing.assert_allclose(probability, bundle.probabilities(features), atol=1e-9)


def test_tree_shap_equals_the_shap_library():
    bundle, rows = fitted_bundle("lightgbm")
    features = derive_features(rows).head(100)
    ours = bundle.explainer.explain(features).values
    reference = shap.TreeExplainer(bundle.estimator).shap_values(features)
    reference = reference[1] if isinstance(reference, list) else reference
    np.testing.assert_allclose(ours, reference, atol=1e-8)


def test_linear_shap_has_the_closed_form_w_times_x_minus_mean():
    bundle, rows = fitted_bundle("logreg")
    features = derive_features(rows)
    explainer = bundle.explainer
    prepared = np.asarray(bundle.estimator.named_steps["prep"].transform(features), dtype=float)
    expected_columns = (prepared - explainer.means) * explainer.weights
    owners = transformed_owners()
    first_numeric = len(owners) - 6  # numeric features come last
    np.testing.assert_allclose(
        explainer.explain(features).values.sum(axis=1), expected_columns.sum(axis=1), atol=1e-9
    )
    assert len(owners) == explainer.weights.size and first_numeric > 0


def test_an_unseen_category_still_scores_and_adds_up():
    bundle, rows = fitted_bundle("logreg")
    odd = rows.head(3).copy()
    odd["contract"] = "Weekly"
    features = derive_features(odd)
    assert features["contract"].isna().all()
    assert additivity_error(bundle.explainer, features) < 1e-9
    assert np.isfinite(bundle.probabilities(features)).all()


def test_top_factors_are_the_three_largest_effects_strongest_first(bundle_and_rows):
    bundle, rows = bundle_and_rows
    features = derive_features(rows.head(10))
    values = bundle.explainer.explain(features).values
    for i in range(10):
        factors = top_factors(values, features, i)
        assert len(factors) == 3
        impacts = [abs(f["impact"]) for f in factors]
        assert impacts == sorted(impacts, reverse=True)
        assert impacts[0] == pytest.approx(np.abs(values[i]).max(), abs=1e-6)
        for f in factors:
            assert f["direction"] == ("increases_risk" if f["impact"] > 0 else "decreases_risk")
            assert f["feature"] in FEATURE_COLUMNS


def test_top_factor_values_are_json_friendly():
    bundle, rows = fitted_bundle("lightgbm")
    features = derive_features(rows.head(1))
    factors = top_factors(bundle.explainer.explain(features).values, features, 0, k=26)
    by_name = {f["feature"]: f["value"] for f in factors}
    assert isinstance(by_name["tenure_months"], int)  # 12.0 is shown as 12
    assert isinstance(by_name["contract"], str)
    json.dumps(factors)


def test_score_gives_the_four_output_columns_and_consistent_bands(bundle_and_rows):
    bundle, rows = bundle_and_rows
    out = bundle.score(rows)
    assert tuple(out.columns) == OUTPUT_COLUMNS and len(out) == len(rows)
    assert out["churn_probability"].between(0, 1).all()
    high = out["risk_band"] == "HIGH"
    assert (out.loc[high, "churn_probability"] >= bundle.thresholds.t_high).all()
    assert (
        out.loc[out["risk_band"] == "LOW", "churn_probability"] < bundle.thresholds.t_medium
    ).all()
    assert (out["model_run_id"] == "test-run").all()
    assert all(len(json.loads(text)) == 3 for text in out["top_factors"])


def test_a_customer_gets_the_same_answer_alone_or_in_a_batch(bundle_and_rows):
    bundle, rows = bundle_and_rows
    batch = bundle.score(rows.head(20))
    for i in (0, 7, 19):
        alone = bundle.score(rows.iloc[[i]])
        assert alone["churn_probability"].iloc[0] == pytest.approx(
            batch["churn_probability"].iloc[i]
        )
        assert alone["top_factors"].iloc[0] == batch["top_factors"].iloc[i]


def test_score_does_not_need_customer_id_or_total_charges(bundle_and_rows):
    bundle, rows = bundle_and_rows
    bare = rows.head(5).drop(columns=["customer_id"]).assign(total_charges=np.nan)
    assert len(bundle.score(bare)) == 5


def test_additivity_check_method_on_the_bundle(bundle_and_rows):
    bundle, rows = bundle_and_rows
    assert bundle.additivity_check(rows, n_rows=50) < 1e-9


def test_signature_comes_from_the_schema_with_the_documented_optional_fields():
    signature = model_signature()
    inputs = {c.name: c for c in signature.inputs.inputs}
    assert list(inputs) == [f.name for f in INPUT_FIELDS] and len(inputs) == 20
    assert {n for n, c in inputs.items() if not c.required} == {"customer_id", "total_charges"}
    assert str(inputs["tenure_months"].type) == "DataType.long"
    assert str(inputs["monthly_charges"].type) == "DataType.double"
    assert [c.name for c in signature.outputs.inputs] == list(OUTPUT_COLUMNS)


def test_model_inputs_gives_the_exact_signature_types():
    frame = pd.DataFrame(
        [{f.name: ("A" if f.kind in ("category", "id") else 1) for f in INPUT_FIELDS}]
    )
    typed = model_inputs(frame.assign(monthly_charges=30, total_charges=30))
    assert typed["monthly_charges"].dtype == "float64" and typed["tenure_months"].dtype == "int64"
    assert typed["contract"].dtype == object
