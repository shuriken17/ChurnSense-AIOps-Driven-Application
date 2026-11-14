"""MLflow packaging of the model.

``ChurnSenseModel`` is a ``mlflow.pyfunc.PythonModel`` that loads a ``ModelBundle`` and
delegates to it. The model is saved together with:

* a signature (the 20 input fields and the 4 output columns, built from ``schema.py``),
* an input example,
* the pinned ``requirements.lock`` (the libraries that trained it),
* a copy of the ``churnsense`` package (``code_paths``), so a model exported to the serving
  folder is self-contained and the serving side needs no access to the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

import churnsense
from churnsense.model.bundle import OUTPUT_COLUMNS, ModelBundle
from churnsense.schema import INPUT_FIELDS

_MLFLOW_TYPE = {
    "category": "string",
    "id": "string",
    "binary": "long",
    "int": "long",
    "float": "double",
}


class ChurnSenseModel(mlflow.pyfunc.PythonModel):
    """Thin MLflow wrapper: raw customer rows in, probability, band and factors out."""

    def load_context(self, context: Any) -> None:
        """Load the pickled ``ModelBundle`` from the model's artifacts."""
        self.bundle: ModelBundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        """Score the rows (a DataFrame with the 20 input fields).

        The ``pd.DataFrame`` hints are the form MLflow accepts without a warning; the
        signature saved with the model describes the individual columns.
        """
        return self.bundle.score(pd.DataFrame(model_input))

    def additivity_check(self, raw: pd.DataFrame, n_rows: int = 100) -> float:
        """Expose the SHAP additivity check (used by the evaluation sanity checks)."""
        return self.bundle.additivity_check(raw, n_rows)


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the 20 input fields and give them exactly the types the signature expects.

    Strings become plain ``object`` columns (parquet gives pandas ``str``), whole-number fields
    ``int64`` and the money fields ``float64``.
    """
    inputs = frame[[f.name for f in INPUT_FIELDS]].copy()
    for spec in INPUT_FIELDS:
        wanted = {"string": "object", "long": "int64", "double": "float64"}[_MLFLOW_TYPE[spec.kind]]
        inputs[spec.name] = inputs[spec.name].astype(wanted)
    return inputs


def model_signature() -> ModelSignature:
    """Input and output schema of the model, generated from the canonical field list.

    ``customer_id`` and ``total_charges`` are optional inputs, exactly as in the API contract.
    """
    inputs = [ColSpec(_MLFLOW_TYPE[f.kind], f.name, required=f.required) for f in INPUT_FIELDS]
    outputs = [
        ColSpec("double", OUTPUT_COLUMNS[0]),
        ColSpec("string", OUTPUT_COLUMNS[1]),
        ColSpec("string", OUTPUT_COLUMNS[2]),
        ColSpec("string", OUTPUT_COLUMNS[3]),
    ]
    return ModelSignature(inputs=Schema(inputs), outputs=Schema(outputs))


def model_kwargs(bundle_file: Path, example: pd.DataFrame, requirements: Path) -> dict[str, Any]:
    """Arguments for ``mlflow.pyfunc.save_model`` and ``log_model`` (so both are identical)."""
    package_dir = Path(churnsense.__file__).resolve().parent
    return {
        "python_model": ChurnSenseModel(),
        "artifacts": {"bundle": str(bundle_file)},
        "signature": model_signature(),
        "input_example": example,
        "pip_requirements": str(requirements),
        "code_paths": [str(package_dir)],
    }
