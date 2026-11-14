#!/usr/bin/env python3
"""Score rows with an exported MLflow model in a FRESH Python process.

Used by the evaluation stage (and the deployment checks) for two reasons:

* It proves that an exported model is self-contained: the process is started WITHOUT the
  repository on the Python path, so everything the model needs must come from the model
  folder itself (its own copy of the ``churnsense`` code and its pinned libraries).
* Two different model versions each run their own code. Inside one process the second
  ``import churnsense`` would silently reuse the first one's modules.

This file deliberately imports nothing from ``churnsense``.

Usage:
    python score_model.py MODEL_DIR INPUT.parquet OUTPUT.parquet [--latency N] [--additivity N]

Prints one JSON line with load time, prediction time, optional single-row latency percentiles,
the SHAP additivity error and basic facts about the model (family, run id, thresholds).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlflow.pyfunc
import pandas as pd


def main() -> None:
    """Load the model, score the rows, print one JSON line of facts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--latency", type=int, default=0, help="number of single-row calls")
    parser.add_argument("--additivity", type=int, default=0, help="rows for the SHAP check")
    args = parser.parse_args()

    started = time.perf_counter()
    model = mlflow.pyfunc.load_model(args.model_dir)
    load_seconds = time.perf_counter() - started

    frame = pd.read_parquet(args.input)
    # parquet gives pandas "str" columns; the MLflow signature wants plain object strings
    frame = frame.astype({c: object for c in frame.select_dtypes(exclude="number").columns})

    started = time.perf_counter()
    predictions = model.predict(frame)
    predict_seconds = time.perf_counter() - started
    predictions.to_parquet(args.output)

    facts: dict[str, object] = {
        "rows": len(predictions),
        "load_seconds": round(load_seconds, 3),
        "predict_seconds": round(predict_seconds, 4),
    }
    inner = model.unwrap_python_model()
    bundle = getattr(inner, "bundle", None)
    if bundle is not None:
        facts["family"] = bundle.family
        facts["run_id"] = bundle.run_id
        facts["trained_at"] = bundle.trained_at
        facts["t_high"] = bundle.thresholds.t_high
        facts["t_medium"] = bundle.thresholds.t_medium
    if args.latency:
        times = []
        for i in range(args.latency):
            row = frame.iloc[[i % len(frame)]]
            tick = time.perf_counter()
            model.predict(row)
            times.append((time.perf_counter() - tick) * 1000)
        times.sort()
        facts["latency_ms_p50"] = round(statistics.median(times), 3)
        facts["latency_ms_p95"] = round(times[int(0.95 * len(times)) - 1], 3)
        facts["latency_ms_max"] = round(times[-1], 3)
    if args.additivity and hasattr(inner, "additivity_check"):
        facts["additivity_max_error"] = float(inner.additivity_check(frame, args.additivity))
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
