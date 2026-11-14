# ruff: noqa: E501  (PromQL expressions are long single-line strings)
"""The three Grafana dashboards, generated from code so they can be tested.

Run ``python -m churnsense.monitoring.dashboards`` to rewrite the JSON files in
``services/grafana/dashboards``. A unit test checks that the committed files equal this output and
that every metric used in a chart is really defined in the source code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DS = {"type": "prometheus", "uid": "prometheus"}
OUTCOME_MAP = {
    "type": "value",
    "options": {
        "0": {"text": "failed", "color": "red"},
        "1": {"text": "rejected", "color": "orange"},
        "2": {"text": "promoted", "color": "green"},
        "3": {"text": "skipped", "color": "blue"},
    },
}
DRIFT_MAP = {
    "type": "value",
    "options": {
        "-1": {"text": "too little data", "color": "blue"},
        "0": {"text": "ok", "color": "green"},
        "1": {"text": "warn", "color": "yellow"},
        "2": {"text": "alert candidate", "color": "orange"},
        "3": {"text": "ALERT", "color": "red"},
    },
}


class Board:
    """Builds one dashboard; panels are laid out on a 24-column grid."""

    def __init__(self, uid: str, title: str) -> None:
        self.uid, self.title, self.panels, self._id = uid, title, [], 0

    def _panel(
        self,
        kind: str,
        title: str,
        targets: list[dict[str, Any]],
        pos: tuple[int, int, int, int],
        unit: str = "short",
        mappings: list | None = None,
        extra: dict | None = None,
        thresholds: list | None = None,
    ) -> None:
        self._id += 1
        x, y, w, h = pos
        defaults: dict[str, Any] = {"unit": unit, "mappings": mappings or []}
        if thresholds:
            defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        self.panels.append(
            {
                "id": self._id,
                "type": kind,
                "title": title,
                "datasource": DS,
                "gridPos": {"x": x, "y": y, "w": w, "h": h},
                "fieldConfig": {"defaults": defaults, "overrides": []},
                "targets": [
                    {"datasource": DS, "refId": chr(65 + i), **t} for i, t in enumerate(targets)
                ],
                **(extra or {}),
            }
        )

    def timeseries(self, title, queries, pos, unit="short"):
        """A line chart; ``queries`` is a list of (expression, legend) pairs."""
        self._panel(
            "timeseries",
            title,
            [{"expr": e, "legendFormat": legend} for e, legend in queries],
            pos,
            unit,
        )

    def stat(self, title, expr, pos, unit="short", mappings=None, thresholds=None, legend=""):
        """One big number (or word, with value mappings)."""
        self._panel(
            "stat",
            title,
            [{"expr": expr, "legendFormat": legend}],
            pos,
            unit,
            mappings,
            {"options": {"colorMode": "background", "graphMode": "none"}},
            thresholds,
        )

    def bars(self, title, expr, pos, legend, unit="short"):
        """A bar gauge of the current value per series."""
        self._panel(
            "bargauge",
            title,
            [{"expr": expr, "legendFormat": legend, "instant": True}],
            pos,
            unit,
            extra={"options": {"orientation": "horizontal", "displayMode": "gradient"}},
        )

    def table(self, title, exprs, pos):
        """A table of instant values (labels become columns)."""
        self._panel(
            "table", title, [{"expr": e, "format": "table", "instant": True} for e in exprs], pos
        )

    def build(self) -> dict[str, Any]:
        """The dashboard as a Grafana JSON object."""
        return {
            "uid": self.uid,
            "title": self.title,
            "schemaVersion": 39,
            "version": 1,
            "editable": False,
            "refresh": "10s",
            "time": {"from": "now-30m", "to": "now"},
            "tags": ["churnsense"],
            "timezone": "browser",
            "panels": self.panels,
        }


def health_board() -> dict[str, Any]:
    """Dashboard 1: are the services up, fast and error-free?"""
    b = Board("churnsense-health", "ChurnSense: service health")
    ok = [{"color": "red", "value": None}, {"color": "green", "value": 4}]
    b.stat("Targets up", "sum(up)", (0, 0, 4, 4), thresholds=ok)
    b.stat(
        "API error rate (5 min)",
        'sum(rate(churnsense_api_requests_total{status=~"5.."}[5m])) / sum(rate(churnsense_api_requests_total[5m]))',
        (4, 0, 5, 4),
        "percentunit",
        thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.05}],
    )
    b.stat(
        "Predictions served (1 h)",
        "sum(increase(churnsense_model_predictions_total[1h]))",
        (9, 0, 5, 4),
    )
    b.table("Model in use", ["churnsense_model_info"], (14, 0, 10, 4))
    b.timeseries(
        "Prediction requests per second",
        [
            (
                'sum by (status) (rate(churnsense_api_requests_total{path="/api/v1/predict"}[1m]))',
                "HTTP {{status}}",
            )
        ],
        (0, 4, 12, 8),
        "reqps",
    )
    b.timeseries(
        "95th percentile prediction time",
        [
            (
                "histogram_quantile(0.95, sum by (le) (rate(churnsense_model_request_seconds_bucket[5m])))",
                "model server",
            ),
            (
                'histogram_quantile(0.95, sum by (le) (rate(churnsense_api_request_seconds_bucket{path="/api/v1/predict"}[5m])))',
                "API incl. validation",
            ),
        ],
        (12, 4, 12, 8),
        "s",
    )
    b.timeseries(
        "Predictions per minute by risk band",
        [("sum by (band) (increase(churnsense_model_predictions_total[1m]))", "{{band}}")],
        (0, 12, 12, 8),
    )
    b.timeseries(
        "Problems",
        [
            ("rate(churnsense_api_validation_errors_total[5m])", "rejected customer data"),
            ("rate(churnsense_api_model_server_errors_total[5m])", "model server errors"),
            ("rate(churnsense_api_log_errors_total[5m])", "inference log errors"),
        ],
        (12, 12, 12, 8),
        "ops",
    )
    b.table(
        "Components (1 = answers its health check)", ["churnsense_dependency_up"], (0, 20, 12, 7)
    )
    b.stat(
        "Controller heartbeat age",
        "churnsense_controller_heartbeat_age_seconds",
        (12, 20, 12, 7),
        "s",
        thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 300}],
    )
    return b.build()


def model_board() -> dict[str, Any]:
    """Dashboard 2: is the model still right? Drift and live accuracy, precision, recall."""
    b = Board("churnsense-model", "ChurnSense: model and drift")
    b.stat(
        "Drift status",
        "churnsense_drift_status",
        (0, 0, 5, 4),
        mappings=[DRIFT_MAP],
        thresholds=[{"color": "green", "value": None}],
    )
    b.stat(
        "Largest PSI",
        "churnsense_drift_max_psi",
        (5, 0, 4, 4),
        "none",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.1},
            {"color": "red", "value": 0.25},
        ],
    )
    b.stat("Predictions in drift window", "churnsense_drift_window_rows", (9, 0, 5, 4))
    b.stat("Labelled outcomes", "churnsense_live_feedback_rows", (14, 0, 5, 4))
    b.stat(
        "Mean predicted churn", "churnsense_drift_mean_probability", (19, 0, 5, 4), "percentunit"
    )
    b.timeseries(
        "Largest PSI against the alert level",
        [
            ("churnsense_drift_max_psi", "largest PSI"),
            ("vector(0.1)", "warn at 0.10"),
            ("vector(0.25)", "alert at 0.25"),
        ],
        (0, 4, 12, 8),
    )
    b.bars(
        "PSI per feature now (top 10)",
        "topk(10, churnsense_drift_psi)",
        (12, 4, 12, 8),
        "{{feature}}",
    )
    b.timeseries(
        "Live accuracy, precision and recall (on reported outcomes)",
        [
            ("churnsense_live_accuracy", "accuracy"),
            ("churnsense_live_precision", "precision (HIGH band)"),
            ("churnsense_live_recall", "recall (HIGH band)"),
        ],
        (0, 12, 12, 8),
        "percentunit",
    )
    b.timeseries(
        "Reported outcomes per 5 minutes",
        [("sum by (outcome) (increase(churnsense_feedback_total[5m]))", "{{outcome}}")],
        (12, 12, 12, 8),
    )
    return b.build()


def pipeline_board() -> dict[str, Any]:
    """Dashboard 3: the pipeline stages and the automatic retraining."""
    b = Board("churnsense-pipeline", "ChurnSense: pipeline and retraining")
    b.stat(
        "Last retraining run",
        "churnsense_controller_last_outcome",
        (0, 0, 5, 4),
        mappings=[OUTCOME_MAP],
        thresholds=[{"color": "blue", "value": None}],
    )
    b.stat(
        "Last run finished",
        "churnsense_controller_last_run_timestamp_seconds * 1000",
        (5, 0, 7, 4),
        "dateTimeFromNow",
    )
    b.stat(
        "Model loaded",
        "churnsense_model_loaded_timestamp_seconds * 1000",
        (12, 0, 6, 4),
        "dateTimeFromNow",
    )
    b.stat(
        "Alerts firing",
        'count(ALERTS{alertstate="firing"}) or vector(0)',
        (18, 0, 6, 4),
        thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
    )
    b.table(
        "Pipeline stages (last run)",
        [
            "churnsense_stage_last_status",
            "churnsense_stage_duration_seconds",
            "churnsense_stage_rows_per_second",
        ],
        (0, 4, 14, 9),
    )
    b.bars(
        "Throughput by stage (rows per second)",
        "churnsense_stage_rows_per_second",
        (14, 4, 10, 9),
        "{{stage}}",
    )
    b.timeseries(
        "Stage duration", [("churnsense_stage_duration_seconds", "{{stage}}")], (0, 13, 12, 8), "s"
    )
    b.timeseries(
        "Model reloads per hour",
        [("sum by (result) (increase(churnsense_model_reloads_total[1h]))", "{{result}}")],
        (12, 13, 12, 8),
    )
    b.table("Firing alerts", ['ALERTS{alertstate="firing"}'], (0, 21, 24, 6))
    return b.build()


BOARDS = {"health.json": health_board, "model.json": model_board, "pipeline.json": pipeline_board}


def render() -> dict[str, str]:
    """File name to JSON text for every dashboard."""
    return {
        name: json.dumps(build(), indent=2, sort_keys=True) + "\n" for name, build in BOARDS.items()
    }


def main() -> None:
    """Write the dashboard files next to the Grafana configuration."""
    target = Path(__file__).resolve().parents[3] / "services/grafana/dashboards"
    target.mkdir(parents=True, exist_ok=True)
    for name, text in render().items():
        (target / name).write_text(text, encoding="utf-8")
        print("wrote", target / name)


if __name__ == "__main__":
    main()
