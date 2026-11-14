"""Health of every component of the system, probed from the backend.

Prometheus scrapes the services that export metrics (model server, backend, drift monitor,
Pushgateway). The others (MLflow, nginx, Grafana, Prometheus) export no ChurnSense metrics, so the
backend asks each of them for its health URL every few seconds and publishes the answer as
``churnsense_dependency_up{name=...}``. The retrain controller is judged by its heartbeat: the age
of its status file. Together this means every component can be seen, and alerted on, in one place.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

DEFAULT_TARGETS = {
    "model-server": ("MODEL_SERVER_URL", "http://model-server:8001", "/health"),
    "drift-monitor": ("DRIFT_URL", "http://drift-monitor:8002", "/health"),
    "mlflow": ("MLFLOW_URL", "http://mlflow:5000", "/health"),
    "nginx": ("NGINX_URL", "http://nginx", "/healthz"),
    "prometheus": ("PROMETHEUS_URL", "http://prometheus:9090", "/-/healthy"),
    "grafana": ("GRAFANA_URL", "http://grafana:3000", "/api/health"),
    "pushgateway": ("PUSHGATEWAY_BASE_URL", "http://pushgateway:9091", "/-/healthy"),
}
STATUS_FILE = "controller_status.json"


def targets_from_environment() -> dict[str, str]:
    """Health URL of every component, each base address overridable by an environment variable."""
    return {
        name: os.environ.get(variable, default).rstrip("/") + path
        for name, (variable, default, path) in DEFAULT_TARGETS.items()
    }


class ComponentProbe:
    """Probes the components and remembers the latest answers."""

    def __init__(
        self,
        targets: dict[str, str],
        state_dir: Path,
        client: httpx.Client | None = None,
        interval: float = 15.0,
    ) -> None:
        self.targets, self.state_dir, self.interval = targets, state_dir, interval
        self.client = client or httpx.Client(timeout=httpx.Timeout(1.5, connect=1.0))
        self.results: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _one(self, name: str) -> tuple[str, float]:
        try:
            return name, 1.0 if self.client.get(self.targets[name]).status_code < 500 else 0.0
        except httpx.HTTPError:
            return name, 0.0

    def probe_once(self) -> dict[str, float]:
        """Ask every component now (in parallel, so the whole round takes one timeout at most)."""
        with ThreadPoolExecutor(max_workers=max(len(self.targets), 1)) as pool:
            self.results = dict(pool.map(self._one, self.targets))
        return self.results

    def heartbeat_age(self) -> float:
        """Seconds since the controller last wrote its status file (NaN if it never did)."""
        try:
            return max(0.0, time.time() - (self.state_dir / STATUS_FILE).stat().st_mtime)
        except OSError:
            return math.nan

    def start(self) -> None:
        """Probe in a background thread until ``stop`` is called."""

        def loop() -> None:
            while not self._stop.is_set():
                self.probe_once()
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True, name="component-probe")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop.set()
