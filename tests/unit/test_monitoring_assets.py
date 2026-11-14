"""Static tests of Prometheus, the alert rules, the Grafana dashboards and the compose wiring."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
import yaml

from churnsense.api.backend import create_app as backend_app
from churnsense.monitoring import dashboards
from tests.conftest import REPO

pytestmark = pytest.mark.unit
SRC = "\n".join(
    p.read_text("utf-8") for p in (REPO / "src").rglob("*.py") if "monitoring" not in str(p)
)
DEFINED = set(re.findall(r'"(churnsense_[a-z_]+)"', SRC)) | {
    f"churnsense_live_{k}" for k in ("accuracy", "precision", "recall")
}
BUILTIN = {"up", "ALERTS", "time", "vector"}


def metric_names(expression: str) -> set[str]:
    """Metric names used in a PromQL expression (histogram suffixes folded to the base name)."""
    found = set(re.findall(r"\b(churnsense_[a-z_]+)", expression))
    return {
        re.sub(r"_(bucket|sum|count)$", "", n) if n.endswith(("_bucket",)) else n for n in found
    }


def test_the_committed_dashboards_are_exactly_what_the_generator_produces():
    for name, text in dashboards.render().items():
        assert (REPO / "services/grafana/dashboards" / name).read_text("utf-8") == text, name


def test_dashboards_are_well_formed_and_use_the_provisioned_datasource():
    boards = [json.loads(t) for t in dashboards.render().values()]
    assert len(boards) == 3 and len({b["uid"] for b in boards}) == 3
    ds = yaml.safe_load(
        (REPO / "services/grafana/provisioning/datasources/prometheus.yml").read_text()
    )
    assert ds["datasources"][0]["uid"] == "prometheus" == dashboards.DS["uid"]
    for board in boards:
        ids = [p["id"] for p in board["panels"]]
        assert len(ids) == len(set(ids)) and "churnsense" in board["tags"]
        for panel in board["panels"]:
            grid = panel["gridPos"]
            assert grid["x"] + grid["w"] <= 24 and grid["w"] > 0 and grid["h"] > 0
            assert panel["datasource"]["uid"] == "prometheus" and panel["targets"]
            assert all(t["expr"].strip() for t in panel["targets"])


def test_every_chart_uses_only_metrics_that_the_code_really_exports():
    for board in (json.loads(t) for t in dashboards.render().values()):
        for panel in board["panels"]:
            for target in panel["targets"]:
                for name in metric_names(target["expr"]):
                    assert name in DEFINED, f"{board['title']} / {panel['title']}: {name}"


def test_every_alert_uses_only_metrics_that_the_code_really_exports():
    rules = yaml.safe_load((REPO / "services/prometheus/rules.yml").read_text("utf-8"))
    for rule in rules["groups"][0]["rules"]:
        for name in metric_names(rule["expr"]):
            assert name in DEFINED, f"{rule['alert']}: {name}"


def test_there_are_twelve_uniquely_named_alerts_each_with_severity_and_summary():
    rules = yaml.safe_load((REPO / "services/prometheus/rules.yml").read_text("utf-8"))["groups"][
        0
    ]["rules"]
    names = [r["alert"] for r in rules]
    assert len(names) == 12 == len(set(names))
    assert all(
        r["labels"]["severity"] in ("warning", "critical") and r["annotations"]["summary"]
        for r in rules
    )
    assert {
        "ServiceDown",
        "NoModelLoaded",
        "DriftAlert",
        "LiveRecallLow",
        "PipelineStageFailed",
        "ControllerRunFailed",
    } <= set(names)


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool is not installed")
def test_promtool_accepts_the_rules_and_the_config(tmp_path):
    rules = REPO / "services/prometheus/rules.yml"
    assert (
        subprocess.run(["promtool", "check", "rules", str(rules)], capture_output=True).returncode
        == 0
    )
    config = (
        (REPO / "services/prometheus/prometheus.yml")
        .read_text()
        .replace("/etc/prometheus/rules.yml", str(rules))
    )
    (tmp_path / "p.yml").write_text(config)
    done = subprocess.run(
        ["promtool", "check", "config", str(tmp_path / "p.yml")], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_scrape_targets_match_the_compose_services_and_the_real_metric_routes():
    prom = yaml.safe_load((REPO / "services/prometheus/prometheus.yml").read_text("utf-8"))
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    jobs = {j["job_name"]: j for j in prom["scrape_configs"]}
    assert set(jobs) == {"prometheus", "model-server", "backend", "drift-monitor", "pushgateway"}
    for name, job in jobs.items():
        host, port = job["static_configs"][0]["targets"][0].split(":")
        assert host in compose, name
    assert jobs["pushgateway"]["honor_labels"] is True  # keeps the stage and controller labels
    routes = {
        r.path for r in backend_app(__import__("httpx").Client(base_url="http://x"), "t").routes
    }
    assert jobs["backend"]["metrics_path"] in routes
    assert prom["global"]["scrape_interval"] == "15s"


def test_grafana_and_prometheus_mounts_match_the_provisioning_files():
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    provider = yaml.safe_load(
        (REPO / "services/grafana/provisioning/dashboards/dashboards.yml").read_text()
    )
    path = provider["providers"][0]["options"]["path"]
    assert f"./services/grafana/dashboards:{path}:ro" in compose["grafana"]["volumes"]
    volumes = compose["prometheus"]["volumes"]
    assert "./services/prometheus:/etc/prometheus:ro" in volumes  # a folder, not single files
    assert not any(v.endswith(("rules.yml:ro", "prometheus.yml:ro")) for v in volumes)
    makefile = (REPO / "Makefile").read_text("utf-8")
    assert "--force-recreate --wait drift-monitor" in makefile  # its params.yaml is a file mount
    assert compose["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"].startswith(
        "${GRAFANA_ADMIN_PASSWORD:?"
    )
    assert compose["grafana"]["environment"]["GF_USERS_ALLOW_SIGN_UP"] == "false"


def test_third_party_images_are_pinned_and_run_as_the_host_user_where_they_write_data():
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    for name in ("pushgateway", "prometheus", "grafana"):
        tag = compose[name]["image"].split(":")[1]
        assert tag != "latest" and re.match(r"^v?\d", tag), name
    for name in (
        "prometheus",
        "grafana",
    ):  # they write into /srv/churnsense: must not be root-owned
        assert compose[name]["user"].startswith("${HOST_UID")
    assert "--storage.tsdb.retention.size=1GB" in compose["prometheus"]["command"]


def test_the_controller_is_wired_to_everything_it_needs():
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    ctl, trainer = compose["controller"], compose["trainer"]
    assert ctl["image"] == trainer["image"] and ctl["command"][-1] == "churnsense.controller.main"
    for url in (
        "MODEL_SERVER_URL",
        "BACKEND_URL",
        "DRIFT_URL",
        "PUSHGATEWAY_URL",
        "MLFLOW_TRACKING_URI",
    ):
        assert ctl["environment"][url].startswith("http://")
    for volume in trainer["volumes"]:
        assert volume in ctl["volumes"]  # same repository, DVC remote, serving, logs, state
    assert (
        ctl["depends_on"]["mlflow"]["condition"] == "service_healthy"
        and ctl["restart"] == "unless-stopped"
    )
    assert ctl["environment"]["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert compose["backend"]["environment"]["STATE_DIR"] == "/data/state"
    assert (
        "/srv/churnsense/state:/data/state" in compose["backend"]["volumes"]
    )  # it queues console requests


def test_the_memory_budget_fits_the_5_8_gib_virtual_machine():
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]

    def mb(value):
        return int(value[:-1]) * (1024 if value.endswith("g") else 1)

    limits = {n: mb(s["mem_limit"]) for n, s in compose.items() if n != "trainer"}
    steady = sum(v for n, v in limits.items() if n != "controller")
    assert steady <= 3000, steady  # everything except the retraining burst
    assert sum(limits.values()) <= 5600  # worst case during a retraining (4 GiB swap behind it)
