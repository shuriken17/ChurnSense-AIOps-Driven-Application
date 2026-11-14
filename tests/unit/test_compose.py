"""Static checks of compose.yaml (the real containers are tested on the VM)."""

from __future__ import annotations

import pytest
import yaml

from tests.conftest import REPO

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def trainer() -> dict:
    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))
    assert compose["name"] == "churnsense"
    assert compose["networks"]["churnnet"]["name"] == "churnnet"
    return compose["services"]["trainer"]


def test_trainer_runs_as_the_host_user_with_the_documented_limits(trainer):
    assert trainer["profiles"] == ["train"]  # never started by a plain `docker compose up`
    assert trainer["user"].startswith("${HOST_UID") and ":${HOST_GID" in trainer["user"]
    assert trainer["shm_size"] == "1gb" and trainer["cpus"] == 3 and trainer["mem_limit"] == "2560m"
    assert trainer["logging"]["options"] == {"max-size": "10m", "max-file": "3"}


def test_the_container_user_has_a_name_even_though_the_image_has_no_such_uid(trainer):
    """DVC asks for the user name; a uid missing from /etc/passwd makes it crash.

    Regression test for: ERROR: unexpected error - 'getpwuid(): uid not found: 1000'.
    """
    environment = trainer["environment"]
    assert environment["USER"] and environment["LOGNAME"]
    assert environment["HOME"] == "/tmp/home"


def test_environment_matches_the_design(trainer):
    environment = trainer["environment"]
    assert environment["RAY_memory_usage_threshold"] == "0.85"
    assert environment["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert environment["PYTHONPATH"] == "/workspace/src"


def test_repository_and_shared_folders_are_mounted_at_the_same_paths(trainer):
    volumes = trainer["volumes"]
    assert ".:/workspace" in volumes
    for path in ("/srv/churnsense/dvc-remote", "/srv/churnsense/serving"):
        assert f"{path}:{path}" in volumes


def test_the_nginx_health_check_uses_ipv4_because_nginx_listens_on_ipv4_only():
    """'localhost' resolves to ::1 first inside the Alpine image, where nothing listens."""
    import yaml

    from tests.conftest import REPO

    nginx = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]["nginx"]
    assert "http://127.0.0.1/healthz" in nginx["healthcheck"]["test"]


def test_the_drift_monitor_and_the_shared_folders_are_wired_in_compose():
    import yaml

    from tests.conftest import REPO

    services = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    drift, backend, trainer = services["drift-monitor"], services["backend"], services["trainer"]
    assert "/srv/churnsense/inference-logs:/data/inference-logs:ro" in drift["volumes"]  # read only
    assert (
        "./models:/data/models:ro" in drift["volumes"]
        and "./params.yaml:/app/params.yaml:ro" in drift["volumes"]
    )
    assert drift["mem_limit"] == "256m" and drift["healthcheck"]["test"][-1].count("8002") == 1
    assert "/srv/churnsense/inference-logs:/data/inference-logs" in backend["volumes"]
    assert backend["environment"]["ID_HASH_SALT"].startswith("${ID_HASH_SALT:?")
    for mount in ("/srv/churnsense/inference-logs", "/srv/churnsense/state"):
        assert f"{mount}:{mount}" in trainer["volumes"]
    assert trainer["environment"]["DRIFT_URL"] == "http://drift-monitor:8002"


def test_the_backend_can_reach_the_drift_monitor_for_the_dashboard():
    import yaml

    from tests.conftest import REPO

    backend = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]["backend"]
    assert backend["environment"]["DRIFT_URL"] == "http://drift-monitor:8002"
    for name in (
        "model-server",
        "backend",
    ):  # a 1-CPU limit needs single-threaded numeric libraries
        env = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"][name][
            "environment"
        ]
        assert env["OMP_NUM_THREADS"] == "1" and env["OPENBLAS_NUM_THREADS"] == "1"
