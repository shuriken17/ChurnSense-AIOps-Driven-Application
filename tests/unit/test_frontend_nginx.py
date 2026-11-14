"""Static checks of the web page and the nginx configuration."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from churnsense.api.backend import create_app
from tests.conftest import REPO

pytestmark = pytest.mark.unit
HTML = (REPO / "frontend/index.html").read_text("utf-8")
JS = (REPO / "frontend/app.js").read_text("utf-8")
NGINX = (REPO / "services/nginx/nginx.conf").read_text("utf-8")


def strip_comments(code: str) -> str:
    return re.sub(r"/\*.*?\*/", "", code, flags=re.S)


def test_the_page_shell_has_navigation_and_no_inline_code():
    for element in ("view", "nav", "model-line"):
        assert f'id="{element}"' in HTML
    assert 'lang="en"' in HTML and "<title>ChurnSense</title>" in HTML
    assert HTML.count('<li><a href="#/') == 6 and 'class="skip-link"' in HTML
    assert re.search(r"<script(?![^>]*src=)[^>]*>", HTML) is None  # the CSP forbids inline script
    assert not re.search(r"\sstyle=|<style|\son[a-z]+=", HTML)
    for name, tag in (
        ("config.js", 'src="config.js"'),
        ("lib.js", 'src="lib.js"'),
        ("app.js", 'src="app.js"'),
        ("style.css", 'href="style.css"'),
    ):
        assert tag in HTML and (REPO / "frontend" / name).is_file()
    assert HTML.index('src="config.js"') < HTML.index('src="lib.js"') < HTML.index('src="app.js"')


def test_the_page_only_calls_routes_that_exist():
    import httpx

    routes = {
        (r.path, m)
        for r in create_app(httpx.Client(base_url="http://x"), "t").routes
        for m in getattr(r, "methods", ()) or ()
    }
    calls = set(re.findall(r'\b(api|post)\("(/[a-z/]+)"', strip_comments(JS)))
    paths = {path for _, path in calls}
    assert paths >= {
        "/fields",
        "/health",
        "/model",
        "/predict",
        "/predict/batch",
        "/feedback",
        "/performance",
        "/stats",
        "/pipeline",
        "/drift",
    }
    for kind, path in calls:
        assert (f"/api/v1{path}", "POST" if kind == "post" else "GET") in routes, (kind, path)


def test_the_page_builds_text_safely_without_innerhtml():
    for code in (strip_comments(JS), strip_comments((REPO / "frontend/lib.js").read_text("utf-8"))):
        assert not re.search(
            r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function", code
        )


def test_the_page_scripts_work_with_the_content_security_policy():
    """CSP is default-src 'self': no inline scripts, no outside hosts, no inline styles."""
    for code in (
        JS,
        (REPO / "frontend/lib.js").read_text("utf-8"),
        (REPO / "frontend/style.css").read_text("utf-8"),
    ):
        assert not re.search(r"https?://(?!www\.w3\.org)", code), "external address found"
    assert 'setAttribute("style"' not in JS and "cssText" not in JS


def test_the_example_customer_is_accepted_by_the_validation():
    import json

    from churnsense.api.validation import validate_customer

    block = re.search(r"const EXAMPLE = (\{.*?\});", JS, re.S).group(1)
    example = json.loads(
        re.sub(r",\s*\}", "}", re.sub(r"(\w+):", r'"\1":', block))
    )  # JS keys -> JSON keys
    assert validate_customer(example)[1] == []


def test_nginx_redirects_http_hides_metrics_and_sets_security_headers():
    assert "return 308 https://$host:8443$request_uri" in NGINX
    assert "location = /api/v1/metrics { return 404; }" in NGINX
    for header in (
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Content-Security-Policy",
        "Referrer-Policy",
    ):
        assert header in NGINX
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in NGINX and "server_tokens off;" in NGINX
    assert "limit_req zone=api" in NGINX and "client_max_body_size 1m;" in NGINX


def test_nginx_survives_a_backend_that_is_not_up_yet():
    assert "resolver 127.0.0.11" in NGINX and "set $backend http://backend:8000;" in NGINX
    assert "proxy_pass $backend;" in NGINX  # a variable: nginx resolves it per request


@pytest.mark.skipif(shutil.which("nginx") is None, reason="nginx is not installed here")
def test_the_real_nginx_accepts_the_configuration(tmp_path):
    subprocess.run(
        ["bash", str(REPO / "scripts/make_cert.sh")],
        check=True,
        capture_output=True,
        env={"TLS_DIR": str(tmp_path / "tls"), "PATH": "/usr/bin:/bin"},
    )
    conf = (
        NGINX.replace("/etc/nginx/tls", str(tmp_path / "tls"))
        .replace("/usr/share/nginx/html", str(tmp_path))
        .replace("pid /tmp/nginx.pid", f"pid {tmp_path}/n.pid")
        .replace("/dev/stdout", str(tmp_path / "a.log"))
        .replace("/dev/stderr", str(tmp_path / "e.log"))
    )
    (tmp_path / "t.conf").write_text(conf)
    done = subprocess.run(
        ["nginx", "-t", "-c", str(tmp_path / "t.conf")], capture_output=True, text=True
    )
    assert "test is successful" in done.stderr, done.stderr


# ------------------------------------------------------------------ the frontend is its own service
def test_the_frontend_is_a_separate_image_and_service_joined_to_the_backend_only_by_a_setting():
    import yaml

    compose = yaml.safe_load((REPO / "compose.yaml").read_text("utf-8"))["services"]
    web = compose["nginx"]
    assert web["build"]["dockerfile"] == "services/frontend/Dockerfile"
    assert (
        web["image"] == "churnsense/frontend:${TAG:-dev}"
        and web["environment"]["API_BASE"] == "/api/v1"
    )
    assert web["volumes"] == [
        "/srv/churnsense/secrets/tls:/etc/nginx/tls:ro"
    ]  # no code is mounted in
    dockerfile = (REPO / "services/frontend/Dockerfile").read_text("utf-8")
    assert "COPY frontend /usr/share/nginx/html" in dockerfile and "40-config.sh" in dockerfile
    assert (
        compose["backend"]["build"]["dockerfile"] == "services/backend/Dockerfile"
    )  # a different image
    assert not (set(web) & {"command"}) and "model-server" not in str(
        web
    )  # the frontend never talks to the model


def run_config_script(tmp_path, **env):
    import os
    import subprocess

    return subprocess.run(
        ["sh", str(REPO / "services/frontend/40-config.sh")],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HTML_DIR": str(tmp_path), **env},
    )


def test_the_start_up_script_writes_the_api_address_the_page_reads(tmp_path):
    done = run_config_script(tmp_path)
    assert done.returncode == 0 and "apiBase=/api/v1" in done.stdout
    assert (tmp_path / "config.js").read_text() == (REPO / "frontend/config.js").read_text().split(
        "*/\n", 1
    )[1]
    run_config_script(tmp_path, API_BASE="https://api.example.org/v2")
    assert 'apiBase: "https://api.example.org/v2"' in (tmp_path / "config.js").read_text()


@pytest.mark.parametrize("unsafe", ['"; alert(1); "', "a b", "x'y", "$(id)", "a`b`", "x<y"])
def test_the_start_up_script_refuses_an_address_that_could_inject_code(tmp_path, unsafe):
    (tmp_path / "config.js").write_text("old")
    done = run_config_script(tmp_path, API_BASE=unsafe)
    assert done.returncode == 1 and "not allowed" in done.stderr
    assert (tmp_path / "config.js").read_text() == "old"  # nothing was written
