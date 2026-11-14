# ChurnSense Makefile. Run `make help` for the list of targets.
# Everything that needs Python runs INSIDE the trainer container, so the host stays clean
# (Ubuntu 24.04 forbids pip install on the host anyway: PEP 668).
SHELL := /bin/bash
.DEFAULT_GOAL := help

-include .env
TAG ?= dev
MIN_FREE_GB ?= 8
COMPOSE := docker compose --profile train
RUN := $(COMPOSE) run --rm trainer

.PHONY: feedback-repair dirs retrain alerts drift perf simulate feedback-batch smoke cert up down ps logs check promote help env data build dvc-init mlflow-up mlflow-down mlflow-logs repro push status dag lint test-unit test-integration test coverage ci shell clean-data \
	rollback test-e2e test-perf install-systemd backup

help: ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

env: ## Create .env with your user id and random secrets (does nothing if it exists)
	@scripts/make_env.sh

data: ## Download the IBM Telco CSV and verify checksum and row count
	@scripts/download_data.sh

build: ## Build the base, trainer, model-server and backend images (run `make env` first)
	@free=$$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9); \
	 if [ "$$free" -lt "$(MIN_FREE_GB)" ]; then echo "only $${free} GB free, need $(MIN_FREE_GB)"; exit 1; fi
	docker build -t churnsense/base:$(TAG) -f services/base/Dockerfile .
	$(COMPOSE) build trainer model-server backend drift-monitor nginx

dvc-init: ## DVC init (once), local remote, track the raw data and the feedback folder; safe to repeat
	$(RUN) bash -c 'set -e; mkdir -p data/feedback; [ -d .dvc ] || dvc init; dvc config core.analytics false; \
	 dvc remote add -d -f localstore /srv/churnsense/dvc-remote; \
	 dvc add data/raw/telco_churn.csv data/feedback'

dirs: ## Create the data folders of Prometheus and Grafana (once)
	@mkdir -p /srv/churnsense/prometheus /srv/churnsense/grafana

retrain: ## Run one retraining now through the controller (promotes only if the gate approves)
	docker compose exec -T controller python -m churnsense.controller.main --once --trigger manual

rollback: ## Go back to the previous model version (works even if MLflow is down)
	docker compose --profile train run --rm --no-deps trainer python -m churnsense.deploy.rollback

alerts: ## List the alerts Prometheus knows about and their state
	@curl -s localhost:9090/api/v1/alerts | jq -c '.data.alerts[] | {alert: .labels.alertname, state}'; curl -s localhost:9090/api/v1/rules | jq -r '[.data.groups[].rules[]] | "rules loaded: \(length)"'

cert: ## Create the self-signed TLS certificate for nginx (once)
	@scripts/make_cert.sh

up: cert dirs ## Start every service (HTTPS 8443, Grafana 3000, Prometheus 9090, MLflow 5000)
	docker compose up -d --wait
	@docker compose kill -s HUP prometheus > /dev/null 2>&1 || true  # pick up changed alert rules
	@# a single-file mount stays stale after a release replaces the file: recreate its container
	@docker compose up -d --no-deps --force-recreate --wait drift-monitor > /dev/null 2>&1 || true

down: ## Stop all services (data stays on disk)
	docker compose down

ps: ## Show the services and their health
	docker compose ps

logs: ## Last log lines of every service
	docker compose logs --tail=20

drift: ## Run a drift check now and print the verdict
	@docker compose exec -T drift-monitor python -c "import json, urllib.request as u; d = json.load(u.urlopen(u.Request('http://localhost:8002/check', method='POST'))); print(json.dumps({k: d[k] for k in ('status', 'n_rows', 'max_psi', 'worst_feature', 'streak', 'needed_streak')}))"

perf: ## Show live accuracy, precision and recall measured on feedback
	@curl -sk https://localhost:8443/api/v1/performance | jq -c '{n, accuracy, precision, recall}'

SCENARIO ?= normal
N ?= 300

simulate: ## Send simulated customers: make simulate SCENARIO=drift_strong N=600 (normal, drift_mild, drift_strong)
	$(RUN) python -m churnsense.simulate.traffic --scenario $(SCENARIO) --n $(N)

feedback-batch: ## Turn new feedback into a labelled batch and let DVC track it
	$(RUN) bash -c 'python -m churnsense.feedback.batcher --scenario $(SCENARIO) && dvc add data/feedback'

feedback-repair: ## Re-register feedback batch files missing from manifest.json, then let DVC track it
	$(RUN) bash -c 'python -m churnsense.feedback.batcher --repair && dvc add data/feedback'

smoke: ## Send the 20 stored smoke customers through the running backend and compare answers
	$(RUN) python -m churnsense.deploy.smoke

check: ## Print PASS or FAIL for each check of the running system
	@scripts/check_system.sh

mlflow-up: ## Start the MLflow server and wait until it is healthy
	docker compose up -d --wait mlflow

mlflow-down: ## Stop the MLflow server (data stays in /srv/churnsense/mlflow)
	docker compose stop mlflow

mlflow-logs: ## Show the last MLflow server log lines
	docker compose logs --tail=50 mlflow

repro: ## Run the DVC pipeline (only stages whose inputs changed)
	$(RUN) dvc repro

push: ## Copy the DVC-tracked data and outputs to the local remote
	$(RUN) dvc push

status: ## Show what changed since the last repro (dvc status)
	$(RUN) dvc status

dag: ## Print the pipeline DAG as Mermaid text
	$(RUN) dvc dag --mermaid

lint: ## ruff check and ruff format --check
	$(RUN) bash -c 'ruff check src tests && ruff format --check src tests'

test-unit: ## Fast tests (no services)
	$(RUN) pytest -m unit -q -p no:cacheprovider

test-integration: ## Tests that run pipeline stages with Ray
	$(RUN) pytest -m integration -q -p no:cacheprovider

test: test-unit test-integration ## All unit and integration tests

coverage: ## All tests with a coverage report
	$(RUN) pytest -q -p no:cacheprovider --cov=src/churnsense --cov-report=term-missing

ci: lint test-unit repro test-integration ## The continuous-integration run: lint, tests, pipeline

shell: ## Open a shell in the trainer container
	$(RUN) bash

promote: ## Register the gated candidate, export it to the serving folder, set the champion alias
	$(RUN) python -m churnsense.deploy.promote

clean-data: ## Delete generated data (keeps the raw data and the DVC cache)
	rm -rf data/interim data/processed data/synthetic data/quarantine reports/eda

define later
	@echo "'$(1)' is added in batch $(2)."; exit 1
endef
test-e2e test-perf: ; $(call later,$@,9)
install-systemd backup: ; $(call later,$@,7)
