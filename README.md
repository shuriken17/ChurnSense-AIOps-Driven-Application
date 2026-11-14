# DA3408 AI Operations: Final Project

**Prabhav Gupta DA24B018**

ChurnSense: AIOps Driven Churn Prediction and Retention Risk Scoring System for Subscription Services.

For a customer of a subscription service, the system gives the chance that the customer will leave, a risk level (High, Medium or Low) and the three main reasons. It also watches its own data and accuracy. It retrains the model when needed and tests the new model. It uses the new model only if the model passes the test. Everything runs on one Ubuntu virtual machine with Docker Compose. No cloud service is used.

## Files

- `src/churnsense/` the code: the nine pipeline stages, the model, the backend, the model server, the drift monitor and the retrain controller
- `tests/` unit tests, integration tests and web page tests (`tests/ui`)
- `frontend/` the web pages
- `services/` Dockerfiles, nginx, Prometheus rules and Grafana dashboards
- `scripts/` scripts for setup, data download, certificate and system check
- `params.yaml` every setting that can be tuned
- `dvc.yaml` the pipeline graph
- `MLproject`, `compose.yaml`, `Makefile`
- `data/raw/SOURCE.md` where the data comes from, with its checksum
- `docs/` five PDFs: architecture, high-level design, low-level design, test plan and report, and user manual

## Tools

Ray for the data stages, Git and DVC for the pipeline and versions, MLflow for experiments and model versions, FastAPI for the backend and the model server, nginx for the web pages, Prometheus, Pushgateway and Grafana for monitoring, and Docker Compose.

## Setup

You need Ubuntu with Docker and Docker Compose. I used a VirtualBox VM with 4 CPUs and about 6 GB of memory.

```bash
git clone https://github.com/shuriken17/ChurnSense-AIOps-Driven-Application.git
cd ChurnSense-AIOps-Driven-Application
bash scripts/setup_vm.sh      # once per machine, makes folders in /srv/churnsense, asks for sudo
make env                      # makes .env with random secrets
make data                     # downloads the IBM Telco CSV and checks it
make build                    # builds the Docker images
make dvc-init                 # sets up DVC, safe to repeat
git add -A && git commit -m "set up data tracking"    # training refuses uncommitted code
```

## Run

```bash
make mlflow-up && make repro  # starts MLflow and runs the pipeline: data, train, evaluate, gate
make promote                  # starts using the approved model
make up                       # starts all services
make check                    # 28 checks, each prints PASS or FAIL
```

Open https://localhost:8443. The browser warns about the certificate the first time, because the app makes its own. Grafana is on port 3000, Prometheus on 9090 and MLflow on 5000. `make help` shows all commands.

To try the retraining loop:

```bash
make simulate SCENARIO=drift_strong N=1000   # sends changed data, the alert comes on the third check
make retrain                                 # or press Retrain now on the Pipeline page
make rollback                                # goes back to the old model
```

To run the tests:

```bash
make lint
make test-unit
make test-integration
cd tests/ui && npm install && npm test       # web page tests
```

## Results

| Measure | Result |
|---|---|
| Model | logistic regression, validation PR-AUC 0.632 (LightGBM 0.621) |
| Golden test set | ROC-AUC 0.862, finds 89.7 percent of leavers at precision 0.50 |
| Prediction time | median 42 ms, 95th percentile 53 ms (whole backend) |
| Tests | 473 Python tests pass, 54 web page tests pass, coverage 95 percent |
| `make check` | 28 of 28 |
| Drift test | normal data PSI 0.043, changed data PSI 1.18, alert on the third check |
| Retraining | one rejected by the gate, one promoted in 393 s, roll-back worked |

The details, the test cases and the problems I found are in `docs/4_test_plan_and_report.pdf`.

## Notes

- The data is the public IBM Telco Customer Churn data with 7,043 customers.
- The model is weaker for one-year and two-year contracts, because very few of those customers left.
- The traffic simulator makes artificial outcomes. They are marked as synthetic.
- The data and the models are kept by DVC in `/srv/churnsense/dvc-remote`, not in Git.
- Not tested: a load test at production size, fairness beyond the rates for each group, and real feedback.

## AI usage

I used Claude to help write the code, the tests and the documents, and to debug problems. I ran every command on my virtual machine. I fixed the problems that came up and checked the numbers against my own runs.
