#!/usr/bin/env bash
# Checks the RUNNING system and prints one PASS or FAIL line per check. Run `make up` first.
cd "$(dirname "$0")/.." || exit 1
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
no() { echo "FAIL  $1  ($2)"; fail=$((fail + 1)); }
check() { local name="$1"; shift; local out; if out=$("$@" 2>&1); then ok "$name"; else no "$name" "$(echo "$out" | head -n 1 | cut -c1-90)"; fi; }
J='content-type: application/json'
U=https://localhost:8443
TOKEN=$(grep -E '^ADMIN_TOKEN=' .env 2>/dev/null | cut -d= -f2)
CUSTOMER='{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure_months":5,"phone_service":"Yes","multiple_lines":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":79.85,"total_charges":399.25}'

healthy() { for s in mlflow model-server backend drift-monitor pushgateway prometheus grafana controller nginx; do [[ "$(docker inspect -f '{{.State.Health.Status}}' "churnsense-$s-1" 2>&1)" == healthy ]] || { echo "$s is not healthy"; return 1; }; done; }
redirect() { [[ "$(curl -s -m 8 -o /dev/null -w '%{http_code}' http://localhost:8080/)" == 308 ]]; }
page() { curl -sk -m 8 "$U/" | grep -q '<title>ChurnSense'; }
tls() { echo | openssl s_client -connect localhost:8443 -tls1_3 2>/dev/null | grep -q 'TLSv1.3' && ! echo | openssl s_client -connect localhost:8443 -tls1_1 2>/dev/null | grep -q 'Cipher is [A-Z]'; }
health() { curl -sk -m 8 "$U/api/v1/health" | jq -e '.status == "ok" and .model.version != null' > /dev/null; }
fields() { [[ "$(curl -sk -m 8 "$U/api/v1/fields" | jq length)" == 20 ]]; }
predict() { curl -sk -m 8 -X POST "$U/api/v1/predict" -H "$J" -d "$CUSTOMER" | jq -e '(.risk_band | IN("HIGH","MEDIUM","LOW")) and (.churn_probability >= 0 and .churn_probability <= 1) and (.top_factors | length == 3)' > /dev/null; }
invalid() { [[ "$(curl -sk -m 8 -o /dev/null -w '%{http_code}' -X POST "$U/api/v1/predict" -H "$J" -d '{"contract":"Weekly"}')" == 422 ]]; }
hidden() { [[ "$(curl -sk -m 8 -o /dev/null -w '%{http_code}' "$U/api/v1/metrics")" == 404 ]]; }
notoken() { [[ "$(curl -sk -m 8 -o /dev/null -w '%{http_code}' -X POST "$U/api/v1/admin/reload-model")" == 401 ]]; }
reload() { curl -sk -m 60 -X POST "$U/api/v1/admin/reload-model" -H "x-admin-token: $TOKEN" | jq -e '.status == "reloaded"' > /dev/null; }
metrics() { docker compose exec -T model-server python -c "import urllib.request as u; print(u.urlopen('http://localhost:8001/metrics').read().decode())" | grep -q churnsense_model_predictions_total; }
batch() { curl -sk -m 20 -X POST "$U/api/v1/predict/batch" -H "$J" -d "{\"customers\":[$CUSTOMER,{\"contract\":\"x\"}]}" | jq -e '.summary.scored == 1 and .summary.rejected == 1' > /dev/null; }
feedback() { local rid; rid=$(curl -sk -m 8 -X POST "$U/api/v1/predict" -H "$J" -d "$CUSTOMER" | jq -r .request_id); curl -sk -m 8 -X POST "$U/api/v1/feedback" -H "$J" -d "{\"request_id\":\"$rid\",\"churned\":true,\"source\":\"user\"}" | jq -e '.status == "recorded"' > /dev/null; }
nofeedback() { [[ "$(curl -sk -m 8 -o /dev/null -w '%{http_code}' -X POST "$U/api/v1/feedback" -H "$J" -d '{"customer_id":"nobody-here","churned":true}')" == 404 ]]; }
perf() { curl -sk -m 8 "$U/api/v1/performance" | jq -e '.n >= 1 and has("accuracy")' > /dev/null; }
pipeline() { curl -sk -m 8 "$U/api/v1/pipeline" | jq -e '.gate.decision != null' > /dev/null; }
logfile() { ls /srv/churnsense/inference-logs/predictions-*.jsonl > /dev/null 2>&1 && ! grep -q '"customer_id"' /srv/churnsense/inference-logs/predictions-*.jsonl; }
drift() { docker compose exec -T drift-monitor python -c "import json, urllib.request as u; d = json.load(u.urlopen(u.Request('http://localhost:8002/check', method='POST'))); assert d['status'] in ('insufficient_data', 'ok', 'warn', 'alert_candidate', 'alert'); print(d['status'])"; }
targets() { [[ "$(curl -s -m 8 'localhost:9090/api/v1/query?query=count(up==1)' | jq -r '.data.result[0].value[1]')" == 5 ]]; }
rules() { [[ "$(curl -s -m 8 localhost:9090/api/v1/rules | jq '[.data.groups[].rules[]] | length')" -ge 12 ]]; }
grafana() { local p; p=$(grep -E '^GRAFANA_ADMIN_PASSWORD=' .env | cut -d= -f2); [[ "$(curl -s -m 8 -u "admin:$p" 'localhost:3000/api/search?tag=churnsense' | jq length)" == 3 ]]; }
controller() { [[ -s /srv/churnsense/state/controller_status.json ]] && jq -e '.decision != null' /srv/churnsense/state/controller_status.json > /dev/null; }
components() { curl -sk -m 15 "$U/api/v1/components" | jq -e '(.components | length) >= 7 and ([.components[].up] | all)' > /dev/null; }
stages() { curl -sk -m 8 "$U/api/v1/pipeline/stages" | jq -e '(.stages | length) == 9 and (.edges | length) == 12' > /dev/null; }
mlflowapi() { docker compose exec -T model-server python -c "import urllib.request as u; assert u.urlopen('http://localhost:8001/mlflow/ping', timeout=5).status == 200"; }
reqid() { curl -skI -m 8 "$U/api/v1/health" | grep -qi '^x-request-id'; }
memory() { local bad; bad=$(docker stats --no-stream --format '{{.Name}} {{.MemPerc}}' | grep churnsense | awk '{gsub("%","",$2); if ($2 + 0 > 90) print $1 " " $2 "%"}'); [[ -z "$bad" ]] || { echo "high memory: $bad"; return 1; }; }

check "all nine services are healthy" healthy
check "http on 8080 redirects to https" redirect
check "web page is served over https" page
check "TLS 1.3 works and TLS 1.1 is refused" tls
check "health reports a loaded model" health
check "20 input fields are listed" fields
check "a valid customer is scored (band, probability, 3 factors)" predict
check "an invalid customer gets HTTP 422" invalid
check "metrics are not reachable from outside" hidden
check "model reload without a token is refused (401)" notoken
check "model reload with the admin token works" reload
check "model server metrics exist inside the network" metrics
check "batch scoring scores good rows and rejects bad ones" batch
check "feedback for a scored customer is recorded" feedback
check "feedback for an unknown customer gets HTTP 404" nofeedback
check "live performance endpoint answers" perf
check "pipeline status endpoint answers" pipeline
check "inference log is written without raw customer ids" logfile
check "drift monitor runs a check and gives a verdict" drift
check "Prometheus scrapes all 5 targets successfully" targets
check "12 alert rules are loaded" rules
check "Grafana shows the 3 ChurnSense dashboards" grafana
check "the retrain controller is watching and decided" controller
check "every component answers its health check (probed by the backend)" components
check "the pipeline console lists 9 stages and the graph" stages
check "MLflow's own scoring API is available next to ours" mlflowapi
check "every response carries a request id" reqid
check "no container is above 90 percent of its memory limit" memory
echo "---- $pass passed, $fail failed"
[[ $fail -eq 0 ]]
