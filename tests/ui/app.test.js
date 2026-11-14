"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const { boot, closeAll, FIELDS, FRONTEND, STAGES, EDGES } = require("./harness.js");

test.after(closeAll);

const IBM_HEADER = "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn";
const ibmRow = (id, churn = "Yes", tenure = "5") =>
  `${id},Female,0,Yes,No,${tenure},Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,79.85,399.25,${churn}`;

// ------------------------------------------------------------------ shell and routing
test("the shell has skip link, five nav links, the current page marked, and the model line", async () => {
  const t = boot();
  await t.settle();
  assert.equal(t.$$("#nav a").length, 6);
  assert.equal(t.$(".skip-link").getAttribute("href"), "#view");
  assert.equal(t.$('#nav a[aria-current="page"]').textContent, "Dashboard");
  assert.match(t.$("#model-line").textContent, /Model v1 is serving \(logistic regression\), trained/);
  assert.equal(t.document.documentElement.lang, "en");
});

test("navigating changes page, title, current link and moves focus to the content", async () => {
  const t = boot();
  await t.settle();
  await t.go("#/score");
  assert.equal(t.document.title, "Score a customer | ChurnSense");
  assert.equal(t.$('#nav a[aria-current="page"]').getAttribute("href"), "#/score");
  assert.equal(t.document.activeElement.id, "view");
  await t.go("#/does-not-exist");
  assert.equal(t.document.title, "Dashboard | ChurnSense");
  assert.equal(t.$('#nav a[aria-current="page"]').getAttribute("href"), "#/");
});

test("the model line says so when the service is down or has no model", async () => {
  const off = boot({ "GET /health": "network-error" });
  await off.settle();
  assert.equal(off.$("#model-line").textContent, "The service cannot be reached");
  const none = boot({ "GET /health": { status: 503, body: { status: "degraded", model: null } } });
  await none.settle();
  assert.equal(none.$("#model-line").textContent, "No model is available right now");
});

test("the interface never builds markup from text (no innerHTML, no eval)", () => {
  for (const file of ["app.js", "lib.js"]) {
    const code = fs.readFileSync(path.join(FRONTEND, file), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");  // comments may name the rule
    assert.doesNotMatch(code, /innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function/, file);
  }
  const html = fs.readFileSync(path.join(FRONTEND, "index.html"), "utf8");
  assert.doesNotMatch(html, /<script(?![^>]*src=)[^>]*>|\sstyle=|\son[a-z]+=/);
});

// ------------------------------------------------------------------ dashboard
test("the dashboard fills six cards from the API", async () => {
  const t = boot();
  await t.settle();
  assert.equal(t.$$(".card").length, 6);
  const text = t.$("#view").textContent;
  assert.match(text, /Version v1 \(logistic regression\)/);
  assert.match(text, /0\.862/);                       // golden ROC-AUC
  assert.match(text, /77\.0 %/);                      // live accuracy
  assert.match(text, /56\.0 %/);
  assert.match(text, /88\.0 %/);
  assert.match(text, /200\s*predictions/);
  assert.match(text, /ALERT: large shift/);
  assert.match(text, /charge_delta/);
  assert.match(text, /REJECT/);
  assert.match(text, /Last retraining: promoted \(drift\)/);
  assert.equal(t.$$(".card a[target=_blank]").length, 3);
  assert.ok(t.$$(".card a").every((a) => a.getAttribute("rel") === "noopener"));
  assert.ok(t.$("svg.spark polyline"));
});

test("the live performance card explains when there are too few outcomes", async () => {
  const t = boot({ "GET /performance": { body: { n: 7, min_samples: 30, enough_data: false } } });
  await t.settle();
  assert.match(t.$("#view").textContent, /Needs 30 reported outcomes.*Now: 7/);
});

test("each card fails on its own with a plain message", async () => {
  const t = boot({ "GET /drift": { status: 503, body: { error: { code: "http_503", message: "x", details: [] } } }, "GET /stats": "network-error" });
  await t.settle();
  const cards = t.$$(".card").map((c) => c.textContent);
  assert.ok(cards.some((c) => /Data drift[\s\S]*drift monitor is not available/.test(c)));
  assert.ok(cards.some((c) => /Predictions[\s\S]*cannot be reached/.test(c)));
  assert.ok(cards.some((c) => /Model in use[\s\S]*Version v1/.test(c)));   // the others still work
});

test("the dashboard refreshes by itself and stops when you leave it", async () => {
  const t = boot();
  await t.settle();
  const app = t.window.ChurnApp;
  assert.ok(app.state.timer, "a refresh timer runs on the dashboard");
  await t.go("#/about");
  assert.equal(app.state.timer, null, "the timer is cleared on other pages");
});

test("PSI bars carry text, not only colour, and a bad value cannot break the layout", async () => {
  const t = boot({ "GET /drift": { body: { status: "warn", n_rows: 300, enough_data: true, max_psi: 99, worst_feature: "x", streak: 0, needed_streak: 3,
    psi_by_feature: { x: 99, y: -1 }, history: [] } } });
  await t.settle();
  const widths = t.$$(".bar > span").map((s) => s.style.width);
  assert.ok(widths.every((w) => parseFloat(w) >= 0 && parseFloat(w) <= 100), widths.join());
  assert.match(t.$("#view").textContent, /99\.00 large shift/);
});

// ------------------------------------------------------------------ score one customer
test("the score form has all 20 fields in three groups and an example", async () => {
  const t = boot();
  await t.go("#/score");
  assert.equal(t.$$("#customer-form input, #customer-form select").length, 20);
  assert.equal(t.$$("fieldset legend").map((l) => l.textContent).join("|"), "Customer|Phone and internet|Contract and billing");
  assert.ok(t.$$("label").every((l) => l.getAttribute("for") && t.document.getElementById(l.getAttribute("for"))));
  t.click(t.$("#example-button"));
  assert.equal(t.$("#f-contract").value, "Month-to-month");
  assert.equal(t.$("#f-monthly_charges").value, "79.85");
});

test("scoring sends typed numbers and shows band, chance, a meter and plain reasons", async () => {
  const t = boot();
  await t.go("#/score");
  t.click(t.$("#example-button"));
  t.type(t.$("#f-customer_id"), "CUST-1");
  t.submit(t.$("#customer-form"));
  await t.settle();
  const sent = t.calls.find((c) => c.key === "POST /predict").body;
  assert.equal(sent.tenure_months, 5);
  assert.equal(typeof sent.monthly_charges, "number");
  assert.equal(sent.customer_id, "CUST-1");
  assert.equal(sent.gender, "Female");
  const result = t.$("#result");
  assert.match(result.textContent, /High risk/);
  assert.match(result.textContent, /Chance of churn: 70\.2 %/);
  assert.match(result.textContent, /Contact this customer soon/);
  const reasons = t.$$("#result ol.reasons li").map((li) => li.textContent);
  assert.equal(reasons.length, 3);
  assert.match(reasons[0], /Is on a month-to-month contract \(▲ raises the risk\)/);
  assert.match(reasons[2], /Does not use fiber optic internet \(▼ lowers the risk\)/);
  assert.equal(t.$$("#result .mark").length, 2);
  assert.equal(t.$("#result .fill").style.width, "70.2%");
  assert.match(result.textContent, /Model v1 \(run abcdef12\), request req-123/);
});

test("the band is told by words and a symbol, not by colour alone", async () => {
  for (const [band, symbol] of [["HIGH", "▲"], ["MEDIUM", "■"], ["LOW", "▼"]]) {
    const t = boot({ "POST /predict": () => ({ body: { customer_id: null, churn_probability: 0.1, risk_band: band, top_factors: [], model: { version: "v1", run_id: "abcdef1234" }, request_id: "r" } }) });
    await t.go("#/score");
    t.click(t.$("#example-button"));
    t.submit(t.$("#customer-form"));
    await t.settle();
    assert.ok(t.$(".result-band").textContent.startsWith(symbol), band);
  }
});

test("missing required fields are flagged in place and nothing is sent", async () => {
  const t = boot();
  await t.go("#/score");
  t.submit(t.$("#customer-form"));
  await t.settle();
  assert.equal(t.calls.filter((c) => c.key === "POST /predict").length, 0);
  assert.equal(t.$("#f-gender").getAttribute("aria-invalid"), "true");
  assert.match(t.$("#e-gender").textContent, /is required/);
  assert.equal(t.$(".alert-box").getAttribute("role"), "alert");
  assert.equal(t.$("#f-customer_id").getAttribute("aria-invalid"), null);   // optional fields are fine empty
});

test("server-side validation errors are shown next to the right fields", async () => {
  const t = boot({ "POST /predict": { status: 422, body: { error: { code: "invalid_customer", message: "x", details: [
    { field: "tenure_months", message: "must be at most 120" }, { field: "(combination)", message: "phone_service and multiple_lines disagree" }] } } } });
  await t.go("#/score");
  t.click(t.$("#example-button"));
  t.submit(t.$("#customer-form"));
  await t.settle();
  assert.equal(t.$("#f-tenure_months").getAttribute("aria-invalid"), "true");
  assert.match(t.$("#e-tenure_months").textContent, /at most 120/);
  assert.match(t.$(".alert-box").textContent, /phone_service and multiple_lines disagree/);
  assert.equal(t.$("#result"), null);
});

test("a down prediction service gives a plain message, and Clear resets everything", async () => {
  const t = boot({ "POST /predict": { status: 503, body: { error: { code: "http_503", message: "no model is loaded", details: [] } } } });
  await t.go("#/score");
  t.click(t.$("#example-button"));
  t.submit(t.$("#customer-form"));
  await t.settle();
  assert.match(t.$(".alert-box").textContent, /prediction service is not available/);
  t.click([...t.$$("button")].find((b) => b.textContent === "Clear"));
  assert.equal(t.$("#f-contract").value, "");
  assert.equal(t.$(".alert-box").hidden, true);
});

test("reporting what happened sends the request id and thanks the user", async () => {
  const t = boot();
  await t.go("#/score");
  t.click(t.$("#example-button"));
  t.submit(t.$("#customer-form"));
  await t.settle();
  const [churned, stayed] = t.$$("#result button");
  t.click(churned);
  await t.settle();
  assert.deepEqual(t.calls.find((c) => c.key === "POST /feedback").body, { request_id: "req-123", churned: true });
  assert.equal(churned.getAttribute("aria-pressed"), "true");
  assert.match(t.$("#result").textContent, /Recorded: the customer churned/);
  assert.equal(stayed.disabled, false);
});

test("text from the server is shown as text and can never become markup", async () => {
  const evil = "<img src=x onerror=alert(1)>";
  const t = boot({ "POST /predict": () => ({ body: { customer_id: evil, churn_probability: 0.5, risk_band: "HIGH",
    top_factors: [{ feature: "contract", value: evil, impact: 0.3, direction: "increases_risk" }], model: { version: "<b>v</b>", run_id: "abcdef1234" }, request_id: "<i>r</i>" } }) });
  await t.go("#/score");
  t.click(t.$("#example-button"));
  t.submit(t.$("#customer-form"));
  await t.settle();
  assert.equal(t.$("#result img"), null);
  assert.equal(t.$("#result b"), null);
  assert.equal(t.$("#result i"), null);
  assert.ok(t.$("#result").textContent.includes(evil));
});

// ------------------------------------------------------------------ score a file
async function pasteAndCheck(t, csv) {
  await t.go("#/batch");
  t.type(t.$("#csv-text"), csv);
  t.click(t.$("#check-csv"));
  await t.settle();
}

test("an IBM-style file is recognised, with the outcome column noticed and not used for scoring", async () => {
  const t = boot();
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A-1"), ibmRow("A-2", "No")].join("\n"));
  const summary = t.$("#batch-summary").textContent;
  assert.match(summary, /2 customers found\. 20 columns recognised, ignored: Churn/);
  assert.match(summary, /A Churn column was found/);
  assert.equal(t.$("#run-batch").textContent, "Score 2 customers");
  t.click(t.$("#run-batch"));
  await t.settle();
  const sent = t.calls.find((c) => c.key === "POST /predict/batch").body.customers;
  assert.equal(sent.length, 2);
  assert.equal(sent[0].tenure_months, 5);              // "tenure" mapped to tenure_months, as a number
  assert.equal(sent[0].customer_id, "A-1");
  assert.ok(!("Churn" in sent[0]) && !("churn" in sent[0]));
});

test("missing columns, too short data and too many rows are explained", async () => {
  const t = boot();
  await pasteAndCheck(t, "gender,contract\nFemale,One year");
  assert.match(t.$(".alert-box").textContent, /required columns are missing: .*tenure_months/);
  await pasteAndCheck(t, IBM_HEADER);
  assert.match(t.$(".alert-box").textContent, /header line and at least one customer row/);
  const many = [IBM_HEADER, ...Array.from({ length: 5001 }, (_, i) => ibmRow("R-" + i))].join("\n");
  await pasteAndCheck(t, many);
  assert.match(t.$(".alert-box").textContent, /limit is 5000/);
});

test("unreadable numbers are caught in the browser and reported as problems, not sent", async () => {
  const t = boot();
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A-1"), ibmRow("A-2", "No", "five")].join("\n"));
  assert.match(t.$("#batch-summary").textContent, /1 rows have unreadable numbers/);
  t.click(t.$("#run-batch"));
  await t.settle();
  assert.equal(t.calls.find((c) => c.key === "POST /predict/batch").body.customers.length, 1);
  const rows = t.$$("tbody tr").map((r) => r.textContent);
  assert.ok(rows.some((r) => /tenure_months must be a number, found 'five'/.test(r)));
  assert.equal(t.$$("#view .kpi .value")[1].textContent, "1");  // one rejected
});

test("a large file is sent in chunks of 500 and the results are paged, sorted and filtered", async () => {
  const t = boot();
  const rows = Array.from({ length: 1200 }, (_, i) => ibmRow("C-" + i));
  await pasteAndCheck(t, [IBM_HEADER, ...rows].join("\n"));
  t.click(t.$("#run-batch"));
  await t.settle(150);
  const batches = t.calls.filter((c) => c.key === "POST /predict/batch").map((c) => c.body.customers.length);
  assert.deepEqual(batches, [500, 500, 200]);
  assert.match(t.$(".pager").textContent, /Page 1 of 48 \(1200 rows\)/);
  assert.equal(t.$$("tbody tr").length, 25);
  const probs = t.$$("tbody tr").map((r) => parseFloat(r.children[2].textContent));
  assert.deepEqual(probs, [...probs].sort((a, b) => b - a));            // highest chance first
  t.click([...t.$$(".pager button")].find((b) => b.textContent === "Next"));
  assert.match(t.$(".pager").textContent, /Page 2 of 48/);
  t.click([...t.$$("[aria-label='Filter by risk band'] button")].find((b) => b.textContent === "High"));
  assert.ok(t.$$("tbody tr").every((r) => /High risk/.test(r.children[3].textContent)));
  assert.match(t.$(".pager").textContent, /Page 1 of /);
  assert.equal(t.$("[aria-label='Filter by risk band'] button[aria-pressed=true]").textContent, "High");
  t.click([...t.$$(".row button")].find((b) => /Sort by file row/.test(b.textContent)));
  const rowNumbers = t.$$("tbody tr").map((r) => Number(r.children[0].textContent));
  assert.deepEqual(rowNumbers, [...rowNumbers].sort((a, b) => a - b));
});

test("rows the server rejects are listed with their reason and can be filtered", async () => {
  const t = boot({ "POST /predict/batch": (body) => ({ body: { model: {}, summary: {}, results: body.customers.map((c, i) => i === 1
    ? { index: i, error: { code: "invalid_customer", details: [{ field: "contract", message: "must be one of: Month-to-month" }] } }
    : { index: i, customer_id: c.customer_id, churn_probability: 0.4, risk_band: "MEDIUM", top_factors: [] }) } }) });
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A"), ibmRow("B"), ibmRow("C")].join("\n"));
  t.click(t.$("#run-batch"));
  await t.settle();
  t.click([...t.$$("[aria-label='Filter by risk band'] button")].find((b) => b.textContent === "Problems"));
  const only = t.$$("tbody tr");
  assert.equal(only.length, 1);
  assert.match(only[0].textContent, /^3B.*contract must be one of: Month-to-month/);   // file row 3 (the header is row 1)
});

test("the results can be downloaded as a CSV that is safe to open in a spreadsheet", async () => {
  const t = boot({ "POST /predict/batch": (body) => ({ body: { model: {}, summary: {}, results: body.customers.map((c, i) => ({ index: i, customer_id: "=cmd|' /C calc'!A0",
    churn_probability: 0.5, risk_band: "HIGH", top_factors: [{ feature: "contract", value: "Two, year", impact: 0.1, direction: "increases_risk" }] })) } }) });
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A")].join("\n"));
  t.click(t.$("#run-batch"));
  await t.settle();
  t.click([...t.$$("button")].find((b) => b.textContent === "Download results (CSV)"));
  const blob = t.downloads.find((d) => d.text !== undefined);
  assert.match(blob.text, /^file_row,customer_id,churn_probability,risk_band,reason_1,reason_2,reason_3,problem\r\n/);
  assert.match(blob.text, /'=cmd\|/);                    // formula neutralised
  assert.match(blob.text, /"Has contract: Two, year"/);  // comma quoted
  assert.ok(t.downloads.some((d) => d.clicked === "churnsense-results.csv"));
});

test("known outcomes in the file can be sent as feedback, only for customers with an id", async () => {
  const t = boot();
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A-1", "Yes"), ibmRow("A-2", "No"), ibmRow("A-3", "unknown")].join("\n"));
  t.click(t.$("#run-batch"));
  await t.settle();
  const button = [...t.$$("button")].find((b) => /known outcomes as feedback/.test(b.textContent));
  assert.equal(button.textContent, "Send 2 known outcomes as feedback");
  t.click(button);
  await t.settle(100);
  const sent = t.calls.filter((c) => c.key === "POST /feedback").map((c) => c.body);
  assert.deepEqual(sent.map((b) => [b.customer_id, b.churned, b.source]).sort(), [["A-1", true, "user"], ["A-2", false, "user"]]);
  assert.match(t.$("#view").textContent, /Outcomes sent: 2 recorded, 0 customers not found/);
});

test("an uploaded file is read and checked like pasted text", async () => {
  const t = boot();
  await t.go("#/batch");
  const csv = [IBM_HEADER, ibmRow("F-1")].join("\n");
  Object.defineProperty(t.$("#csv-file"), "files", { value: [{ text: async () => csv }] });
  t.$("#csv-file").dispatchEvent(new t.window.Event("change", { bubbles: true }));
  await t.settle();
  assert.match(t.$("#batch-summary").textContent, /1 customers found/);
  assert.ok(t.$("#csv-text").value.startsWith("customerID,"));
});

test("a failing prediction service during a file run stops with a clear message", async () => {
  const t = boot({ "POST /predict/batch": { status: 503, body: { error: { code: "http_503", message: "no model is loaded", details: [] } } } });
  await pasteAndCheck(t, [IBM_HEADER, ibmRow("A")].join("\n"));
  t.click(t.$("#run-batch"));
  await t.settle();
  assert.match(t.$("#batch-summary").textContent, /prediction service is not available/);
  assert.equal(t.$("#run-batch").disabled, false);   // can try again
});

// ------------------------------------------------------------------ report an outcome, about
test("the feedback page validates, reports success and explains an unknown customer", async () => {
  const t = boot();
  await t.go("#/feedback");
  t.submit(t.$("form"));
  assert.match(t.$(".alert-box").textContent, /give the customer id and choose/);
  t.type(t.$("#fb-id"), " C-77 ");
  t.$("#fb-churned").checked = true;
  t.submit(t.$("form"));
  await t.settle();
  assert.deepEqual(t.calls.find((c) => c.key === "POST /feedback").body, { customer_id: "C-77", churned: true, source: "user" });
  assert.match(t.$(".info-box").textContent, /HIGH band, so this counts as true positive/);
  assert.equal(t.$("#fb-id").value, "");   // cleared after success
});

test("an unknown id on the feedback page says what to do", async () => {
  const t = boot({ "POST /feedback": { status: 404, body: { error: { code: "unknown_prediction", message: "x", details: [] } } } });
  await t.go("#/feedback");
  t.type(t.$("#fb-id"), "ghost");
  t.$("#fb-stayed").checked = true;
  t.submit(t.$("form"));
  await t.settle();
  assert.match(t.$(".alert-box").textContent, /Score the customer first, using the same id/);
  assert.equal(t.$("#fb-id").value, "ghost");   // kept so the user can correct it
  assert.match(t.$("#view").textContent, /Live performance so far/);
});

test("the about page explains the bands, the limits and lists every field", async () => {
  const t = boot();
  await t.go("#/about");
  const text = t.$("#view").textContent;
  assert.match(text, /HIGH risk: chance of churn of 26\.0 % or more/);
  assert.match(text, /MEDIUM risk: from 15\.0 %/);
  assert.match(text, /Known limits/);
  assert.match(text, /synthetic/);
  assert.equal(t.$$("tbody tr").length, FIELDS.length);
  assert.ok(t.$("caption").textContent.length > 0);
});

test("every table has a caption and column headers with scope", async () => {
  const t = boot();
  await t.go("#/about");
  assert.ok(t.$$("table").every((tb) => tb.querySelector("caption")));
  assert.ok(t.$$("th").every((th) => th.getAttribute("scope") === "col"));
});


// ------------------------------------------------------------------ pipeline console
test("the pipeline page draws the DVC graph with every stage and edge, and says each status in words", async () => {
  const t = boot();
  await t.go("#/pipeline");
  assert.equal(t.document.title, "Pipeline console | ChurnSense");
  assert.equal(t.$$("svg.dag g.node").length, STAGES.length);
  assert.equal(t.$$("svg.dag path.edge").length, EDGES.length);
  assert.deepEqual(t.$$("svg.dag g.node").map((n) => n.dataset.stage), STAGES);
  assert.match(t.$("svg.dag").getAttribute("aria-label"), /Pipeline graph with 9 stages: ingest ok/);
  assert.equal(t.$("svg.dag").getAttribute("role"), "img");
  assert.ok(t.$$("svg.dag g.node text").some((x) => x.textContent === "ok"));
  assert.equal(t.$$("tbody tr[data-stage]").length, STAGES.length);
});

test("the stage table shows time, rows and the speed of the Ray stages", async () => {
  const t = boot();
  await t.go("#/pipeline");
  const text = t.$("#view").textContent;
  assert.match(text, /600 rows\/s/);                 // ingest
  assert.match(text, /ingest \(Ray\)/);
  assert.match(text, /9 of 9stages with a report/);
  assert.match(text, /slowest stage/);
  assert.match(t.$("tr[data-stage=gate]").textContent, /REJECT/);
});

test("a failed or missing stage is shown as FAILED or not run, with its error", async () => {
  const stages = STAGES.map((stage) => ({ stage, status: "ok", duration_seconds: 1 }));
  stages[6] = { stage: "train", status: "failed", duration_seconds: 5, error: "uncommitted changes in params.yaml" };
  stages[7] = { stage: "evaluate", status: "not run" };
  const t = boot({ "GET /pipeline/stages": { body: { edges: EDGES, stages } } });
  await t.go("#/pipeline");
  assert.match(t.$("g.node[data-stage=train]").textContent, /FAILED/);
  assert.ok(t.$("g.node[data-stage=train]").getAttribute("class").includes("level-alert"));
  assert.ok(t.$("g.node[data-stage=evaluate]").getAttribute("class").includes("level-missing"));
  assert.match(t.$("tr[data-stage=train]").textContent, /uncommitted changes in params\.yaml/);
  assert.match(t.$("tr[data-stage=evaluate]").textContent, /not run/);
});

test("the run history shows outcomes, and a failed step with its reason, newest first", async () => {
  const t = boot();
  await t.go("#/pipeline");
  const rows = t.$$("tr[data-run]");
  assert.deepEqual(rows.map((r) => r.dataset.run), ["20261111T123731Z", "20261111T112718Z"]);
  assert.match(rows[0].textContent, /promoted/);
  assert.match(rows[1].textContent, /failed/);
  assert.match(rows[1].textContent, /FAILED: dvc_repro \(601 of 2201 feedback rows were quarantined\)/);
  assert.match(rows[0].textContent, /6 min 33 s/);
  assert.ok(rows[1].querySelector("li.step-fail"));
  assert.ok(rows[0].querySelector("details.run-steps"));
});

test("with no runs, and with waiting requests, the history says so", async () => {
  const t = boot({ "GET /pipeline/runs": { body: { runs: [], pending_requests: ["retrain", "rollback"] } } });
  await t.go("#/pipeline");
  assert.match(t.$("#view").textContent, /No retraining run has been recorded yet/);
  assert.match(t.$(".info-box").textContent, /Waiting for the controller: retrain, rollback/);
});

test("components are listed with running or NOT ANSWERING, and a quiet controller is flagged", async () => {
  const t = boot();
  await t.go("#/pipeline");
  const list = t.$(".component-list").textContent;
  assert.match(list, /running\s*model-server/);
  assert.match(list, /NOT ANSWERING\s*grafana/);
  assert.match(list, /running\s*retrain controller \(last sign of life 12 s ago\)/);
  const stalled = boot({ "GET /components": { body: { components: { nginx: { up: true } }, controller_heartbeat_age_seconds: 900 } } });
  await stalled.go("#/pipeline");
  assert.match(stalled.$(".component-list").textContent, /NOT REPORTING\s*retrain controller/);
  const never = boot({ "GET /components": { body: { components: {}, controller_heartbeat_age_seconds: null } } });
  await never.go("#/pipeline");
  assert.match(never.$(".component-list").textContent, /NOT REPORTING/);
});

test("Retrain now needs the admin token, asks for confirmation and sends the token as a header", async () => {
  const t = boot();
  await t.go("#/pipeline");
  t.click(t.$("#retrain-button"));
  await t.settle();
  assert.match(t.$("#control-status").textContent, /Enter the admin token first/);
  assert.equal(t.calls.filter((c) => c.key === "POST /admin/pipeline/request").length, 0);
  t.type(t.$("#admin-token"), "s3cret");
  t.window.confirm = () => false;
  t.click(t.$("#retrain-button"));
  await t.settle();
  assert.equal(t.calls.filter((c) => c.key === "POST /admin/pipeline/request").length, 0);   // declined
  t.window.confirm = () => true;
  t.click(t.$("#retrain-button"));
  await t.settle();
  const sent = t.calls.find((c) => c.key === "POST /admin/pipeline/request");
  assert.deepEqual(sent.body, { action: "retrain" });
  assert.equal(sent.headers["x-admin-token"], "s3cret");
  assert.match(t.$("#control-status").textContent, /Request sent/);
  t.click(t.$("#rollback-button"));
  await t.settle();
  assert.deepEqual(t.calls.filter((c) => c.key === "POST /admin/pipeline/request").map((c) => c.body.action), ["retrain", "rollback"]);
});

test("a wrong token, a waiting request and a disabled console each get a plain message", async () => {
  for (const [status, expected] of [[401, /token is not correct/], [409, /already waiting/], [503, /switched off/]]) {
    const t = boot({ "POST /admin/pipeline/request": { status, body: { error: { code: "x", message: "m", details: [] } } } });
    await t.go("#/pipeline");
    t.type(t.$("#admin-token"), "x");
    t.click(t.$("#retrain-button"));
    await t.settle();
    assert.match(t.$("#control-status").textContent, expected, String(status));
  }
});

test("the admin token field is a password field that never autofills and the token is not kept in the page", async () => {
  const t = boot();
  await t.go("#/pipeline");
  assert.equal(t.$("#admin-token").type, "password");
  assert.equal(t.$("#admin-token").getAttribute("autocomplete"), "off");
  assert.ok(!JSON.stringify(t.window.ChurnApp.state).includes("s3cret"));
});

test("the pipeline page refreshes itself and each panel fails alone", async () => {
  const t = boot({ "GET /pipeline/runs": "network-error", "GET /components": { status: 503, body: { error: { code: "x", message: "m", details: [] } } } });
  await t.go("#/pipeline");
  assert.ok(t.window.ChurnApp.state.timer);
  assert.match(t.$("#view").textContent, /cannot be reached/);                // run history
  assert.match(t.$("#view").textContent, /component check is not available/);
  assert.equal(t.$$("svg.dag g.node").length, STAGES.length);                  // the graph still works
  await t.go("#/about");
  assert.equal(t.window.ChurnApp.state.timer, null);
});

// ------------------------------------------------------------------ loose coupling: a configurable API address
test("every call goes to the configured API address", async () => {
  const direct = boot({}, "#/", { config: { apiBase: "https://api.example.org/churn/v2" } });
  const seen = [];
  const inner = direct.window.fetch;
  direct.window.fetch = async (url, options) => { seen.push(String(url)); return inner(url, options); };
  await direct.go("#/about");
  assert.ok(seen.length > 0 && seen.every((u) => u.startsWith("https://api.example.org/churn/v2/")), seen.join());
});

test("without a configuration the page falls back to the same-host API", async () => {
  const t = boot({}, "#/", { config: {} });
  await t.settle();
  assert.ok(t.calls.every((c) => c.url.startsWith("/api/v1/")));
});
