"use strict";
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const FRONTEND = path.resolve(__dirname, "../../frontend");
const FIELDS = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/fields.json"), "utf8"));

const MODEL = { version: "v1", run_id: "abcdef1234567890", family: "logreg", trained_at: "2026-11-11T08:00:00+00:00",
  thresholds: { t_high: 0.26, t_medium: 0.15 } };
const FACTORS = [
  { feature: "is_month_to_month", value: 1, impact: 0.51, direction: "increases_risk" },
  { feature: "tenure_months", value: 5, impact: 0.44, direction: "increases_risk" },
  { feature: "is_fiber", value: 0, impact: -0.2, direction: "decreases_risk" },
];

const STAGES = ["ingest", "validate", "eda", "split", "synthesize", "featurize", "train", "evaluate", "gate"];
const EDGES = [["ingest", "validate"], ["validate", "eda"], ["validate", "split"], ["split", "synthesize"], ["split", "featurize"],
  ["synthesize", "featurize"], ["featurize", "train"], ["split", "train"], ["train", "evaluate"], ["split", "evaluate"], ["evaluate", "gate"], ["train", "gate"]];

/* The default answers of the API; a test overrides single routes. */
function defaults() {
  return {
    "GET /fields": { body: FIELDS },
    "GET /health": { body: { status: "ok", model: MODEL } },
    "GET /model": { body: MODEL },
    "GET /performance": { body: { n: 120, min_samples: 30, enough_data: true, accuracy: 0.77, precision: 0.56, recall: 0.88, tp: 40, fp: 31, fn: 5, tn: 44 } },
    "GET /stats": { body: { window_hours: 24, predictions: 200, bands: { HIGH: 80, MEDIUM: 20, LOW: 100 }, mean_probability: 0.31, last_prediction_at: new Date().toISOString() } },
    "GET /pipeline": { body: { gate: { decision: "REJECT", reasons: ["recent_data_sufficient: 0 recent rows"] }, golden: { n: 1057, roc_auc: 0.8619, recall_at_min_precision: 0.897 },
      last_promotion: "promoted", controller: { watching: { decision: "no retraining", reason: "healthy" }, runs: { promoted: 1 }, last_run: { outcome: "promoted", trigger: "drift", finished_at: new Date().toISOString() } } } },
    "GET /drift": { body: { status: "alert", n_rows: 500, enough_data: true, max_psi: 1.18, worst_feature: "charge_delta", streak: 4, needed_streak: 3,
      psi_by_feature: { charge_delta: 1.18, contract: 0.4, tenure_months: 0.05 }, history: [{ max_psi: 0.04 }, { max_psi: 0.9 }, { max_psi: 1.18 }] } },
    "POST /predict": (body) => ({ body: { customer_id: body.customer_id || null, churn_probability: 0.702, risk_band: "HIGH", top_factors: FACTORS, model: MODEL, request_id: "req-123" } }),
    "POST /predict/batch": (body) => ({ body: { results: body.customers.map((c, i) => ({ index: i, customer_id: c.customer_id || null, churn_probability: 0.9 - i * 0.01, risk_band: i % 3 === 0 ? "HIGH" : i % 3 === 1 ? "MEDIUM" : "LOW", top_factors: FACTORS })),
      model: MODEL, summary: { total: body.customers.length } } }),
    "GET /pipeline/stages": { body: { edges: EDGES, stages: STAGES.map((stage, i) => ({ stage, status: "ok", duration_seconds: 2 + i * 10,
      rows_in: 7043, rows_out: 7043, rows_per_second: 600 * (i + 1), finished_at: new Date().toISOString(), detail: stage === "gate" ? "REJECT" : null, error: null })) } },
    "GET /pipeline/runs": { body: { pending_requests: [], runs: [
      { run_id: "20261111T123731Z", trigger: "manual", reason: "requested by the operator", outcome: "promoted", duration_seconds: 392.8, finished_at: new Date().toISOString(),
        steps: [{ step: "batch", ok: true, detail: "1000 rows" }, { step: "dvc_repro", ok: true, detail: "" }, { step: "promote", ok: true, detail: "" }] },
      { run_id: "20261111T112718Z", trigger: "drift", reason: "drift alert on contract", outcome: "failed", duration_seconds: 411.8, finished_at: new Date().toISOString(),
        steps: [{ step: "batch", ok: true, detail: "" }, { step: "dvc_repro", ok: false, detail: "601 of 2201 feedback rows were quarantined" }] }] } },
    "GET /components": { body: { components: { "model-server": { up: true }, mlflow: { up: true }, nginx: { up: true }, grafana: { up: false } }, controller_heartbeat_age_seconds: 12.4 } },
    "POST /admin/pipeline/request": (body) => ({ status: 202, body: { status: "accepted", action: body.action, request_id: "r1", message: "ok" } }),
    "POST /feedback": (body) => ({ body: { status: "recorded", counted_as: "true_positive", matched: { request_id: "r", risk_band: "HIGH" } } }),
  };
}

const OPEN = [];
/* Close every simulated window (their refresh timers would keep Node running). */
function closeAll() { while (OPEN.length) OPEN.pop().close(); }

/* Start the page in a simulated browser with a fake API. */
function boot(overrides = {}, hash = "#/", options = {}) {
  const html = fs.readFileSync(path.join(FRONTEND, "index.html"), "utf8").replace(/<script[^>]*><\/script>/g, "");
  const dom = new JSDOM(html, { url: "https://localhost:8443/" + hash, runScripts: "outside-only", pretendToBeVisual: true });
  const { window } = dom;
  OPEN.push(window);
  const routes = Object.assign(defaults(), overrides);
  const calls = [];
  window.fetch = async (url, options = {}) => {
    const key = (options.method || "GET") + " " + String(url).replace("/api/v1", "");
    const body = options.body ? JSON.parse(options.body) : null;
    calls.push({ key, body, url: String(url), headers: options.headers || {} });
    const handler = routes[key];
    if (handler === "network-error") throw new Error("offline");
    const answer = handler === undefined ? { status: 404, body: { error: { code: "http_404", message: "not stubbed " + key, details: [] } } }
      : typeof handler === "function" ? handler(body) : handler;
    const status = answer.status || 200;
    return { ok: status < 400, status, json: async () => answer.body };
  };
  const downloads = [];
  window.Blob = class { constructor(parts, options) { this.text = parts.join(""); this.type = options && options.type; } };
  window.URL.createObjectURL = (blob) => { downloads.push(blob); return "blob:test"; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () { downloads.push({ clicked: this.getAttribute("download") }); };
  window.confirm = () => true;
  if (options.config !== undefined) window.eval("window.CHURNSENSE_CONFIG = " + JSON.stringify(options.config));
  else window.eval(fs.readFileSync(path.join(FRONTEND, "config.js"), "utf8"));
  window.eval(fs.readFileSync(path.join(FRONTEND, "lib.js"), "utf8"));
  window.eval(fs.readFileSync(path.join(FRONTEND, "app.js"), "utf8"));
  const settle = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));
  const $ = (selector) => window.document.querySelector(selector);
  const $$ = (selector) => [...window.document.querySelectorAll(selector)];
  const click = (node) => node.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  const type = (node, value) => { node.value = value; node.dispatchEvent(new window.Event("input", { bubbles: true })); };
  const submit = (form) => form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  const go = async (route) => { window.location.hash = route; await settle(); };
  return { window, document: window.document, calls, downloads, settle, $, $$, click, type, submit, go };
}

module.exports = { closeAll, boot, defaults, FIELDS, MODEL, FACTORS, FRONTEND, STAGES, EDGES };
