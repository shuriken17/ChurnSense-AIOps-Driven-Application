/* ChurnSense web interface. All text is inserted with textContent (never innerHTML), so no value
   from the server or from an uploaded file can inject markup. Pure helpers live in lib.js. */
"use strict";
(function () {
  const L = window.ChurnLib;
  const API = (window.CHURNSENSE_CONFIG && window.CHURNSENSE_CONFIG.apiBase) || "/api/v1";
  const view = document.getElementById("view");
  const modelLine = document.getElementById("model-line");
  const state = { fields: null, model: null, timer: null, batch: null };
  const FAMILY = { logreg: "logistic regression", lightgbm: "gradient boosted trees (LightGBM)" };
  const PAGE_SIZE = 25;
  const CHUNK = 500;
  const MAX_FILE_ROWS = 5000;
  const EXAMPLE = {
    gender: "Female", senior_citizen: 0, partner: "Yes", dependents: "No", tenure_months: 5,
    phone_service: "Yes", multiple_lines: "No", internet_service: "Fiber optic", online_security: "No",
    online_backup: "No", device_protection: "No", tech_support: "No", streaming_tv: "No",
    streaming_movies: "No", contract: "Month-to-month", paperless_billing: "Yes",
    payment_method: "Electronic check", monthly_charges: 79.85, total_charges: 399.25,
  };
  const GROUPS = [
    ["Customer", ["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure_months"]],
    ["Phone and internet", ["phone_service", "multiple_lines", "internet_service", "online_security",
      "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]],
    ["Contract and billing", ["contract", "paperless_billing", "payment_method", "monthly_charges", "total_charges"]],
  ];

  // ------------------------------------------------------------------ small helpers
  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  async function api(path, options) {
    try {
      const response = await fetch(API + path, Object.assign({ headers: { "Content-Type": "application/json" } }, options));
      let body = null;
      try { body = await response.json(); } catch (problem) { body = null; }
      return { ok: response.ok, status: response.status, body };
    } catch (problem) {
      return { ok: false, status: 0, body: null };
    }
  }
  const post = (path, payload, headers) => api(path, {
    method: "POST", body: JSON.stringify(payload), headers: Object.assign({ "Content-Type": "application/json" }, headers),
  });

  async function loadFields() {
    if (!state.fields) {
      const answer = await api("/fields");
      if (answer.ok) state.fields = answer.body;
    }
    return state.fields;
  }
  async function loadModel() {
    const answer = await api("/model");
    if (answer.ok) state.model = answer.body;
    return state.model;
  }

  function errorText(answer, what) {
    if (answer.status === 0) return "The service cannot be reached. Try again in a moment.";
    if (answer.status === 503) return "The " + what + " is not available right now.";
    return (answer.body && answer.body.error && answer.body.error.message) || "Something went wrong (HTTP " + answer.status + ").";
  }

  function badge(label, cssClass) { return h("span", { class: "badge " + cssClass, text: label }); }
  function kpi(value, label) { return h("div", { class: "kpi" }, h("div", { class: "value", text: value }), h("div", { class: "label", text: label })); }
  function bars(rows, maxValue) {
    return rows.map((row) => {
      const fill = h("span");
      fill.style.width = Math.max(0, Math.min(100, (row.value / maxValue) * 100)) + "%";
      return h("div", { class: "bar-row " + (row.css || "") },
        h("span", { text: row.label }), h("div", { class: "bar", role: "img", "aria-label": row.label + " " + row.text }, fill), h("span", { text: row.text }));
    });
  }

  function card(title) {
    const body = h("div", { class: "card-body" }, h("p", { class: "muted", text: "Loading..." }));
    const el = h("section", { class: "card", "aria-label": title }, h("h2", { text: title }), body);
    return {
      el,
      set(...content) { body.replaceChildren(...content.flat().filter(Boolean)); },
      fail(message) { body.replaceChildren(h("p", { class: "muted", text: message })); },
    };
  }

  // ------------------------------------------------------------------ the model line under the header
  async function refreshModelLine() {
    const answer = await api("/health");
    const model = answer.body && answer.body.model;
    modelLine.textContent = answer.ok && model
      ? "Model " + model.version + " is serving (" + (FAMILY[model.family] || model.family) + "), trained " + L.relativeTime(model.trained_at)
      : (answer.status === 0 ? "The service cannot be reached" : "No model is available right now");
  }

  // ------------------------------------------------------------------ dashboard
  async function fillModel(c) {
    const [model, pipe] = await Promise.all([api("/model"), api("/pipeline")]);
    if (!model.ok) return c.fail(errorText(model, "model"));
    const m = model.body;
    const golden = (pipe.ok && pipe.body.golden) || {};
    c.set(
      h("p", {}, h("strong", { text: "Version " + m.version }), " (" + (FAMILY[m.family] || m.family) + ")"),
      h("div", { class: "kpis" },
        kpi(golden.roc_auc != null ? golden.roc_auc.toFixed(3) : "n/a", "ROC-AUC on the fixed test set"),
        kpi(golden.recall_at_min_precision != null ? L.percent(golden.recall_at_min_precision, 0) : "n/a", "recall at 50 % precision")),
      h("p", { class: "small muted", text: "Risk bands: HIGH from " + m.thresholds.t_high.toFixed(3) + ", MEDIUM from " + m.thresholds.t_medium.toFixed(3) + ". Trained " + L.relativeTime(m.trained_at) + "." }));
  }

  async function fillPerformance(c) {
    const answer = await api("/performance");
    if (!answer.ok) return c.fail(errorText(answer, "performance service"));
    const p = answer.body;
    if (!p.enough_data) {
      return c.set(h("p", { text: "Needs " + p.min_samples + " reported outcomes before the numbers mean anything. Now: " + p.n + "." }),
        h("p", { class: "small muted", text: "Report what happened to scored customers on the Report an outcome page." }));
    }
    c.set(
      h("div", { class: "kpis" }, kpi(L.percent(p.accuracy), "accuracy"), kpi(L.percent(p.precision), "precision"), kpi(L.percent(p.recall), "recall")),
      h("p", { class: "small muted", text: "Measured on " + p.n + " reported outcomes. A customer in the HIGH band counts as predicted churn. Right: " + (p.tp + p.tn) + ", missed churners: " + p.fn + ", false alarms: " + p.fp + "." }));
  }

  async function fillTraffic(c) {
    const answer = await api("/stats");
    if (!answer.ok) return c.fail(errorText(answer, "statistics service"));
    const s = answer.body;
    const total = Math.max(s.predictions, 1);
    c.set(
      h("div", { class: "kpis" }, kpi(String(s.predictions), "predictions"), kpi(L.percent(s.mean_probability), "average churn chance")),
      bars(["HIGH", "MEDIUM", "LOW"].map((band) => ({
        label: L.bandInfo(band).label, value: s.bands[band], css: L.bandInfo(band).css,
        text: s.bands[band] + " (" + L.percent(s.bands[band] / total, 0) + ")" })), total),
      h("p", { class: "small muted", text: s.last_prediction_at ? "Last prediction " + L.relativeTime(s.last_prediction_at) + "." : "No predictions yet." }));
  }

  function sparkline(points) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "spark");
    svg.setAttribute("viewBox", "0 0 200 40");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Largest PSI of the last " + points.length + " checks");
    const top = Math.max(0.5, ...points);
    const y = (v) => 38 - (Math.min(v, top) / top) * 36;
    const alert = document.createElementNS(ns, "line");
    [["x1", 0], ["x2", 200], ["y1", y(0.25)], ["y2", y(0.25)]].forEach(([k, v]) => alert.setAttribute(k, v));
    svg.append(alert);
    if (points.length > 1) {
      const line = document.createElementNS(ns, "polyline");
      line.setAttribute("points", points.map((v, i) => (i / (points.length - 1)) * 200 + "," + y(v)).join(" "));
      svg.append(line);
    }
    return svg;
  }

  async function fillDrift(c) {
    const answer = await api("/drift");
    if (!answer.ok) return c.fail(errorText(answer, "drift monitor"));
    const d = answer.body;
    const level = d.status === "alert" ? "alert" : (d.status === "warn" || d.status === "alert_candidate") ? "warn" : "ok";
    const top = Object.entries(d.psi_by_feature || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
    c.set(
      h("p", {}, badge(L.driftText(d.status), "level-" + level)),
      d.enough_data === false
        ? h("p", { class: "muted", text: "Only " + d.n_rows + " predictions so far; a verdict needs more." })
        : h("div", { class: "kpis" },
          kpi(d.max_psi.toFixed(2), "largest PSI (" + (d.worst_feature || "none") + ")"),
          kpi(d.streak + " of " + d.needed_streak, "checks in a row at alert level"),
          kpi(String(d.n_rows), "predictions compared")),
      bars(top.map(([name, value]) => ({ label: name, value, css: "level-" + L.psiLevel(value).level, text: value.toFixed(2) + " " + L.psiLevel(value).label })), 0.5),
      (d.history || []).length > 1 ? sparkline(d.history.map((x) => x.max_psi)) : null,
      h("p", { class: "small muted", text: "PSI compares live inputs with the training data: under 0.10 stable, 0.10 to 0.25 moderate, above 0.25 large. A retraining starts only after several large checks in a row." }));
  }

  async function fillPipeline(c) {
    const answer = await api("/pipeline");
    if (!answer.ok) return c.fail(errorText(answer, "pipeline status"));
    const p = answer.body;
    const controller = p.controller || {};
    const last = controller.last_run;
    const decision = p.gate && p.gate.decision;
    c.set(
      h("p", {}, "Latest gate decision: ", decision ? badge(decision, decision === "PROMOTE" ? "level-ok" : "level-warn") : h("span", { class: "muted", text: "none yet" })),
      ((p.gate && p.gate.reasons) || []).length ? h("ul", { class: "small" }, p.gate.reasons.slice(0, 3).map((r) => h("li", { text: r }))) : null,
      h("p", { class: "small", text: last ? "Last retraining: " + last.outcome + " (" + last.trigger + "), " + L.relativeTime(last.finished_at) + "." : "No automatic retraining has run yet." }),
      controller.watching && controller.watching.decision
        ? h("p", { class: "small muted", text: "Watcher: " + controller.watching.decision + (controller.watching.reason ? " (" + controller.watching.reason + ")" : "") }) : null);
  }

  function fillTools(c) {
    const base = location.protocol + "//" + location.hostname;
    const link = (port, name, note) => h("li", {}, h("a", { href: base + ":" + port + "/", target: "_blank", rel: "noopener" }, name), " " + note);
    c.set(h("ul", {}, link(3000, "Grafana", "dashboards and alerts"), link(9090, "Prometheus", "metrics and alert rules"), link(5000, "MLflow", "experiments and the model registry")));
  }

  function dashboardPage() {
    const cards = { model: card("Model in use"), perf: card("Live performance"), traffic: card("Predictions, last 24 hours"),
      drift: card("Data drift"), pipeline: card("Pipeline and retraining"), tools: card("Tools") };
    const refresh = () => Promise.all([fillModel(cards.model), fillPerformance(cards.perf), fillTraffic(cards.traffic),
      fillDrift(cards.drift), fillPipeline(cards.pipeline)]);
    fillTools(cards.tools);
    refresh();
    state.timer = setInterval(refresh, 15000);
    return h("div", {}, h("h1", { text: "Dashboard" }),
      h("p", { class: "muted", text: "The model, the data it sees and the retraining pipeline. Refreshes every 15 seconds." }),
      h("div", { class: "grid" }, Object.values(cards).map((c) => c.el)));
  }

  // ------------------------------------------------------------------ score one customer
  function fieldInput(field) {
    let input;
    if (field.allowed) {
      input = h("select", { id: "f-" + field.name, name: field.name, "aria-describedby": "e-" + field.name }, h("option", { value: "", text: "Choose..." }),
        field.allowed.map((v) => h("option", { value: v, text: v })));
    } else {
      input = h("input", { id: "f-" + field.name, name: field.name, "aria-describedby": "e-" + field.name,
        type: field.kind === "id" ? "text" : "number", step: field.kind === "float" ? "0.01" : field.kind === "id" ? null : "1",
        min: field.min, max: field.max });
    }
    const wrapper = h("div", {},
      h("label", { for: "f-" + field.name }, field.label + (field.required ? "" : " (optional)"), h("span", { class: "hint", text: field.description || "" })),
      input, h("span", { class: "field-error", id: "e-" + field.name }));
    return wrapper;
  }

  function readForm(form, fields) {
    const customer = {};
    for (const field of fields) {
      const element = form.elements[field.name];
      if (!element || element.value === "") continue;
      customer[field.name] = element.type === "number" ? Number(element.value) : element.value;
    }
    return customer;
  }

  function showFieldErrors(form, fields, details, summary) {
    for (const field of fields) {
      const element = form.elements[field.name];
      if (element) element.removeAttribute("aria-invalid");
      const slot = document.getElementById("e-" + field.name);
      if (slot) slot.textContent = "";
    }
    for (const detail of details) {
      const element = form.elements[detail.field];
      if (element) element.setAttribute("aria-invalid", "true");
      const slot = document.getElementById("e-" + detail.field);
      if (slot) slot.textContent = detail.message;
    }
    summary.replaceChildren(...(details.length ? [h("strong", { text: "Please fix these problems:" }), h("ul", {}, details.map((d) => h("li", { text: d.field + " " + d.message })))] : []));
    summary.hidden = details.length === 0;
  }

  function meter(probability, model, css) {
    const fill = h("div", { class: "fill" });
    fill.style.width = Number(Math.min(100, probability * 100).toFixed(1)) + "%";
    const node = h("div", { class: "meter " + css, role: "img", "aria-label": "Churn chance " + L.percent(probability) }, fill);
    if (model) {
      [["MEDIUM", model.thresholds.t_medium], ["HIGH", model.thresholds.t_high]].forEach(([name, value]) => {
        const mark = h("div", { class: "mark" });
        mark.style.left = Number((value * 100).toFixed(1)) + "%";
        const label = h("span", { class: "mark-label", text: name + " from " + L.percent(value, 0) });
        label.style.left = Number((value * 100).toFixed(1)) + "%";
        node.append(mark, label);
      });
    }
    return node;
  }

  function feedbackBlock(result) {
    const status = h("p", { class: "small", "aria-live": "polite" });
    const send = async (churned, button, other) => {
      button.disabled = other.disabled = true;
      const answer = await post("/feedback", { request_id: result.request_id, churned });
      button.disabled = other.disabled = false;
      button.setAttribute("aria-pressed", String(answer.ok));
      status.textContent = answer.ok
        ? "Recorded: " + (churned ? "the customer churned" : "the customer stayed") + ". It counts toward the live performance numbers."
        : errorText(answer, "feedback service");
    };
    const yes = h("button", { type: "button", class: "secondary", "aria-pressed": "false", text: "The customer churned" });
    const no = h("button", { type: "button", class: "secondary", "aria-pressed": "false", text: "The customer stayed" });
    yes.addEventListener("click", () => send(true, yes, no));
    no.addEventListener("click", () => send(false, no, yes));
    return h("div", {}, h("h3", { text: "What actually happened?" }),
      h("p", { class: "small muted", text: "Reporting the real outcome lets the system measure itself and, with enough outcomes, retrain." }),
      h("div", { class: "row" }, yes, no), status);
  }

  function resultCard(result, model) {
    const info = L.bandInfo(result.risk_band);
    return h("section", { class: "card", id: "result", "aria-label": "Result" },
      h("h2", { text: "Result" }),
      h("p", {}, h("span", { class: "badge result-band " + info.css, text: info.symbol + " " + info.label })),
      h("p", { text: info.advice }),
      h("p", {}, h("strong", { text: "Chance of churn: " + L.percent(result.churn_probability) })),
      meter(result.churn_probability, model, info.css),
      h("h3", { text: "Main reasons" }),
      h("ol", { class: "reasons" }, result.top_factors.map((factor) => {
        const e = L.explainFactor(factor);
        return h("li", {}, e.text + " ", h("span", { class: e.raises ? "raises" : "lowers", text: "(" + e.arrow + ")" }));
      })),
      h("p", { class: "small muted", text: "Model " + result.model.version + " (run " + result.model.run_id.slice(0, 8) + "), request " + result.request_id }),
      feedbackBlock(result));
  }

  async function scorePage() {
    const root = h("div", {}, h("h1", { text: "Score a customer" }),
      h("p", { class: "muted", text: "Enter what you know about one customer. The model returns the chance of churn, a risk band and the main reasons." }));
    const fields = await loadFields();
    if (!fields) return root.append(h("p", { class: "alert-box", role: "alert", text: "The field list could not be loaded. Try again in a moment." })) || root;
    const model = state.model || (await loadModel());
    const form = h("form", { id: "customer-form", novalidate: true });
    const byName = new Map(fields.map((f) => [f.name, f]));
    GROUPS.forEach(([title, names]) => form.append(h("fieldset", {}, h("legend", { text: title }),
      h("div", { class: "fields" }, names.map((n) => byName.get(n)).filter(Boolean).map(fieldInput)))));
    const summary = h("div", { class: "alert-box", role: "alert", hidden: true });
    const output = h("div", { "aria-live": "polite" });
    const submit = h("button", { type: "submit", id: "score-button", text: "Score customer" });
    const example = h("button", { type: "button", class: "secondary", id: "example-button", text: "Fill an example" });
    const clearButton = h("button", { type: "button", class: "secondary", text: "Clear" });
    form.append(summary, h("div", { class: "row" }, submit, example, clearButton));
    example.addEventListener("click", () => fields.forEach((f) => { if (f.name in EXAMPLE) form.elements[f.name].value = EXAMPLE[f.name]; }));
    clearButton.addEventListener("click", () => { form.reset(); showFieldErrors(form, fields, [], summary); output.replaceChildren(); });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const customer = readForm(form, fields);
      const missing = fields.filter((f) => f.required && !(f.name in customer)).map((f) => ({ field: f.name, message: "is required" }));
      showFieldErrors(form, fields, missing, summary);
      if (missing.length) { output.replaceChildren(); return; }
      submit.disabled = true;
      const answer = await post("/predict", customer);
      submit.disabled = false;
      if (answer.ok) { output.replaceChildren(resultCard(answer.body, model)); return; }
      output.replaceChildren();
      const details = (answer.body && answer.body.error && answer.body.error.details) || [];
      showFieldErrors(form, fields, details, summary);
      if (!details.length) { summary.replaceChildren(h("strong", { text: errorText(answer, "prediction service") })); summary.hidden = false; }
    });
    root.append(form, output);
    return root;
  }

  // ------------------------------------------------------------------ score a file
  function resultRows(results) {
    return results.map((r) => {
      const reasons = (r.factors || []).map((f) => L.explainFactor(f).text);
      return [r.row, r.customer_id || "", r.probability === null ? "" : r.probability.toFixed(4), r.band || "", reasons[0] || "", reasons[1] || "", reasons[2] || "", r.error || ""];
    });
  }

  function resultsTable(results, view) {
    let rows = results.filter((r) => view.filter === "ALL" || (view.filter === "ERROR" ? r.error : r.band === view.filter));
    rows = rows.slice().sort((a, b) => view.sort === "row" ? a.row - b.row : (b.probability ?? -1) - (a.probability ?? -1));
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    view.page = Math.min(view.page, pages - 1);
    const shown = rows.slice(view.page * PAGE_SIZE, (view.page + 1) * PAGE_SIZE);
    const table = h("table", {}, h("caption", { text: "Scored customers (" + rows.length + ")" }),
      h("thead", {}, h("tr", {}, ["File row", "Customer", "Chance of churn", "Risk band", "Main reasons", "Problem"].map((t) => h("th", { scope: "col", text: t })))),
      h("tbody", {}, shown.map((r) => {
        const info = r.band ? L.bandInfo(r.band) : null;
        return h("tr", {}, h("td", { text: r.row }), h("td", { text: r.customer_id || "" }),
          h("td", { text: r.probability === null ? "" : L.percent(r.probability) }),
          h("td", {}, info ? h("span", { class: "badge " + info.css, text: info.symbol + " " + info.label }) : ""),
          h("td", { text: (r.factors || []).map((f) => L.explainFactor(f).text).join("; ") }), h("td", { text: r.error || "" }));
      })));
    return { table, pages, count: rows.length };
  }

  async function sendOutcomes(results, outcomes, statusNode) {
    const todo = results.filter((r) => r.customer_id && outcomes.has(r.row) && outcomes.get(r.row) !== null);
    const tally = { recorded: 0, unknown: 0, failed: 0 };
    let next = 0;
    const worker = async () => {
      while (next < todo.length) {
        const r = todo[next++];
        const answer = await post("/feedback", { customer_id: r.customer_id, churned: outcomes.get(r.row), source: "user" });
        if (answer.ok) tally.recorded++; else if (answer.status === 404) tally.unknown++; else tally.failed++;
        statusNode.textContent = "Sent " + (tally.recorded + tally.unknown + tally.failed) + " of " + todo.length + " outcomes...";
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    statusNode.textContent = "Outcomes sent: " + tally.recorded + " recorded, " + tally.unknown + " customers not found among earlier predictions, " + tally.failed + " failed.";
  }

  function renderResults(host, batch) {
    const results = batch.results;
    const view = { filter: "ALL", sort: "probability", page: 0 };
    const bands = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    results.forEach((r) => { if (r.band) bands[r.band]++; });
    const errors = results.filter((r) => r.error).length;
    const tableHost = h("div", { class: "table-wrap" });
    const pager = h("div", { class: "pager" });
    const draw = () => {
      const { table, pages, count } = resultsTable(results, view);
      tableHost.replaceChildren(table);
      const prev = h("button", { type: "button", class: "secondary", text: "Previous", disabled: view.page === 0 });
      const next = h("button", { type: "button", class: "secondary", text: "Next", disabled: view.page >= pages - 1 });
      prev.addEventListener("click", () => { view.page--; draw(); });
      next.addEventListener("click", () => { view.page++; draw(); });
      pager.replaceChildren(prev, h("span", { text: "Page " + (view.page + 1) + " of " + pages + " (" + count + " rows)" }), next);
    };
    const filters = h("div", { class: "row", role: "group", "aria-label": "Filter by risk band" },
      [["ALL", "All"], ["HIGH", "High"], ["MEDIUM", "Medium"], ["LOW", "Low"], ["ERROR", "Problems"]].map(([key, label]) => {
        const button = h("button", { type: "button", class: "secondary", "aria-pressed": String(key === "ALL"), text: label });
        button.addEventListener("click", () => {
          view.filter = key; view.page = 0;
          filters.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
          draw();
        });
        return button;
      }));
    const sortButton = h("button", { type: "button", class: "secondary", text: "Sort by file row" });
    sortButton.addEventListener("click", () => {
      view.sort = view.sort === "probability" ? "row" : "probability";
      sortButton.textContent = view.sort === "probability" ? "Sort by file row" : "Sort by chance of churn";
      draw();
    });
    const download = h("button", { type: "button", text: "Download results (CSV)" });
    download.addEventListener("click", () => {
      const csv = L.toCsv(["file_row", "customer_id", "churn_probability", "risk_band", "reason_1", "reason_2", "reason_3", "problem"], resultRows(results));
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      const link = h("a", { href: url, download: "churnsense-results.csv" });
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });
    const outcomeStatus = h("p", { class: "small", "aria-live": "polite" });
    const outcomeCount = batch.outcomes ? results.filter((r) => r.customer_id && batch.outcomes.get(r.row) !== null && batch.outcomes.has(r.row)).length : 0;
    const outcomeButton = outcomeCount
      ? h("div", { class: "info-box" }, h("p", { text: "The file has a Churn column and customer ids: " + outcomeCount + " known outcomes can be sent as feedback. This is real labelled data and counts toward live performance." }),
        (() => { const b = h("button", { type: "button", text: "Send " + outcomeCount + " known outcomes as feedback" });
          b.addEventListener("click", async () => { b.disabled = true; await sendOutcomes(results, batch.outcomes, outcomeStatus); }); return b; })(), outcomeStatus)
      : null;
    host.replaceChildren(
      h("h2", { text: "Results" }),
      h("div", { class: "kpis" }, kpi(String(results.length - errors), "scored"), kpi(String(errors), "rejected"),
        kpi(String(bands.HIGH), "high risk"), kpi(String(bands.MEDIUM), "medium risk"), kpi(String(bands.LOW), "low risk")),
      outcomeButton, filters, h("div", { class: "row" }, sortButton, download), tableHost, pager);
    draw();
  }

  async function scoreFile(host, batch, progress, button) {
    button.disabled = true;
    batch.results = [];
    const sendable = batch.customers.filter((c) => c.problems.length === 0);
    batch.customers.filter((c) => c.problems.length).forEach((c) => batch.results.push({
      row: c.row, customer_id: c.customer.customer_id || "", probability: null, band: null, factors: [],
      error: c.problems.map((p) => p.field + " " + p.message).join("; ") }));
    const groups = L.chunk(sendable, CHUNK);
    for (let i = 0; i < groups.length; i++) {
      progress.textContent = "Scoring... " + (i * CHUNK) + " of " + sendable.length;
      const answer = await post("/predict/batch", { customers: groups[i].map((c) => c.customer) });
      if (!answer.ok) {
        progress.textContent = errorText(answer, "prediction service");
        progress.className = "alert-box";
        button.disabled = false;
        return;
      }
      answer.body.results.forEach((r, index) => {
        const source = groups[i][index];
        if (r.error) {
          batch.results.push({ row: source.row, customer_id: source.customer.customer_id || "", probability: null, band: null, factors: [],
            error: r.error.details.map((d) => d.field + " " + d.message).join("; ") });
        } else {
          batch.results.push({ row: source.row, customer_id: r.customer_id || "", probability: r.churn_probability, band: r.risk_band, factors: r.top_factors, error: "" });
        }
      });
    }
    progress.className = "muted small";
    progress.textContent = "Done: " + sendable.length + " customers sent.";
    button.disabled = false;
    renderResults(host, batch);
  }

  function checkText(text, fields, summaryHost, resultHost) {
    state.batch = null;
    resultHost.replaceChildren();
    const rows = L.parseCsv(text);
    const problem = (message) => { summaryHost.replaceChildren(h("div", { class: "alert-box", role: "alert", text: message })); };
    if (rows.length < 2) return problem("The data needs a header line and at least one customer row.");
    if (rows.length - 1 > MAX_FILE_ROWS) return problem("The file has " + (rows.length - 1) + " rows; the limit is " + MAX_FILE_ROWS + ". Please split it.");
    const mapping = L.mapHeaders(rows[0], fields);
    if (mapping.missing.length) return problem("These required columns are missing: " + mapping.missing.join(", ") + ".");
    const customers = rows.slice(1).map((cells, i) => Object.assign({ row: i + 2 }, L.toCustomer(cells, mapping)));
    const outcomes = new Map();
    if (mapping.outcomeColumn >= 0) rows.slice(1).forEach((cells, i) => outcomes.set(i + 2, L.parseOutcome(cells[mapping.outcomeColumn])));
    const localProblems = customers.filter((c) => c.problems.length).length;
    const recognised = mapping.columns.filter((c) => c.field).length;
    const progress = h("p", { class: "muted small", "aria-live": "polite" });
    const button = h("button", { type: "button", id: "run-batch", text: "Score " + (customers.length - localProblems) + " customers" });
    state.batch = { customers, outcomes: mapping.outcomeColumn >= 0 ? outcomes : null, results: [] };
    button.addEventListener("click", () => scoreFile(resultHost, state.batch, progress, button));
    summaryHost.replaceChildren(h("div", { class: "info-box", id: "batch-summary" },
      h("p", { text: customers.length + " customers found. " + recognised + " columns recognised" + (mapping.ignored.length ? ", ignored: " + mapping.ignored.join(", ") : "") + "." }),
      localProblems ? h("p", { class: "raises", text: localProblems + " rows have unreadable numbers and will be reported as problems." }) : null,
      mapping.outcomeColumn >= 0 ? h("p", { text: "A Churn column was found: known outcomes can be sent as feedback after scoring." }) : null,
      h("div", { class: "row" }, button), progress));
  }

  async function batchPage() {
    const root = h("div", {}, h("h1", { text: "Score a file" }),
      h("p", { class: "muted", text: "Upload a CSV file (or paste its text) with one customer per line. Column names may be ours (tenure_months) or those of the IBM Telco file (tenure, MonthlyCharges). A Churn column, if present, is not used for scoring." }));
    const fields = await loadFields();
    if (!fields) return root.append(h("p", { class: "alert-box", role: "alert", text: "The field list could not be loaded. Try again in a moment." })) || root;
    const summaryHost = h("div", { "aria-live": "polite" });
    const resultHost = h("div");
    const area = h("textarea", { id: "csv-text", rows: "7", "aria-label": "CSV text", placeholder: "customerID,gender,SeniorCitizen,..." });
    const file = h("input", { type: "file", id: "csv-file", accept: ".csv,text/csv", "aria-label": "CSV file" });
    file.addEventListener("change", async () => {
      if (!file.files.length) return;
      const text = await file.files[0].text();
      area.value = text.length > 200000 ? text.slice(0, 2000) + "\n..." : text;
      checkText(text, fields, summaryHost, resultHost);
    });
    const check = h("button", { type: "button", id: "check-csv", text: "Check the data" });
    check.addEventListener("click", () => checkText(area.value, fields, summaryHost, resultHost));
    root.append(h("div", { class: "card" }, h("label", { for: "csv-file", text: "CSV file" }), file,
      h("p", { class: "muted small", text: "or paste the text:" }), area, h("div", { class: "row" }, check)), summaryHost, resultHost);
    return root;
  }

  // ------------------------------------------------------------------ report an outcome
  async function feedbackPage() {
    const status = h("div", { "aria-live": "polite" });
    const id = h("input", { id: "fb-id", type: "text", maxlength: "64", autocomplete: "off" });
    const yes = h("input", { type: "radio", name: "outcome", id: "fb-churned", value: "yes" });
    const no = h("input", { type: "radio", name: "outcome", id: "fb-stayed", value: "no" });
    const submit = h("button", { type: "submit", id: "fb-submit", text: "Send the outcome" });
    const form = h("form", { novalidate: true },
      h("div", {}, h("label", { for: "fb-id" }, "Customer id", h("span", { class: "hint", text: "The id you used when you scored the customer." })), id),
      h("fieldset", {}, h("legend", { text: "What happened?" }),
        h("label", {}, yes, " The customer churned (left)"), h("label", {}, no, " The customer stayed")),
      h("div", { class: "row" }, submit));
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!id.value.trim() || (!yes.checked && !no.checked)) {
        status.replaceChildren(h("div", { class: "alert-box", role: "alert", text: "Please give the customer id and choose what happened." }));
        return;
      }
      submit.disabled = true;
      const answer = await post("/feedback", { customer_id: id.value.trim(), churned: yes.checked, source: "user" });
      submit.disabled = false;
      if (answer.ok) {
        status.replaceChildren(h("div", { class: "info-box", text: "Recorded. The model had put this customer in the " + answer.body.matched.risk_band + " band, so this counts as " + answer.body.counted_as.replace(/_/g, " ") + "." }));
        id.value = ""; yes.checked = no.checked = false;
      } else if (answer.status === 404) {
        status.replaceChildren(h("div", { class: "alert-box", role: "alert", text: "No earlier prediction was found for that id. Score the customer first, using the same id." }));
      } else {
        status.replaceChildren(h("div", { class: "alert-box", role: "alert", text: errorText(answer, "feedback service") }));
      }
    });
    const perf = card("Live performance so far");
    fillPerformance(perf);
    return h("div", {}, h("h1", { text: "Report an outcome" }),
      h("p", { class: "muted", text: "When you learn whether a scored customer really left, report it. The system uses these outcomes to measure its own accuracy, precision and recall, and, with enough of them, to retrain." }),
      h("div", { class: "grid" }, h("section", { class: "card" }, h("h2", { text: "Report one customer" }), form, status), perf.el));
  }

  // ------------------------------------------------------------------ about
  async function aboutPage() {
    const [model, fields] = await Promise.all([loadModel(), loadFields()]);
    const root = h("div", {}, h("h1", { text: "About the model" }));
    root.append(h("section", { class: "card" }, h("h2", { text: "What it does" }),
      h("p", { text: "ChurnSense estimates the chance that a subscription customer will leave, puts the customer in a risk band, and names the main reasons. It is trained on the public IBM Telco Customer Churn sample (7,043 customers)." }),
      model ? h("p", { text: "Serving now: version " + model.version + ", " + (FAMILY[model.family] || model.family) + ", trained " + L.relativeTime(model.trained_at) + "." }) : null,
      model ? h("ul", {}, h("li", { text: "HIGH risk: chance of churn of " + L.percent(model.thresholds.t_high) + " or more. At least half of the customers in this band really leave." }),
        h("li", { text: "MEDIUM risk: from " + L.percent(model.thresholds.t_medium) + ". Together with HIGH this band catches about nine in ten churners." }),
        h("li", { text: "LOW risk: below that." })) : null));
    root.append(h("section", { class: "card" }, h("h2", { text: "How to read the numbers" }),
      h("ul", {}, h("li", { text: "Precision: of the customers called HIGH risk, the share that really left." }),
        h("li", { text: "Recall: of the customers who really left, the share that were called HIGH risk." }),
        h("li", { text: "Accuracy: the share of all customers called correctly." }),
        h("li", { text: "These live numbers are measured on outcomes people report; the dashboard shows them once at least 30 outcomes exist." }),
        h("li", { text: "Reasons are exact SHAP values: they add up to the model's score, so they are an honest account of what moved it." }))));
    root.append(h("section", { class: "card" }, h("h2", { text: "Known limits" }),
      h("ul", {}, h("li", { text: "The model learned mostly from month-to-month customers; it is much weaker for one-year and two-year contracts, where few customers in the data left." }),
        h("li", { text: "Gender and senior citizen status are model inputs. Fairness is not assessed beyond per-group numbers." }),
        h("li", { text: "Outcomes sent by the traffic simulator are synthetic and are marked as such in the data." }))));
    if (fields) {
      root.append(h("section", { class: "card" }, h("h2", { text: "The " + fields.length + " customer fields" }),
        h("div", { class: "table-wrap" }, h("table", {}, h("caption", { text: "Fields the model uses" }),
          h("thead", {}, h("tr", {}, ["Field", "Meaning", "Allowed values"].map((t) => h("th", { scope: "col", text: t })))),
          h("tbody", {}, fields.map((f) => h("tr", {}, h("td", { text: f.name }), h("td", { text: f.description || "" }),
            h("td", { text: f.allowed ? f.allowed.join(", ") : (f.min !== null && f.max !== null ? f.min + " to " + f.max : "number") }))))))));
    }
    return root;
  }

  // ------------------------------------------------------------------ pipeline console
  const STATUS_CSS = { ok: "level-ok", failed: "level-alert", "not run": "level-missing" };
  const OUTCOME_CSS = { promoted: "level-ok", rejected: "level-warn", failed: "level-alert", skipped: "level-missing", rolled_back: "level-warn" };

  function dagSvg(stages, edges) {
    const ns = "http://www.w3.org/2000/svg";
    const make = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, String(v)));
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const layout = L.stageLayout(stages.map((x) => x.stage), edges);
    const { box } = layout;
    const at = new Map(layout.nodes.map((n) => [n.stage, n]));
    const by = new Map(stages.map((x) => [x.stage, x]));
    const svg = make("svg", { class: "dag", viewBox: "0 0 " + layout.width + " " + layout.height, role: "img",
      "aria-label": "Pipeline graph with " + stages.length + " stages: " + stages.map((x) => x.stage + " " + x.status).join(", ") });
    const marker = make("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" });
    marker.append(make("path", { d: "M0,0 L10,5 L0,10 z" }));
    const defs = make("defs");
    defs.append(marker);
    svg.append(defs);
    edges.forEach(([from, to]) => {
      const a = at.get(from);
      const b = at.get(to);
      if (!a || !b) return;
      const x1 = a.x + box.w, y1 = a.y + box.h / 2, x2 = b.x, y2 = b.y + box.h / 2, xm = (x1 + x2) / 2;
      svg.append(make("path", { class: "edge", d: "M" + x1 + "," + y1 + " C" + xm + "," + y1 + " " + xm + "," + y2 + " " + x2 + "," + y2, "marker-end": "url(#arrow)" }));
    });
    layout.nodes.forEach((n) => {
      const info = by.get(n.stage);
      const group = make("g", { class: "node " + (STATUS_CSS[info.status] || "level-missing"), "data-stage": n.stage });
      group.append(make("rect", { x: n.x, y: n.y, width: box.w, height: box.h, rx: 6 }),
        make("text", { class: "stage-name", x: n.x + 8, y: n.y + 20 }, n.stage),
        make("text", { x: n.x + 8, y: n.y + 37 }, info.status === "failed" ? "FAILED" : info.status),
        make("text", { x: n.x + 8, y: n.y + 51 }, info.duration_seconds != null ? L.formatDuration(info.duration_seconds) : ""));
      svg.append(group);
    });
    return svg;
  }

  async function fillGraph(c) {
    const answer = await api("/pipeline/stages");
    if (!answer.ok) return c.fail(errorText(answer, "pipeline status"));
    const { stages, edges } = answer.body;
    const ran = stages.filter((x) => x.duration_seconds != null);
    const total = ran.reduce((sum, x) => sum + x.duration_seconds, 0);
    const slowest = ran.slice().sort((a, b) => b.duration_seconds - a.duration_seconds)[0];
    const rate = (name) => (stages.find((x) => x.stage === name) || {}).rows_per_second;
    c.set(
      dagSvg(stages, edges),
      h("div", { class: "kpis" }, kpi(ran.length + " of " + stages.length, "stages with a report"),
        kpi(L.formatDuration(total), "time of the last run, all stages"), kpi(slowest ? slowest.stage : "n/a", "slowest stage"),
        kpi(L.formatRate(rate("ingest")), "ingest (Ray)"), kpi(L.formatRate(rate("synthesize")), "synthesize (Ray)"), kpi(L.formatRate(rate("featurize")), "featurize (Ray)")),
      h("div", { class: "table-wrap" }, h("table", {}, h("caption", { text: "Stages of the last run" }),
        h("thead", {}, h("tr", {}, ["Stage", "Status", "Time", "Rows out", "Speed", "Note"].map((t) => h("th", { scope: "col", text: t })))),
        h("tbody", {}, stages.map((x) => h("tr", { "data-stage": x.stage }, h("td", { text: x.stage }),
          h("td", {}, badge(x.status === "failed" ? "FAILED" : x.status, STATUS_CSS[x.status] || "level-missing")),
          h("td", { text: x.duration_seconds != null ? L.formatDuration(x.duration_seconds) : "" }),
          h("td", { text: x.rows_out != null ? String(x.rows_out) : "" }), h("td", { text: x.rows_per_second != null ? L.formatRate(x.rows_per_second) : "" }),
          h("td", { text: x.error || x.detail || "" })))))));
  }

  async function fillRuns(c) {
    const answer = await api("/pipeline/runs");
    if (!answer.ok) return c.fail(errorText(answer, "run history"));
    const { runs, pending_requests: pending } = answer.body;
    c.set(
      pending.length ? h("p", { class: "info-box", text: "Waiting for the controller: " + pending.join(", ") + "." }) : null,
      runs.length === 0 ? h("p", { class: "muted", text: "No retraining run has been recorded yet." })
        : h("div", { class: "table-wrap" }, h("table", {}, h("caption", { text: "Retraining and rollback runs, newest first" }),
          h("thead", {}, h("tr", {}, ["When", "Trigger", "Outcome", "Time", "Why / steps"].map((t) => h("th", { scope: "col", text: t })))),
          h("tbody", {}, runs.map((r) => h("tr", { "data-run": r.run_id },
            h("td", { text: r.finished_at ? L.relativeTime(r.finished_at) : r.run_id }), h("td", { text: r.trigger }),
            h("td", {}, badge(r.outcome, OUTCOME_CSS[r.outcome] || "level-missing")),
            h("td", { text: L.formatDuration(r.duration_seconds) }),
            h("td", {}, h("div", { text: r.reason || "" }),
              h("details", { class: "run-steps" }, h("summary", { text: (r.steps || []).length + " steps" }),
                h("ul", {}, (r.steps || []).map((st) => h("li", { class: st.ok ? "" : "step-fail" }, (st.ok ? "ok: " : "FAILED: ") + st.step + (st.detail ? " (" + st.detail + ")" : ""))))))))))));
  }

  async function fillComponents(c) {
    const answer = await api("/components");
    if (!answer.ok) return c.fail(errorText(answer, "component check"));
    const { components, controller_heartbeat_age_seconds: age } = answer.body;
    c.set(h("ul", { class: "component-list" }, Object.entries(components).map(([name, info]) =>
      h("li", {}, badge(info.up ? "running" : "NOT ANSWERING", info.up ? "level-ok" : "level-alert"), " " + name)),
      h("li", {}, badge(age !== null && age < 300 ? "running" : "NOT REPORTING", age !== null && age < 300 ? "level-ok" : "level-alert"),
        " retrain controller" + (age !== null ? " (last sign of life " + Math.round(age) + " s ago)" : ""))));
  }

  function controlCard(refresh) {
    const token = h("input", { id: "admin-token", type: "password", autocomplete: "off", "aria-describedby": "token-hint" });
    const status = h("p", { class: "small", id: "control-status", "aria-live": "polite" });
    const send = async (action, question) => {
      if (!token.value) { status.textContent = "Enter the admin token first."; return; }
      if (!window.confirm(question)) return;
      const answer = await post("/admin/pipeline/request", { action }, { "x-admin-token": token.value });
      status.textContent = answer.status === 202 ? "Request sent. The controller starts within a few seconds; the history below updates by itself."
        : answer.status === 401 ? "The admin token is not correct."
          : answer.status === 409 ? "That action is already waiting for the controller."
            : answer.status === 503 ? "Admin actions are switched off on the server." : errorText(answer, "pipeline console");
      refresh();
    };
    const retrain = h("button", { type: "button", id: "retrain-button", text: "Retrain now" });
    const rollback = h("button", { type: "button", id: "rollback-button", class: "secondary", text: "Roll back to the previous model" });
    retrain.addEventListener("click", () => send("retrain", "Start a retraining now? It uses the newest feedback and takes several minutes. A new model is only used if it passes the gate."));
    rollback.addEventListener("click", () => send("rollback", "Go back to the previous model version? Predictions switch to it immediately."));
    return h("section", { class: "card control-box", "aria-label": "Controls" }, h("h2", { text: "Controls" }),
      h("label", { for: "admin-token" }, "Admin token", h("span", { class: "hint", id: "token-hint", text: "Needed for the two buttons; it is kept only on this page and never stored." })), token,
      h("div", { class: "row" }, retrain, rollback), status);
  }

  async function pipelinePage() {
    const graph = card("Pipeline graph (the DVC stages)");
    const runs = card("Retraining history");
    const comps = card("Components");
    const refresh = () => Promise.all([fillGraph(graph), fillRuns(runs), fillComponents(comps)]);
    refresh();
    state.timer = setInterval(refresh, 10000);
    const base = location.protocol + "//" + location.hostname;
    const tools = h("p", { class: "small muted" }, "More detail in the tools: ",
      [[5000, "MLflow (experiments, registry)"], [3000, "Grafana (live metrics)"], [9090, "Prometheus (alerts)"]].map(([port, name], i) =>
        [i ? ", " : "", h("a", { href: base + ":" + port + "/", target: "_blank", rel: "noopener", text: name })]));
    return h("div", {}, h("h1", { text: "Pipeline console" }),
      h("p", { class: "muted", text: "The data and training pipeline (a DVC graph run with Ray), what each stage did last time and how fast, the history of automatic and manual retraining, and the health of every component. Refreshes every 10 seconds." }),
      graph.el, h("div", { class: "grid" }, runs.el, h("div", {}, comps.el, controlCard(refresh))), tools);
  }

  // ------------------------------------------------------------------ router
  const ROUTES = {
    "/": ["Dashboard", dashboardPage], "/score": ["Score a customer", scorePage], "/batch": ["Score a file", batchPage],
    "/feedback": ["Report an outcome", feedbackPage], "/pipeline": ["Pipeline console", pipelinePage],
    "/about": ["About the model", aboutPage],
  };

  async function render() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    const path = location.hash.replace(/^#/, "") || "/";
    const [title, page] = ROUTES[path] || ROUTES["/"];
    document.title = title + " | ChurnSense";
    document.querySelectorAll("#nav a").forEach((a) => {
      if (a.dataset.route === (ROUTES[path] ? path : "/")) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    view.replaceChildren(h("p", { class: "muted", text: "Loading..." }));
    const content = await page();
    view.replaceChildren(content);
    view.focus();
  }

  window.addEventListener("hashchange", render);
  refreshModelLine();
  render();
  window.ChurnApp = { render, state };
})();
