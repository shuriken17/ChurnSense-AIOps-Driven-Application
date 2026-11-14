/* Pure helper functions of the ChurnSense web interface (no DOM, no network).
   Loaded by the page as window.ChurnLib and by the tests with require(). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ChurnLib = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const ALIASES = { tenure: "tenure_months" }; // the IBM file calls it "tenure"
  const IGNORED_OUTCOME_HEADERS = ["churn"]; // the label column of the IBM file

  function normalizeHeader(header) {
    return String(header).replace(/^\uFEFF/, "").toLowerCase().replace(/[^a-z0-9]/g, "");
  }

  /* RFC 4180 style: quoted cells, "" as an escaped quote, CRLF / LF / CR line ends.
     Lines that are completely empty are skipped. Returns an array of rows (arrays of strings). */
  function parseCsv(text) {
    const rows = [];
    let row = [];
    let cell = "";
    let quoted = false;
    let wasQuoted = false;
    const endCell = () => { row.push(cell); cell = ""; wasQuoted = false; };
    const endRow = () => {
      endCell();
      if (!(row.length === 1 && row[0] === "" && !wasQuoted)) rows.push(row);
      row = [];
    };
    const source = String(text).replace(/^\uFEFF/, "");
    for (let i = 0; i < source.length; i++) {
      const c = source[i];
      if (quoted) {
        if (c === '"' && source[i + 1] === '"') { cell += '"'; i++; }
        else if (c === '"') quoted = false;
        else cell += c;
      } else if (c === '"' && cell === "") { quoted = true; wasQuoted = true; }
      else if (c === ",") endCell();
      else if (c === "\r") { if (source[i + 1] === "\n") i++; endRow(); }
      else if (c === "\n") endRow();
      else cell += c;
    }
    if (cell !== "" || row.length > 0) endRow();
    return rows;
  }

  /* Match the header cells of a file with the field list of the API. */
  function mapHeaders(headers, fields) {
    const byKey = new Map(fields.map((f) => [normalizeHeader(f.name), f]));
    const used = new Set();
    const columns = headers.map((header, index) => {
      const key = normalizeHeader(header);
      const field = byKey.get(key) || byKey.get(normalizeHeader(ALIASES[key] || "")) || null;
      if (field && used.has(field.name)) return { index, header, field: null, duplicate: true };
      if (field) used.add(field.name);
      return { index, header, field };
    });
    return {
      columns,
      missing: fields.filter((f) => f.required && !used.has(f.name)).map((f) => f.name),
      ignored: columns.filter((c) => !c.field).map((c) => c.header),
      outcomeColumn: columns.findIndex((c) => !c.field && IGNORED_OUTCOME_HEADERS.includes(normalizeHeader(c.header))),
    };
  }

  function canonicalCategory(value, allowed) {
    const wanted = value.trim().toLowerCase();
    return (allowed || []).find((a) => a.toLowerCase() === wanted) || value.trim();
  }

  /* Turn one row of strings into a typed customer; empty cells are left out. */
  function toCustomer(cells, mapping) {
    const customer = {};
    const problems = [];
    for (const column of mapping.columns) {
      if (!column.field) continue;
      const raw = (cells[column.index] ?? "").trim();
      if (raw === "") continue;
      const { name, kind, allowed } = column.field;
      if (kind === "id") customer[name] = raw;
      else if (kind === "category") customer[name] = canonicalCategory(raw, allowed);
      else if (/^-?\d+(\.\d+)?$/.test(raw)) customer[name] = Number(raw);
      else problems.push({ field: name, message: "must be a number, found '" + raw.slice(0, 20) + "'" });
    }
    return { customer, problems };
  }

  /* Outcome of the IBM "Churn" column: Yes/No/1/0/true/false, else null. */
  function parseOutcome(value) {
    const v = String(value ?? "").trim().toLowerCase();
    if (["yes", "1", "true", "churned"].includes(v)) return true;
    if (["no", "0", "false", "stayed"].includes(v)) return false;
    return null;
  }

  const BANDS = {
    HIGH: { label: "High risk", symbol: "\u25B2", css: "band-high", advice: "Contact this customer soon." },
    MEDIUM: { label: "Medium risk", symbol: "\u25A0", css: "band-medium", advice: "Keep an eye on this customer." },
    LOW: { label: "Low risk", symbol: "\u25BC", css: "band-low", advice: "No action needed now." },
  };
  const bandInfo = (band) => BANDS[band] || { label: String(band), symbol: "?", css: "band-unknown", advice: "" };

  const percent = (p, digits = 1) => (p === null || p === undefined || Number.isNaN(p) ? "n/a" : (p * 100).toFixed(digits) + " %");
  const money = (v) => Number(v).toFixed(2);

  const LABELS = {
    contract: "contract", payment_method: "payment method", internet_service: "internet service",
    paperless_billing: "paperless billing", senior_citizen: "senior citizen", partner: "partner",
    dependents: "dependents", gender: "gender", phone_service: "phone service",
    multiple_lines: "multiple lines", online_security: "online security", online_backup: "online backup",
    device_protection: "device protection", tech_support: "tech support", streaming_tv: "streaming TV",
    streaming_movies: "streaming movies",
  };

  /* One sentence for one SHAP factor: what the customer is like, and which way it pushes. */
  function explainFactor(factor) {
    const { feature, value } = factor;
    const yes = value === 1 || value === "1" || value === true;
    let text;
    switch (feature) {
      case "tenure_months": text = "Has been a customer for " + value + (value === 1 ? " month" : " months"); break;
      case "tenure_bucket": text = "Is in the " + value + " month tenure group"; break;
      case "monthly_charges": text = "Pays " + money(value) + " a month"; break;
      case "total_charges": text = "Has paid " + money(value) + " in total so far"; break;
      case "avg_charge_per_month": text = "Has paid " + money(value) + " a month on average"; break;
      case "charge_delta": text = "Current bill is " + money(Math.abs(value)) + (value >= 0 ? " above" : " below") + " their average"; break;
      case "n_addons": text = "Uses " + value + (value === 1 ? " add-on service" : " add-on services"); break;
      case "is_month_to_month": text = yes ? "Is on a month-to-month contract" : "Is not on a month-to-month contract"; break;
      case "is_fiber": text = yes ? "Uses fiber optic internet" : "Does not use fiber optic internet"; break;
      case "is_auto_pay": text = yes ? "Pays automatically" : "Does not pay automatically"; break;
      default:
        text = (value === null || value === undefined)
          ? "Unknown " + (LABELS[feature] || feature.replace(/_/g, " "))
          : "Has " + (LABELS[feature] || feature.replace(/_/g, " ")) + ": " + (feature === "senior_citizen" ? (yes ? "yes" : "no") : value);
    }
    const raises = factor.direction === "increases_risk";
    return { text, raises, arrow: raises ? "\u25B2 raises the risk" : "\u25BC lowers the risk", impact: factor.impact };
  }

  /* CSV output. Cells starting with = + - @ are prefixed so a spreadsheet cannot run them as formulas. */
  function csvCell(value) {
    let text = value === null || value === undefined ? "" : String(value);
    if (/^[=+\-@\t\r]/.test(text) && typeof value === "string") text = "'" + text;
    return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  const toCsv = (header, rows) => [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n";

  const chunk = (items, size) => {
    const out = [];
    for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
    return out;
  };

  /* PSI reading: below 0.10 no change, 0.10 to 0.25 moderate, above 0.25 large. */
  function psiLevel(value) {
    if (value >= 0.25) return { level: "alert", label: "large shift" };
    if (value >= 0.1) return { level: "warn", label: "moderate shift" };
    return { level: "ok", label: "stable" };
  }

  const DRIFT_TEXT = {
    ok: "Stable", warn: "Moderate shift", alert_candidate: "Large shift (confirming)",
    alert: "ALERT: large shift", insufficient_data: "Too little data yet",
  };
  const driftText = (status) => DRIFT_TEXT[status] || "Unknown";

  function relativeTime(iso, now = Date.now()) {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "unknown";
    const s = Math.max(0, Math.round((now - t) / 1000));
    const plural = (n, unit) => n + " " + unit + (n === 1 ? "" : "s") + " ago";
    if (s < 60) return plural(s, "second");
    if (s < 3600) return plural(Math.round(s / 60), "minute");
    if (s < 86400) return plural(Math.round(s / 3600), "hour");
    return plural(Math.round(s / 86400), "day");
  }


  /* Place the pipeline stages in columns: a stage sits one column right of its furthest predecessor.
     Returns {nodes: [{stage, column, row, x, y}], width, height}. Pure, so it can be tested. */
  function stageLayout(stageNames, edges, box = { w: 104, h: 58, gapX: 22, gapY: 26, pad: 12 }) {
    const depth = new Map(stageNames.map((n) => [n, 0]));
    for (let pass = 0; pass < stageNames.length; pass++) {
      for (const [from, to] of edges) {
        if (depth.has(from) && depth.has(to) && depth.get(to) < depth.get(from) + 1) depth.set(to, depth.get(from) + 1);
      }
    }
    const rowsUsed = new Map();
    const nodes = stageNames.map((stage) => {
      const column = depth.get(stage);
      const row = rowsUsed.get(column) || 0;
      rowsUsed.set(column, row + 1);
      return { stage, column, row, x: box.pad + column * (box.w + box.gapX), y: box.pad + row * (box.h + box.gapY) };
    });
    const columns = Math.max(...nodes.map((n) => n.column)) + 1;
    const rows = Math.max(...nodes.map((n) => n.row)) + 1;
    return { nodes, box, width: box.pad * 2 + columns * box.w + (columns - 1) * box.gapX,
      height: box.pad * 2 + rows * box.h + (rows - 1) * box.gapY };
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "n/a";
    if (seconds < 1) return Math.round(seconds * 1000) + " ms";
    if (seconds < 90) return seconds.toFixed(1) + " s";
    return Math.floor(seconds / 60) + " min " + Math.round(seconds % 60) + " s";
  }

  const formatRate = (v) => (v === null || v === undefined ? "n/a" : Math.round(v).toLocaleString("en-US") + " rows/s");

  return { normalizeHeader, parseCsv, mapHeaders, toCustomer, parseOutcome, bandInfo, percent, money,
    explainFactor, toCsv, csvCell, chunk, psiLevel, driftText, relativeTime, stageLayout, formatDuration,
    formatRate, BANDS };
});
