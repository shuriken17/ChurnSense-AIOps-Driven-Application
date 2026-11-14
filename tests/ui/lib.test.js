"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../frontend/lib.js");
const { FIELDS } = require("./harness.js");

test("parseCsv handles quotes, escaped quotes, commas, CRLF, BOM and empty lines", () => {
  assert.deepEqual(L.parseCsv('a,b\r\n"x,1","he said ""hi"""\n\n3,4'), [["a", "b"], ["x,1", 'he said "hi"'], ["3", "4"]]);
  assert.deepEqual(L.parseCsv("\uFEFFa,b\n1,2\n"), [["a", "b"], ["1", "2"]]);
  assert.deepEqual(L.parseCsv("a\rb\r"), [["a"], ["b"]]);
  assert.deepEqual(L.parseCsv('a,"multi\nline",c'), [["a", "multi\nline", "c"]]);
  assert.deepEqual(L.parseCsv("a,,c\n,,\n"), [["a", "", "c"], ["", "", ""]]);
  assert.deepEqual(L.parseCsv(""), []);
  assert.deepEqual(L.parseCsv('"",x\n'), [["", "x"]]);
});

test("headers match our names and the IBM names, and report what is missing or ignored", () => {
  const ibm = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"];
  const m = L.mapHeaders(ibm, FIELDS);
  assert.deepEqual(m.missing, []);
  assert.deepEqual(m.ignored, ["Churn"]);
  assert.equal(m.outcomeColumn, 20);
  assert.equal(m.columns[5].field.name, "tenure_months");
  const ours = L.mapHeaders(FIELDS.map((f) => f.name), FIELDS);
  assert.deepEqual(ours.missing, []);
  assert.equal(ours.outcomeColumn, -1);
});

test("missing required columns are listed, optional ones are not, duplicate columns are ignored", () => {
  const m = L.mapHeaders(["gender", "contract", "contract", "mystery"], FIELDS);
  assert.ok(m.missing.includes("tenure_months") && m.missing.includes("monthly_charges"));
  assert.ok(!m.missing.includes("customer_id") && !m.missing.includes("total_charges"));
  assert.deepEqual(m.ignored, ["contract", "mystery"]);
});

test("a row becomes a typed customer; blanks are left out; bad numbers are reported", () => {
  const m = L.mapHeaders(["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "SeniorCitizen"], FIELDS);
  const ok = L.toCustomer(["A-1", "12", "29.85", " ", "month-to-month", "0"], m);
  assert.deepEqual(ok.customer, { customer_id: "A-1", tenure_months: 12, monthly_charges: 29.85, contract: "Month-to-month", senior_citizen: 0 });
  assert.deepEqual(ok.problems, []);
  const bad = L.toCustomer(["A-2", "twelve", "1e3", "", "One year", "0"], m);
  assert.deepEqual(bad.problems.map((p) => p.field), ["tenure_months", "monthly_charges"]);
  assert.equal(L.toCustomer(["x"], m).customer.customer_id, "x");
});

test("outcomes: Yes/No/1/0/true/false, anything else is unknown", () => {
  assert.deepEqual(["Yes", "no", "1", "0", "TRUE", "false", "churned", "stayed", "", "maybe", null].map(L.parseOutcome),
    [true, false, true, false, true, false, true, false, null, null, null]);
});

test("every factor feature gets a plain sentence with the right direction", () => {
  const f = (feature, value, direction = "increases_risk") => L.explainFactor({ feature, value, direction, impact: 0.1 });
  assert.equal(f("tenure_months", 5).text, "Has been a customer for 5 months");
  assert.equal(f("tenure_months", 1).text, "Has been a customer for 1 month");
  assert.equal(f("monthly_charges", 79.85).text, "Pays 79.85 a month");
  assert.equal(f("charge_delta", -12.5).text, "Current bill is 12.50 below their average");
  assert.equal(f("charge_delta", 3).text, "Current bill is 3.00 above their average");
  assert.equal(f("n_addons", 0).text, "Uses 0 add-on services");
  assert.equal(f("is_month_to_month", 1).text, "Is on a month-to-month contract");
  assert.equal(f("is_month_to_month", 0).text, "Is not on a month-to-month contract");
  assert.equal(f("is_fiber", 1).text, "Uses fiber optic internet");
  assert.equal(f("is_auto_pay", 0).text, "Does not pay automatically");
  assert.equal(f("contract", "Two year").text, "Has contract: Two year");
  assert.equal(f("senior_citizen", 1).text, "Has senior citizen: yes");
  assert.equal(f("online_security", null).text, "Unknown online security");
  assert.equal(f("some_new_feature", 3).text, "Has some new feature: 3");
  assert.match(f("is_fiber", 0, "decreases_risk").arrow, /lowers the risk/);
  assert.equal(f("is_fiber", 1).raises, true);
});

test("bands and numbers are formatted for people", () => {
  assert.equal(L.bandInfo("HIGH").label, "High risk");
  assert.equal(L.bandInfo("LOW").symbol, "\u25BC");
  assert.equal(L.bandInfo("ODD").label, "ODD");
  assert.equal(L.percent(0.7023), "70.2 %");
  assert.equal(L.percent(0.5, 0), "50 %");
  assert.equal(L.percent(null), "n/a");
  assert.equal(L.percent(NaN), "n/a");
});

test("CSV output quotes cells and neutralises spreadsheet formulas", () => {
  assert.equal(L.csvCell("a,b"), '"a,b"');
  assert.equal(L.csvCell('say "hi"'), '"say ""hi"""');
  assert.equal(L.csvCell("=HYPERLINK(1)"), "'=HYPERLINK(1)");
  assert.equal(L.csvCell("@cmd"), "'@cmd");
  assert.equal(L.csvCell("-5+3"), "'-5+3");
  assert.equal(L.csvCell(-5), "-5");  // a real number is not a formula
  assert.equal(L.csvCell(null), "");
  assert.equal(L.toCsv(["a", "b"], [[1, "x,y"]]), 'a,b\r\n1,"x,y"\r\n');
});

test("chunking, PSI levels, drift words and relative times", () => {
  assert.deepEqual(L.chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(L.chunk([], 3), []);
  assert.equal(L.psiLevel(0.05).level, "ok");
  assert.equal(L.psiLevel(0.1).level, "warn");
  assert.equal(L.psiLevel(0.2499).level, "warn");
  assert.equal(L.psiLevel(0.25).level, "alert");
  assert.equal(L.driftText("alert"), "ALERT: large shift");
  assert.equal(L.driftText("nonsense"), "Unknown");
  const now = Date.parse("2026-11-11T12:00:00Z");
  assert.equal(L.relativeTime("2026-11-11T11:59:30Z", now), "30 seconds ago");
  assert.equal(L.relativeTime("2026-11-11T11:59:59Z", now), "1 second ago");
  assert.equal(L.relativeTime("2026-11-11T11:30:00Z", now), "30 minutes ago");
  assert.equal(L.relativeTime("2026-11-11T11:00:00Z", now), "1 hour ago");
  assert.equal(L.relativeTime("2026-11-11T07:00:00Z", now), "5 hours ago");
  assert.equal(L.relativeTime("2026-11-09T12:00:00Z", now), "2 days ago");
  assert.equal(L.relativeTime("2026-11-10T12:00:00Z", now), "1 day ago");
  assert.equal(L.relativeTime("2027-01-01T00:00:00Z", now), "0 seconds ago");  // the future is shown as now
  assert.equal(L.relativeTime("not a date", now), "unknown");
});


test("the graph layout puts every stage one column right of its furthest predecessor", () => {
  const { STAGES, EDGES } = require("./harness.js");
  const layout = L.stageLayout(STAGES, EDGES);
  const column = Object.fromEntries(layout.nodes.map((n) => [n.stage, n.column]));
  assert.deepEqual(column, { ingest: 0, validate: 1, eda: 2, split: 2, synthesize: 3, featurize: 4, train: 5, evaluate: 6, gate: 7 });
  for (const [from, to] of EDGES) assert.ok(column[to] > column[from], from + " -> " + to);
  const at = Object.fromEntries(layout.nodes.map((n) => [n.stage, n]));
  assert.notEqual(at.eda.y, at.split.y);                     // two stages in one column do not overlap
  assert.ok(layout.width > 0 && layout.height > 0);
  assert.ok(layout.nodes.every((n) => n.x + layout.box.w <= layout.width && n.y + layout.box.h <= layout.height));
});

test("the layout copes with a single stage, unknown edge ends and (by design) never loops on a cycle", () => {
  assert.equal(L.stageLayout(["a"], []).nodes[0].column, 0);
  assert.equal(L.stageLayout(["a", "b"], [["a", "ghost"]]).nodes.length, 2);
  const looped = L.stageLayout(["a", "b"], [["a", "b"], ["b", "a"]]);
  assert.equal(looped.nodes.length, 2);
});

test("durations and speeds are written for people", () => {
  assert.equal(L.formatDuration(0.24), "240 ms");
  assert.equal(L.formatDuration(11.34), "11.3 s");
  assert.equal(L.formatDuration(163), "2 min 43 s");
  assert.equal(L.formatDuration(null), "n/a");
  assert.equal(L.formatRate(21635.2), "21,635 rows/s");
  assert.equal(L.formatRate(undefined), "n/a");
});
