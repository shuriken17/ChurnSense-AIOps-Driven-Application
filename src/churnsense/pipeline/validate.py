"""Stage 2: clean and validate.

Input:   data/interim/ingested.parquet
Outputs: data/interim/validated.parquet, data/quarantine/rejected.parquet,
         reports/validation_report.json

Cleaning rules (from the design document):

* a blank ``total_charges`` with tenure 0 is a brand-new customer, so it becomes 0.0;
* a blank ``total_charges`` with tenure > 0 is estimated as tenure * monthly_charges and the
  row is flagged ``total_charges_imputed = 1``;
* text is trimmed and its letter case is normalised ("yes" -> "Yes").

Guardrails (any of these fails the stage, so DVC skips everything downstream):

* schema drift: the column set must equal the schema exactly;
* more than ``validation.max_null_frac`` missing values in any column;
* more than ``validation.max_quarantine_frac`` of rows rejected;
* fewer than ``validation.min_rows`` valid rows;
* a churn rate outside ``validation.churn_rate_bounds`` (a sign of a broken label).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from churnsense.config import Params, Paths
from churnsense.logging_utils import get_logger
from churnsense.schema import normalise_categories
from churnsense.tracking import StageError, StageReport, tracked_stage
from churnsense.validation import INGESTED, VALIDATED, check_columns, validate_rows

log = get_logger(__name__)

_INTEGER_COLUMNS = ("senior_citizen", "churn", "tenure_months", "total_charges_imputed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules and add the ``total_charges_imputed`` flag."""
    out = normalise_categories(df)
    out["total_charges_imputed"] = 0
    blank = out["total_charges"].isna()
    new_customer = blank & (out["tenure_months"] == 0)
    estimate = blank & (out["tenure_months"] > 0)
    out.loc[new_customer, "total_charges"] = 0.0
    out.loc[estimate, "total_charges"] = (
        out.loc[estimate, "tenure_months"] * out.loc[estimate, "monthly_charges"]
    )
    out.loc[estimate, "total_charges_imputed"] = 1
    return out


def reason_counts(rejected: pd.DataFrame) -> dict[str, int]:
    """Count rejected rows per kind of problem (the text before the first colon)."""
    counter: Counter[str] = Counter()
    for reason in rejected.get("reason", []):
        for part in str(reason).split("; "):
            counter[part.split(":")[0].strip()] += 1
    return dict(counter.most_common())


@tracked_stage("validate", report="validation_report.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the validate stage."""
    rules = params.validation
    df = pd.read_parquet(paths.ingested)
    rows_in = len(df)
    check_columns(df, INGESTED)  # schema drift raises here

    null_fraction = df.isna().mean()
    null_columns = {c: round(float(f), 4) for c, f in null_fraction.items() if f > 0}
    too_null = [c for c, f in null_columns.items() if f > float(rules.max_null_frac)]
    if too_null:
        raise StageError(
            f"too many missing values (limit {rules.max_null_frac:.0%}) in: {too_null}",
            StageReport(rows_in=rows_in, extra={"null_fraction": null_columns}),
        )

    cleaned = clean(df)
    outcome = validate_rows(cleaned, VALIDATED)
    valid, rejected = outcome.valid, outcome.rejected
    valid = valid.astype({column: "int64" for column in _INTEGER_COLUMNS})
    rejected["stage"] = "validate"

    paths.ensure_parents(paths.validated, paths.quarantine)
    valid.to_parquet(paths.validated, index=False)
    rejected.to_parquet(paths.quarantine, index=False)

    churn_rate = float(valid["churn"].mean()) if len(valid) else float("nan")
    quarantined_fraction = len(rejected) / rows_in if rows_in else 0.0
    report = StageReport(
        rows_in=rows_in,
        rows_out=len(valid),
        extra={
            "rows_rejected": len(rejected),
            "quarantined_fraction": round(quarantined_fraction, 4),
            "rejection_reasons": reason_counts(rejected),
            "null_fraction": null_columns,
            "total_charges_new_customers_set_to_zero": int(
                ((df["total_charges"].isna()) & (df["tenure_months"] == 0)).sum()
            ),
            "total_charges_imputed": int(valid["total_charges_imputed"].sum()),
            "churn_rate": round(churn_rate, 4),
            "valid_rows_by_source": {
                str(k): int(v) for k, v in valid["source"].value_counts().items()
            },
        },
    )

    if quarantined_fraction > float(rules.max_quarantine_frac):
        raise StageError(
            f"{len(rejected)} of {rows_in} rows rejected (limit {rules.max_quarantine_frac:.0%})",
            report,
        )
    if len(valid) < int(rules.min_rows):
        raise StageError(f"only {len(valid)} valid rows (need {rules.min_rows})", report)
    low, high = (float(b) for b in rules.churn_rate_bounds)
    if not low <= churn_rate <= high:
        raise StageError(
            f"churn rate {churn_rate:.3f} is outside the allowed range {low}-{high}", report
        )
    return report


if __name__ == "__main__":
    run.main()
