"""EDA stage: automated exploratory data analysis.

The course guidelines ask for EDA that "understands data characteristics, identifies patterns
and detects potential issues (missing values, outliers)". Running it as a DVC stage makes it
versioned and repeatable: if the data changes, the plots and numbers are regenerated.

Input:   data/interim/validated.parquet
Outputs: reports/eda/*.png (DVC-tracked), reports/eda_summary.json (small, kept in Git)

It looks at the REAL rows only. The findings are written up in docs/eda.md.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on the server

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from churnsense.config import Params, Paths  # noqa: E402
from churnsense.logging_utils import get_logger  # noqa: E402
from churnsense.schema import TENURE_BINS, TENURE_LABELS  # noqa: E402
from churnsense.tracking import StageReport, tracked_stage  # noqa: E402

log = get_logger(__name__)

# Colour-blind safe pair (blue and orange).
BLUE, ORANGE = "#0072B2", "#E69F00"
NUMERIC = ("tenure_months", "monthly_charges", "total_charges")
SEGMENTS = (
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "senior_citizen",
    "gender",
    "partner",
    "dependents",
)


def churn_rate_by(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    """Churn rate and customer count for every value of ``column``."""
    grouped = df.groupby(column, observed=True)["churn"].agg(["mean", "count"])
    return {
        str(k): {"churn_rate": round(float(r["mean"]), 4), "customers": int(r["count"])}
        for k, r in grouped.iterrows()
    }


def iqr_outliers(series: pd.Series) -> dict[str, float]:
    """Count values outside 1.5 x IQR from the quartiles (the usual box-plot rule)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    return {
        "lower_fence": round(float(low), 2),
        "upper_fence": round(float(high), 2),
        "outliers": int(((series < low) | (series > high)).sum()),
    }


def _save(fig: plt.Figure, path: Any) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _bar_rates(ax: plt.Axes, rates: dict[str, dict[str, float]], title: str) -> None:
    labels = list(rates)
    values = [rates[k]["churn_rate"] * 100 for k in labels]
    ax.bar(labels, values, color=BLUE)
    ax.set_title(title)
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=20)


def make_plots(df: pd.DataFrame, directory: Any) -> list[str]:
    """Write the EDA figures and return their file names."""
    directory.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_rates(ax, churn_rate_by(df, "contract"), "Churn rate by contract type")
    _save(fig, directory / "churn_by_contract.png")
    names.append("churn_by_contract.png")

    bucket = pd.cut(
        df["tenure_months"], bins=list(TENURE_BINS), labels=list(TENURE_LABELS), include_lowest=True
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_rates(
        ax,
        churn_rate_by(df.assign(tenure_bucket=bucket), "tenure_bucket"),
        "Churn rate by tenure (months)",
    )
    _save(fig, directory / "churn_by_tenure.png")
    names.append("churn_by_tenure.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for value, colour, label in ((0, BLUE, "stayed"), (1, ORANGE, "churned")):
        ax.hist(
            df.loc[df["churn"] == value, "monthly_charges"],
            bins=30,
            alpha=0.6,
            color=colour,
            label=label,
        )
    ax.set_title("Monthly charges, stayed vs churned")
    ax.set_xlabel("Monthly charges")
    ax.set_ylabel("Customers")
    ax.legend()
    _save(fig, directory / "monthly_charges_by_churn.png")
    names.append("monthly_charges_by_churn.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, column in zip(
        axes.flat,
        ("internet_service", "payment_method", "paperless_billing", "senior_citizen"),
        strict=True,
    ):
        _bar_rates(ax, churn_rate_by(df, column), column.replace("_", " ").capitalize())
    _save(fig, directory / "churn_by_category.png")
    names.append("churn_by_category.png")

    numeric = df[[*NUMERIC, "senior_citizen", "churn"]]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), labels=corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation of numeric fields")
    _save(fig, directory / "correlation.png")
    names.append("correlation.png")
    return names


@tracked_stage("eda", report="eda_summary.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the EDA stage."""
    df = pd.read_parquet(paths.validated)
    real = df[df["source"] == "real"].reset_index(drop=True)
    figures = make_plots(real, paths.eda_dir)

    correlation = real[[*NUMERIC, "senior_citizen", "churn"]].corr()["churn"].drop("churn")
    summary: dict[str, Any] = {
        "rows_real": len(real),
        "columns": int(df.shape[1]),
        "churn_rate": round(float(real["churn"].mean()), 4),
        "class_balance": {
            "stayed": int((real["churn"] == 0).sum()),
            "churned": int((real["churn"] == 1).sum()),
        },
        "missing_values_after_cleaning": int(real.isna().sum().sum()),
        "total_charges_imputed": int(real["total_charges_imputed"].sum()),
        "new_customers_tenure_0": int((real["tenure_months"] == 0).sum()),
        "duplicate_customer_ids": int(real["customer_id"].duplicated().sum()),
        "numeric_summary": {
            c: {
                k: round(float(v), 2) for k, v in real[c].agg(["mean", "std", "min", "max"]).items()
            }
            for c in NUMERIC
        },
        "outliers_iqr_rule": {c: iqr_outliers(real[c]) for c in NUMERIC},
        "correlation_with_churn": {k: round(float(v), 3) for k, v in correlation.items()},
        "churn_rate_by_segment": {c: churn_rate_by(real, c) for c in SEGMENTS},
        "figures": figures,
        "note": "gender and senior_citizen are model inputs; fairness is not assessed "
        "beyond these per-segment churn rates",
    }
    log.info("eda finished", extra={"figures": len(figures), "rows": len(real)})
    return StageReport(rows_in=len(df), rows_out=len(real), extra=summary)


if __name__ == "__main__":
    run.main()
