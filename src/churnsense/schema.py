"""Canonical data schema: the single source of truth for ChurnSense.

Every other part of the system derives from this module: the ingest and validation
stages, the model's feature list, the MLflow signature, the FastAPI request models and
the frontend form (through GET /api/v1/schema). If a field changes, it changes here only.

The raw fields follow the IBM Telco Customer Churn dataset. Names are snake_case; the
original IBM header for each field is kept so that either spelling is accepted on input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when data does not match the canonical schema in a way we cannot repair."""


# --------------------------------------------------------------------------- vocabularies
YES_NO: Final = ("Yes", "No")
NO_PHONE: Final = "No phone service"
NO_INTERNET: Final = "No internet service"
ADDON_VALUES: Final = ("Yes", "No", NO_INTERNET)

#: Where a row came from. Real vs synthetic must always be visible (course guideline).
SOURCES: Final = ("real", "synthetic", "feedback")

#: The IBM file has no timestamps. Every real row gets this nominal time, so any feedback
#: row (always later) wins when duplicates are removed ("keep the latest").
NOMINAL_RAW_EVENT_TS: Final = pd.Timestamp("2018-01-01T00:00:00", tz="UTC")
RAW_BATCH_ID: Final = "raw"


@dataclass(frozen=True)
class FieldSpec:
    """Description of one input field.

    Attributes:
        name: canonical snake_case name used everywhere inside ChurnSense.
        ibm_name: header used in the IBM Telco CSV.
        kind: one of "id", "category", "binary", "int", "float".
        allowed: allowed values for category fields (exact spelling).
        min_value: lower bound for numeric fields.
        max_value: upper bound for numeric fields.
        min_exclusive: True if min_value itself is not allowed (e.g. charges must be > 0).
        nullable: True if the value may be missing (only total_charges).
        required: False if the API may omit the field (customer_id, total_charges).
        description: plain-language meaning, shown in the frontend and the docs.
    """

    name: str
    ibm_name: str
    kind: str
    allowed: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_exclusive: bool = False
    nullable: bool = False
    required: bool = True
    description: str = ""


def _cat(name: str, ibm: str, allowed: tuple[str, ...], text: str) -> FieldSpec:
    """Shorthand for a categorical field."""
    return FieldSpec(name, ibm, "category", allowed=allowed, description=text)


#: The 20 input fields, in the order they appear in the IBM file.
INPUT_FIELDS: Final[tuple[FieldSpec, ...]] = (
    FieldSpec(
        "customer_id",
        "customerID",
        "id",
        required=False,
        description="Unique customer identifier (optional when scoring).",
    ),
    _cat("gender", "gender", ("Male", "Female"), "Customer gender."),
    FieldSpec(
        "senior_citizen",
        "SeniorCitizen",
        "binary",
        min_value=0,
        max_value=1,
        description="1 if the customer is a senior citizen, otherwise 0.",
    ),
    _cat("partner", "Partner", YES_NO, "Has a partner."),
    _cat("dependents", "Dependents", YES_NO, "Has dependents."),
    FieldSpec(
        "tenure_months",
        "tenure",
        "int",
        min_value=0,
        max_value=120,
        description="Months the customer has stayed with the company.",
    ),
    _cat("phone_service", "PhoneService", YES_NO, "Has phone service."),
    _cat("multiple_lines", "MultipleLines", (*YES_NO, NO_PHONE), "Has multiple phone lines."),
    _cat(
        "internet_service",
        "InternetService",
        ("DSL", "Fiber optic", "No"),
        "Type of internet service.",
    ),
    _cat("online_security", "OnlineSecurity", ADDON_VALUES, "Has online security add-on."),
    _cat("online_backup", "OnlineBackup", ADDON_VALUES, "Has online backup add-on."),
    _cat("device_protection", "DeviceProtection", ADDON_VALUES, "Has device protection add-on."),
    _cat("tech_support", "TechSupport", ADDON_VALUES, "Has tech support add-on."),
    _cat("streaming_tv", "StreamingTV", ADDON_VALUES, "Has streaming TV add-on."),
    _cat("streaming_movies", "StreamingMovies", ADDON_VALUES, "Has streaming movies add-on."),
    _cat("contract", "Contract", ("Month-to-month", "One year", "Two year"), "Contract type."),
    _cat("paperless_billing", "PaperlessBilling", YES_NO, "Uses paperless billing."),
    _cat(
        "payment_method",
        "PaymentMethod",
        (
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ),
        "How the customer pays.",
    ),
    FieldSpec(
        "monthly_charges",
        "MonthlyCharges",
        "float",
        min_value=0,
        max_value=500,
        min_exclusive=True,
        description="Current monthly bill.",
    ),
    FieldSpec(
        "total_charges",
        "TotalCharges",
        "float",
        min_value=0,
        max_value=100_000,
        nullable=True,
        required=False,
        description="Total billed so far (estimated from tenure when missing).",
    ),
)

TARGET_FIELD: Final = FieldSpec(
    "churn",
    "Churn",
    "binary",
    min_value=0,
    max_value=1,
    description="1 if the customer left (label).",
)

ID_COLUMN: Final = "customer_id"
TARGET: Final = "churn"
INPUT_COLUMNS: Final = tuple(f.name for f in INPUT_FIELDS)  # 20 fields
RAW_COLUMNS: Final = (*INPUT_COLUMNS, TARGET)  # 21 fields = the IBM file
#: Columns added by the ingest stage.
LINEAGE_COLUMNS: Final = ("source", "event_ts", "batch_id")
INGESTED_COLUMNS: Final = (*RAW_COLUMNS, *LINEAGE_COLUMNS)
#: The validate stage adds a flag for total_charges that had to be estimated.
VALIDATED_COLUMNS: Final = (*INGESTED_COLUMNS, "total_charges_imputed")

ADDON_COLUMNS: Final = (
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
)
FIELD_BY_NAME: Final = {f.name: f for f in (*INPUT_FIELDS, TARGET_FIELD)}
CATEGORICAL_INPUTS: Final = tuple(f.name for f in INPUT_FIELDS if f.kind == "category")

# --------------------------------------------------------------------------- features
#: Order of the tenure buckets; the labels are also the category values.
TENURE_BINS: Final = (0, 6, 12, 24, 48, 1000)
TENURE_LABELS: Final = ("0-6", "7-12", "13-24", "25-48", "49+")

#: The model input: 19 raw features (everything except customer_id) + 7 derived ones.
RAW_FEATURES: Final = tuple(c for c in INPUT_COLUMNS if c != ID_COLUMN)
DERIVED_FEATURES: Final = (
    "tenure_bucket",
    "avg_charge_per_month",
    "charge_delta",
    "n_addons",
    "is_month_to_month",
    "is_fiber",
    "is_auto_pay",
)
FEATURE_COLUMNS: Final = (*RAW_FEATURES, *DERIVED_FEATURES)  # 26 features
CATEGORICAL_FEATURES: Final = (*CATEGORICAL_INPUTS, "tenure_bucket")
NUMERIC_FEATURES: Final = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES)
#: Numeric features watched for data drift (PSI), as in the design document.
DRIFT_NUMERIC_FEATURES: Final = (
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_charge_per_month",
    "charge_delta",
    "n_addons",
)

CATEGORY_VALUES: Final[dict[str, tuple[str, ...]]] = {
    **{f.name: f.allowed or () for f in INPUT_FIELDS if f.kind == "category"},
    "tenure_bucket": TENURE_LABELS,
}


def categorical_dtype(column: str) -> pd.CategoricalDtype:
    """Return the fixed pandas categorical dtype of a feature (same order every time).

    LightGBM uses these categories natively. A value outside the list becomes missing
    (NaN), which is how unseen categories are handled at serving time.
    """
    try:
        return pd.CategoricalDtype(categories=list(CATEGORY_VALUES[column]), ordered=False)
    except KeyError as exc:
        raise SchemaError(f"{column!r} is not a categorical field") from exc


# --------------------------------------------------------------------------- header mapping
def normalise_header(name: str) -> str:
    """Lower-case a header and drop everything except letters and digits."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_header_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for spec in (*INPUT_FIELDS, TARGET_FIELD):
        lookup[normalise_header(spec.name)] = spec.name
        lookup[normalise_header(spec.ibm_name)] = spec.name
    for extra in (*LINEAGE_COLUMNS, "total_charges_imputed"):
        lookup[normalise_header(extra)] = extra
    return lookup


_HEADER_LOOKUP: Final = _build_header_lookup()


def canonical_name(header: str) -> str | None:
    """Map a header (snake_case or IBM spelling) to the canonical name, or None."""
    return _HEADER_LOOKUP.get(normalise_header(header))


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known headers to canonical names; unknown headers are left untouched.

    Unknown headers are not dropped here, so the validation stage can report them as
    schema drift instead of hiding them.

    Raises:
        SchemaError: if two headers map to the same canonical name.
    """
    mapping = {}
    for col in df.columns:
        mapped = canonical_name(col)
        if mapped is not None:
            mapping[col] = mapped
    targets = list(mapping.values())
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise SchemaError(f"several headers map to the same field: {duplicates}")
    return df.rename(columns=mapping)


# --------------------------------------------------------------------------- repair helpers
def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and fix the letter case of categorical values.

    "yes" becomes "Yes", " month-to-month " becomes "Month-to-month". A value that is not
    a known category (even ignoring case) is left as it is, so validation can report it.
    """
    out = df.copy()
    for column, allowed in CATEGORY_VALUES.items():
        if column not in out.columns or column == "tenure_bucket":
            continue
        lookup = {value.lower(): value for value in allowed}
        stripped = (
            out[column].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        )
        out[column] = stripped.map(
            lambda v, lk=lookup: lk.get(v.lower(), v) if isinstance(v, str) else v
        )
    return out


def cross_field_violations(df: pd.DataFrame) -> pd.Series:
    """Return a reason string for every row that breaks a cross-field rule ("" if fine).

    Rules:
        phone_service is "No" if and only if multiple_lines is "No phone service";
        internet_service is "No" if and only if all six add-ons are "No internet service".
    """
    phone_bad = (df["phone_service"] == "No") != (df["multiple_lines"] == NO_PHONE)
    all_marked = (df[list(ADDON_COLUMNS)] == NO_INTERNET).all(axis=1)
    net_bad = (df["internet_service"] == "No") != all_marked
    msg_phone = "cross-field: phone_service and multiple_lines disagree"
    msg_net = "cross-field: internet_service and the add-ons disagree"
    reasons = np.select(
        [phone_bad & net_bad, phone_bad, net_bad],
        [f"{msg_phone}; {msg_net}", msg_phone, msg_net],
        default="",
    )
    return pd.Series(reasons, index=df.index, dtype="object")


def enforce_cross_field(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the two cross-field rules so a synthetic row can never break them."""
    out = df.copy()
    no_phone = out["phone_service"] == "No"
    out.loc[no_phone, "multiple_lines"] = NO_PHONE
    out.loc[~no_phone & (out["multiple_lines"] == NO_PHONE), "multiple_lines"] = "No"
    no_net = out["internet_service"] == "No"
    for column in ADDON_COLUMNS:
        out.loc[no_net, column] = NO_INTERNET
        out.loc[~no_net & (out[column] == NO_INTERNET), column] = "No"
    return out


def field_catalogue() -> list[dict[str, object]]:
    """Describe all input fields as plain dictionaries (used by GET /api/v1/schema)."""
    catalogue: list[dict[str, object]] = []
    for spec in INPUT_FIELDS:
        catalogue.append(
            {
                "name": spec.name,
                "label": spec.name.replace("_", " ").capitalize(),
                "kind": spec.kind,
                "required": spec.required,
                "nullable": spec.nullable,
                "allowed": list(spec.allowed) if spec.allowed else None,
                "min": spec.min_value,
                "max": spec.max_value,
                "description": spec.description,
            }
        )
    return catalogue


def is_finite_frame(df: pd.DataFrame) -> bool:
    """True if no numeric cell of the frame is NaN or infinite."""
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
