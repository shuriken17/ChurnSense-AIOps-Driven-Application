"""Turn raw input (IBM headers, mixed types) into the canonical, typed layout.

This is a pure function so that Ray can run it on many chunks in parallel
(``Dataset.map_batches``) and the result is the same as running it on the whole file.
"""

from __future__ import annotations

import pandas as pd

from churnsense.schema import (
    CATEGORICAL_INPUTS,
    ID_COLUMN,
    INGESTED_COLUMNS,
    NOMINAL_RAW_EVENT_TS,
    RAW_BATCH_ID,
    SchemaError,
    rename_to_canonical,
)

_BINARY_COLUMNS = ("senior_citizen", "churn")
_NUMERIC_COLUMNS = ("tenure_months", "monthly_charges", "total_charges")
_TEXT_COLUMNS = (ID_COLUMN, *CATEGORICAL_INPUTS)
_YES_NO = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def _to_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No/true/false/1/0 to 1/0. Anything else becomes NaN (validation reports it)."""

    def convert(value: object) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return float("nan")
        return float(_YES_NO.get(str(value).strip().lower(), float("nan")))

    return series.map(convert).astype(float)


def standardise_frame(
    df: pd.DataFrame,
    *,
    source: str,
    event_ts: str | pd.Timestamp | None = None,
    batch_id: str = RAW_BATCH_ID,
) -> pd.DataFrame:
    """Return ``df`` with canonical names and types plus lineage columns.

    * headers: IBM spelling or snake_case are both renamed to canonical names;
    * numbers: converted with ``errors="coerce"`` so a blank ``TotalCharges`` becomes NaN;
    * binary fields: "Yes"/"No" become 1/0;
    * text fields: surrounding whitespace removed;
    * lineage: ``source``, ``event_ts`` and ``batch_id`` are added if absent.

    Unknown columns are kept, so that the validation stage can report schema drift.

    Raises:
        SchemaError: if two input headers map to the same field.
    """
    out = rename_to_canonical(df.copy())

    for column in _TEXT_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("object").map(_strip)
    for column in _NUMERIC_COLUMNS:
        if column in out.columns:
            # A blank string (the IBM file has " " for 11 new customers) becomes NaN.
            out[column] = pd.to_numeric(
                out[column].astype("object").map(_strip), errors="coerce"
            ).astype(float)
    for column in _BINARY_COLUMNS:
        if column in out.columns:
            out[column] = _to_binary(out[column])

    # Lineage columns, added in the canonical order: source, event_ts, batch_id.
    if "source" not in out.columns:
        out["source"] = source
    if "event_ts" not in out.columns:
        stamp = pd.Timestamp(event_ts) if event_ts not in (None, "") else NOMINAL_RAW_EVENT_TS
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        out["event_ts"] = stamp
    out["event_ts"] = pd.to_datetime(out["event_ts"], utc=True).astype("datetime64[us, UTC]")
    if "batch_id" not in out.columns:
        out["batch_id"] = batch_id
    return out


def standardise_batch(
    batch: pd.DataFrame, source: str, event_ts: str, batch_id: str
) -> pd.DataFrame:
    """Ray ``map_batches`` adapter for ``standardise_frame`` (arguments must be picklable)."""
    return standardise_frame(batch, source=source, event_ts=event_ts, batch_id=batch_id)


def require_ingest_columns(df: pd.DataFrame) -> None:
    """Fail loudly if the frame lacks any column the ingest output must have."""
    missing = [c for c in INGESTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns after standardising: {missing}")
