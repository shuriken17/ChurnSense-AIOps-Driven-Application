"""Row validation with pandera.

The pandera schema is GENERATED from ``schema.py``, so the allowed values, ranges and
nullability are defined once. Validation is "lazy": every problem of every row is
collected, so each rejected row can be saved together with the reasons it was rejected.

Two levels of problem:

* schema drift (a column is missing or unexpected) is not fixable row by row, so it raises
  ``SchemaDriftError`` and the stage fails;
* a bad row (unknown category, out-of-range number, broken cross-field rule) is moved to
  the quarantine with its reason; the stage decides whether too many rows were rejected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from churnsense.frames import to_native_dtypes
from churnsense.schema import (
    INGESTED_COLUMNS,
    INPUT_FIELDS,
    SOURCES,
    TARGET_FIELD,
    VALIDATED_COLUMNS,
    FieldSpec,
    SchemaError,
    cross_field_violations,
)

INGESTED = "ingested"
VALIDATED = "validated"


class SchemaDriftError(SchemaError):
    """The columns of the data differ from the canonical schema."""


@dataclass
class ValidationOutcome:
    """Result of validating a frame: good rows, and bad rows with a ``reason`` column."""

    valid: pd.DataFrame
    rejected: pd.DataFrame


def expected_columns(stage: str) -> tuple[str, ...]:
    """Return the exact column set the data must have at the given stage."""
    if stage == INGESTED:
        return INGESTED_COLUMNS
    if stage == VALIDATED:
        return VALIDATED_COLUMNS
    raise ValueError(f"unknown stage {stage!r}; use {INGESTED!r} or {VALIDATED!r}")


def check_columns(df: pd.DataFrame, stage: str) -> None:
    """Raise ``SchemaDriftError`` if the column set differs from the schema."""
    expected = set(expected_columns(stage))
    got = set(df.columns)
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        raise SchemaDriftError(f"column mismatch: missing={missing} unexpected={extra}")


def _column_for(spec: FieldSpec, *, allow_missing_total: bool) -> pa.Column:
    """Build the pandera column for one field spec."""
    nullable = spec.nullable and allow_missing_total
    checks: list[pa.Check] = []
    if spec.kind == "category":
        checks.append(pa.Check.isin(list(spec.allowed or ())))
    elif spec.kind == "binary":
        checks.append(pa.Check.isin([0, 1]))
    elif spec.kind == "int":
        checks.append(pa.Check.in_range(spec.min_value, spec.max_value))
        checks.append(pa.Check(lambda s: s % 1 == 0, name="whole_number"))
    elif spec.kind == "float":
        checks.append(
            pa.Check.in_range(spec.min_value, spec.max_value, include_min=not spec.min_exclusive)
        )
    elif spec.kind == "id":
        checks.append(pa.Check.str_length(min_value=1))
    return pa.Column(None, checks=checks, nullable=nullable, unique=spec.kind == "id")


def build_schema(stage: str) -> pa.DataFrameSchema:
    """Generate the pandera schema for a stage ("ingested" or "validated")."""
    expected_columns(stage)  # validates the stage name
    columns: dict[str, pa.Column] = {
        spec.name: _column_for(spec, allow_missing_total=stage == INGESTED) for spec in INPUT_FIELDS
    }
    columns[TARGET_FIELD.name] = _column_for(TARGET_FIELD, allow_missing_total=False)
    columns["source"] = pa.Column(None, [pa.Check.isin(list(SOURCES))], nullable=False)
    columns["event_ts"] = pa.Column(None, nullable=False)
    columns["batch_id"] = pa.Column(None, [pa.Check.str_length(min_value=1)], nullable=False)
    if stage == VALIDATED:
        columns["total_charges_imputed"] = pa.Column(None, [pa.Check.isin([0, 1])], nullable=False)
    return pa.DataFrameSchema(columns, strict=True, ordered=False)


def validate_rows(df: pd.DataFrame, stage: str = VALIDATED) -> ValidationOutcome:
    """Split a frame into valid rows and rejected rows (with reasons).

    Raises:
        SchemaDriftError: if columns are missing or unexpected, or a rule fails for the
            frame as a whole (not for one row).
    """
    check_columns(df, stage)
    frame = to_native_dtypes(df.reset_index(drop=True))
    reasons: dict[int, list[str]] = defaultdict(list)

    try:
        build_schema(stage).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases
        frame_level = failures[failures["index"].isna()]
        if len(frame_level):
            raise SchemaDriftError(
                f"frame-level schema failure: {frame_level['check'].astype(str).unique().tolist()}"
            ) from exc
        for index, column, check, value in zip(
            failures["index"],
            failures["column"],
            failures["check"],
            failures["failure_case"],
            strict=True,
        ):
            reasons[int(index)].append(f"{column}: {check} (got {value!r})")

    cross = cross_field_violations(frame)
    for index in cross.index[cross != ""]:
        reasons[int(index)].append(str(cross[index]))

    bad_index = sorted(reasons)
    rejected = frame.loc[bad_index].copy()
    rejected["reason"] = ["; ".join(reasons[i]) for i in bad_index]
    valid = frame.drop(index=bad_index).reset_index(drop=True)
    return ValidationOutcome(valid=valid, rejected=rejected.reset_index(drop=True))
