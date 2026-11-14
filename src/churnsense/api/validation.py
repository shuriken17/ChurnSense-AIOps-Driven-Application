"""Validation of one customer record, with messages a person can act on.

The rules come from ``schema.INPUT_FIELDS`` (the same single source of truth that the data
pipeline and the model use), so the API can never accept something the model was not built for.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from churnsense.schema import INPUT_FIELDS, FieldSpec, cross_field_violations

MAX_ID_LENGTH = 64


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value == value


def _check_field(spec: FieldSpec, value: Any) -> tuple[Any, str | None]:
    """Return ``(clean value, error message)`` for one field."""
    if value is None:
        if spec.nullable:
            return None, None
        return value, "is required"
    if spec.kind == "id":
        if not isinstance(value, str) or len(value) > MAX_ID_LENGTH:
            return value, f"must be text of at most {MAX_ID_LENGTH} characters"
        return value, None
    if spec.kind == "category":
        if value not in (spec.allowed or ()):
            return value, "must be one of: " + ", ".join(spec.allowed or ())
        return value, None
    if not _is_number(value):
        return value, "must be a number"
    if spec.kind in ("binary", "int"):
        if float(value) != int(value):
            return value, "must be a whole number"
        value = int(value)
    if spec.kind == "binary" and value not in (0, 1):
        return value, "must be 0 or 1"
    low, high = spec.min_value, spec.max_value
    if low is not None and (value < low or (spec.min_exclusive and value == low)):
        return value, f"must be {'above' if spec.min_exclusive else 'at least'} {low:g}"
    if high is not None and value > high:
        return value, f"must be at most {high:g}"
    return value, None


def validate_customer(payload: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Check one customer record.

    Returns:
        The cleaned record (whole numbers as ints) and a list of ``{"field", "message"}``
        problems; the list is empty when the record is acceptable.
    """
    if not isinstance(payload, dict):
        return {}, [{"field": "(body)", "message": "must be a JSON object of customer fields"}]
    known = {spec.name: spec for spec in INPUT_FIELDS}
    problems = [
        {"field": name, "message": "is not a known field"} for name in payload if name not in known
    ]
    clean: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in payload:
            if spec.required:
                problems.append({"field": name, "message": "is required"})
            continue
        value, message = _check_field(spec, payload[name])
        if message:
            problems.append({"field": name, "message": message})
        else:
            clean[name] = value
    if not problems:  # relationships between fields, checked only on otherwise valid data
        frame = pd.DataFrame([clean])
        reason = str(cross_field_violations(frame).iloc[0])
        if reason:
            problems.append({"field": "(combination)", "message": reason})
    return clean, problems
