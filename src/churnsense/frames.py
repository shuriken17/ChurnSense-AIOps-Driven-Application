"""Helpers for DataFrame column types.

Ray Data (and some parquet files) return Arrow-backed pandas columns such as
``double[pyarrow]`` or ``large_string[pyarrow]``. Some pandas operations are not implemented
for them (for example ``%``), and they make results depend on which path produced the frame.
``to_native_dtypes`` converts them to ordinary pandas types, so the Ray path and the plain
pandas path always give identical frames.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa


def _native_column(series: pd.Series) -> pd.Series:
    """Convert one Arrow-backed column to a native pandas column (others are untouched)."""
    dtype = series.dtype
    if not isinstance(dtype, pd.ArrowDtype):
        return series
    arrow_type = dtype.pyarrow_dtype
    if pa.types.is_floating(arrow_type):
        return series.astype("float64")
    if pa.types.is_integer(arrow_type):
        return series.astype("float64" if series.isna().any() else "int64")
    if pa.types.is_boolean(arrow_type):
        return series.astype("object" if series.isna().any() else "bool")
    if pa.types.is_timestamp(arrow_type):
        return pd.to_datetime(series.astype("object"), utc=True).astype("datetime64[us, UTC]")
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return series.astype("object")
    return series.astype("object")


def to_native_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` in which every Arrow-backed column has a native pandas type."""
    return pd.DataFrame(
        {column: _native_column(df[column]) for column in df.columns}, index=df.index
    )
