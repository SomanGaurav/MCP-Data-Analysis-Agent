"""Shared helpers for turning a DataFrame into compact, LLM-friendly summaries.

Everything here returns plain JSON-serializable dicts/lists (no numpy scalars),
so tool results stay small and safe to hand to any provider.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _py(value: Any) -> Any:
    """Coerce numpy/pandas scalars to native Python types for JSON safety."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def column_schema(df: pd.DataFrame) -> list[dict]:
    """Per-column dtype, null count, and cardinality — the schema the LLM plans against."""
    n = len(df)
    schema = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        schema.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "nulls": nulls,
                "null_pct": round(100 * nulls / n, 2) if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
            }
        )
    return schema


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """A few example rows as JSON-safe dicts."""
    head = df.head(n).where(pd.notna(df.head(n)), None)
    return [
        {str(k): _py(v) for k, v in row.items()}
        for row in head.to_dict(orient="records")
    ]


def schema_summary(df: pd.DataFrame, dataset_id: str, source: str) -> dict:
    """Compact overview returned on load — shape, columns, sample."""
    return {
        "dataset_id": dataset_id,
        "source": source,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": column_schema(df),
        "sample_rows": sample_rows(df, 5),
    }
