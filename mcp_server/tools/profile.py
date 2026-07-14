"""Profiling tool: full data profile for a registered dataset."""

from __future__ import annotations

import pandas as pd

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import _py, column_schema, sample_rows


def register(mcp) -> None:
    """Attach profiling tools to the given FastMCP instance."""

    @mcp.tool()
    def profile_dataset(dataset_id: str, top_k: int = 5) -> dict:
        """Produce a full profile of a loaded dataset.

        Includes per-column schema (dtype, nulls, cardinality), numeric summary
        statistics, top value counts for categorical columns, duplicate-row
        count, and a few sample rows. Use this before deciding on analysis steps.

        Args:
            dataset_id: The id returned by a load tool.
            top_k: How many top categorical values to report per column.
        """
        df = REGISTRY.get(dataset_id).df

        numeric = df.select_dtypes(include="number")
        numeric_summary = {}
        if not numeric.empty:
            desc = numeric.describe().to_dict()
            numeric_summary = {
                str(col): {k: _py(v) for k, v in stats.items()}
                for col, stats in desc.items()
            }

        categorical = df.select_dtypes(exclude="number")
        categorical_top = {}
        for col in categorical.columns:
            vc = df[col].value_counts(dropna=True).head(top_k)
            categorical_top[str(col)] = [
                {"value": _py(idx), "count": int(cnt)} for idx, cnt in vc.items()
            ]

        return {
            "dataset_id": dataset_id,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": column_schema(df),
            "numeric_summary": numeric_summary,
            "categorical_top": categorical_top,
            "sample_rows": sample_rows(df, 5),
        }
