"""Data cleaning tools.

All cleaning is **non-destructive**: each tool registers a *new* dataset derived
from the source (tracking ``parent_id``) and returns the new ``dataset_id`` plus
a summary of what changed. The original dataset is never mutated, so every
transformation is auditable and reversible.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import _py, schema_summary


def _derive(parent_id: str, df: pd.DataFrame, op: str, note: str) -> dict:
    parent = REGISTRY.get(parent_id)
    entry = REGISTRY.add(df, source=f"clean:{op}", parent_id=parent_id, note=note, prefix="clean")
    summary = schema_summary(df, entry.dataset_id, entry.source)
    summary["parent_id"] = parent_id
    summary["operation"] = op
    summary["change"] = note
    return summary


def register(mcp) -> None:
    """Attach cleaning tools to the given FastMCP instance."""

    @mcp.tool()
    def drop_duplicates(dataset_id: str, subset: Optional[list[str]] = None) -> dict:
        """Remove duplicate rows, returning a new dataset.

        Args:
            dataset_id: Source dataset.
            subset: Columns to consider for identifying duplicates (all if omitted).
        """
        df = REGISTRY.get(dataset_id).df
        before = len(df)
        out = df.drop_duplicates(subset=subset).reset_index(drop=True)
        return _derive(dataset_id, out, "drop_duplicates",
                       f"removed {before - len(out)} duplicate row(s)")

    @mcp.tool()
    def handle_missing(
        dataset_id: str,
        strategy: str = "drop",
        columns: Optional[list[str]] = None,
        fill_value: Optional[float] = None,
    ) -> dict:
        """Handle missing values, returning a new dataset.

        Args:
            dataset_id: Source dataset.
            strategy: "drop" (drop rows with NA), "mean", "median", "mode",
                or "constant" (uses fill_value).
            columns: Restrict to these columns (all if omitted).
            fill_value: Value used when strategy="constant".
        """
        df = REGISTRY.get(dataset_id).df.copy()
        cols = columns or list(df.columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not in {dataset_id}: {missing}")
        na_before = int(df[cols].isna().sum().sum())

        if strategy == "drop":
            out = df.dropna(subset=cols).reset_index(drop=True)
            note = f"dropped rows with NA in {cols}; {na_before} NA cells were present"
        elif strategy in {"mean", "median"}:
            out = df.copy()
            for c in cols:
                if pd.api.types.is_numeric_dtype(out[c]):
                    val = out[c].mean() if strategy == "mean" else out[c].median()
                    out[c] = out[c].fillna(val)
            note = f"filled numeric NA with column {strategy}"
        elif strategy == "mode":
            out = df.copy()
            for c in cols:
                mode = out[c].mode(dropna=True)
                if not mode.empty:
                    out[c] = out[c].fillna(mode.iloc[0])
            note = "filled NA with column mode"
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("strategy='constant' requires fill_value")
            out = df.copy()
            out[cols] = out[cols].fillna(fill_value)
            note = f"filled NA with constant {fill_value}"
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        return _derive(dataset_id, out, f"handle_missing:{strategy}", note)

    @mcp.tool()
    def cast_types(dataset_id: str, casts: dict) -> dict:
        """Cast columns to new dtypes, returning a new dataset.

        Args:
            dataset_id: Source dataset.
            casts: Mapping of column -> dtype, e.g. {"date": "datetime", "units": "int"}.
                Supported dtypes: int, float, str, bool, datetime, category.
        """
        df = REGISTRY.get(dataset_id).df.copy()
        applied = {}
        for col, dtype in casts.items():
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in {dataset_id}")
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
            applied[col] = dtype
        return _derive(dataset_id, df, "cast_types", f"cast {applied}")

    @mcp.tool()
    def handle_outliers(dataset_id: str, column: str, method: str = "iqr",
                        action: str = "clip") -> dict:
        """Detect and handle outliers in a numeric column, returning a new dataset.

        Args:
            dataset_id: Source dataset.
            column: Numeric column to treat.
            method: "iqr" (1.5*IQR fences) or "zscore" (|z|>3).
            action: "clip" (cap at the fences) or "remove" (drop outlier rows).
        """
        df = REGISTRY.get(dataset_id).df.copy()
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not in {dataset_id}")
        s = pd.to_numeric(df[column], errors="coerce")

        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif method == "zscore":
            mean, std = s.mean(), s.std()
            low, high = mean - 3 * std, mean + 3 * std
        else:
            raise ValueError(f"Unknown method '{method}'")

        mask = (s < low) | (s > high)
        n_out = int(mask.sum())
        if action == "clip":
            df[column] = s.clip(low, high)
            note = f"clipped {n_out} outlier(s) in '{column}' to [{_py(round(low,3))}, {_py(round(high,3))}]"
        elif action == "remove":
            df = df[~mask].reset_index(drop=True)
            note = f"removed {n_out} outlier row(s) by '{column}'"
        else:
            raise ValueError(f"Unknown action '{action}'")

        return _derive(dataset_id, df, f"handle_outliers:{method}:{action}", note)

    @mcp.tool()
    def rename_columns(dataset_id: str, mapping: dict) -> dict:
        """Rename columns, returning a new dataset.

        Args:
            dataset_id: Source dataset.
            mapping: {old_name: new_name} pairs.
        """
        df = REGISTRY.get(dataset_id).df.copy()
        missing = [c for c in mapping if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not in {dataset_id}: {missing}")
        out = df.rename(columns=mapping)
        return _derive(dataset_id, out, "rename_columns", f"renamed {mapping}")
