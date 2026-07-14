"""Exploratory data analysis tools: value counts, correlations, group aggregates."""

from __future__ import annotations

from typing import Optional

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import _py

AGG_FUNCS = {"sum", "mean", "median", "min", "max", "count", "std", "var", "nunique"}


def register(mcp) -> None:
    """Attach EDA tools to the given FastMCP instance."""

    @mcp.tool()
    def value_counts(dataset_id: str, column: str, top_k: int = 20) -> dict:
        """Frequency of each value in a column, most common first.

        Args:
            dataset_id: The dataset to inspect.
            column: Column name to count.
            top_k: Number of top values to return.
        """
        df = REGISTRY.get(dataset_id).df
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not in {dataset_id}. Have: {list(df.columns)}")
        vc = df[column].value_counts(dropna=False).head(top_k)
        return {
            "dataset_id": dataset_id,
            "column": column,
            "counts": [{"value": _py(idx), "count": int(cnt)} for idx, cnt in vc.items()],
        }

    @mcp.tool()
    def correlations(dataset_id: str, method: str = "pearson") -> dict:
        """Pairwise correlation matrix over numeric columns.

        Args:
            dataset_id: The dataset to analyze.
            method: "pearson", "spearman", or "kendall".
        """
        df = REGISTRY.get(dataset_id).df
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return {"dataset_id": dataset_id, "method": method, "matrix": {},
                    "note": "Need at least 2 numeric columns for correlation."}
        corr = numeric.corr(method=method)
        matrix = {
            str(r): {str(c): _py(round(corr.loc[r, c], 4)) for c in corr.columns}
            for r in corr.index
        }
        # Flag the strongest off-diagonal pairs to guide the agent.
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
        return {
            "dataset_id": dataset_id,
            "method": method,
            "matrix": matrix,
            "strongest_pairs": [
                {"a": str(a), "b": str(b), "corr": round(v, 4)} for a, b, v in pairs[:5]
            ],
        }

    @mcp.tool()
    def groupby_aggregate(
        dataset_id: str,
        group_by: list[str],
        metric: str,
        agg: str = "sum",
        top_k: int = 50,
    ) -> dict:
        """Group rows and aggregate a metric column.

        Args:
            dataset_id: The dataset to analyze.
            group_by: One or more columns to group by.
            metric: Numeric column to aggregate.
            agg: One of sum, mean, median, min, max, count, std, var, nunique.
            top_k: Max number of result rows (sorted by the aggregate, descending).
        """
        if agg not in AGG_FUNCS:
            raise ValueError(f"agg must be one of {sorted(AGG_FUNCS)}, got '{agg}'")
        df = REGISTRY.get(dataset_id).df
        missing = [c for c in [*group_by, metric] if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not in {dataset_id}: {missing}. Have: {list(df.columns)}")
        grouped = df.groupby(group_by, dropna=False)[metric].agg(agg).reset_index()
        grouped = grouped.sort_values(metric, ascending=False).head(top_k)
        return {
            "dataset_id": dataset_id,
            "group_by": group_by,
            "metric": metric,
            "agg": agg,
            "rows": [
                {str(k): _py(v) for k, v in row.items()}
                for row in grouped.to_dict(orient="records")
            ],
        }
