"""Statistical analysis tools: hypothesis tests, correlation tests, trends."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import _py


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    return (
        f"p={p:.4g} < {alpha}: statistically significant."
        if p < alpha
        else f"p={p:.4g} >= {alpha}: not statistically significant."
    )


def register(mcp) -> None:
    """Attach statistics tools to the given FastMCP instance."""

    @mcp.tool()
    def hypothesis_test(
        dataset_id: str,
        test: str,
        value_column: str,
        group_column: Optional[str] = None,
        alpha: float = 0.05,
    ) -> dict:
        """Run a hypothesis test and interpret the result.

        Args:
            dataset_id: Dataset to test.
            test: "ttest" (compare a numeric value_column across the two groups
                in group_column), "anova" (across 3+ groups), or "chi2"
                (independence of value_column vs group_column, both categorical).
            value_column: The measured column (numeric for ttest/anova).
            group_column: The grouping column (required for all three tests).
            alpha: Significance threshold.
        """
        df = REGISTRY.get(dataset_id).df
        for c in [value_column, group_column]:
            if c and c not in df.columns:
                raise KeyError(f"Column '{c}' not in {dataset_id}")

        if test == "ttest":
            groups = df.dropna(subset=[value_column, group_column]).groupby(group_column)[value_column]
            samples = [g.values for _, g in groups]
            if len(samples) != 2:
                raise ValueError(f"ttest needs exactly 2 groups, found {len(samples)}")
            stat, p = sps.ttest_ind(samples[0], samples[1], equal_var=False)
        elif test == "anova":
            groups = df.dropna(subset=[value_column, group_column]).groupby(group_column)[value_column]
            samples = [g.values for _, g in groups]
            if len(samples) < 3:
                raise ValueError(f"anova expects 3+ groups, found {len(samples)}")
            stat, p = sps.f_oneway(*samples)
        elif test == "chi2":
            table = pd.crosstab(df[value_column], df[group_column])
            stat, p, _, _ = sps.chi2_contingency(table)
        else:
            raise ValueError(f"Unknown test '{test}' (use ttest, anova, chi2)")

        return {
            "dataset_id": dataset_id,
            "test": test,
            "statistic": _py(round(float(stat), 4)),
            "p_value": _py(round(float(p), 6)),
            "alpha": alpha,
            "significant": bool(p < alpha),
            "interpretation": _interpret_p(float(p), alpha),
        }

    @mcp.tool()
    def correlation_test(dataset_id: str, x: str, y: str, method: str = "pearson") -> dict:
        """Test whether two numeric columns are correlated, with a p-value.

        Args:
            dataset_id: Dataset to test.
            x, y: Numeric columns.
            method: "pearson" or "spearman".
        """
        df = REGISTRY.get(dataset_id).df[[x, y]].dropna()
        a = pd.to_numeric(df[x], errors="coerce")
        b = pd.to_numeric(df[y], errors="coerce")
        valid = a.notna() & b.notna()
        a, b = a[valid], b[valid]
        if method == "pearson":
            r, p = sps.pearsonr(a, b)
        elif method == "spearman":
            r, p = sps.spearmanr(a, b)
        else:
            raise ValueError(f"Unknown method '{method}'")
        return {
            "dataset_id": dataset_id,
            "x": x, "y": y, "method": method,
            "correlation": _py(round(float(r), 4)),
            "p_value": _py(round(float(p), 6)),
            "n": int(len(a)),
            "interpretation": _interpret_p(float(p)),
        }

    @mcp.tool()
    def trend_analysis(dataset_id: str, time_column: str, value_column: str) -> dict:
        """Fit a linear trend of a value over time (ordinal regression).

        Returns slope, direction, R², and significance — use for "is X going
        up or down over time" questions.

        Args:
            dataset_id: Dataset to analyze.
            time_column: A date/datetime or ordered numeric column.
            value_column: The numeric metric to trend.
        """
        df = REGISTRY.get(dataset_id).df[[time_column, value_column]].copy()
        t = pd.to_datetime(df[time_column], errors="coerce")
        if t.notna().any():
            x = t.map(pd.Timestamp.toordinal)
        else:
            x = pd.to_numeric(df[time_column], errors="coerce")
        y = pd.to_numeric(df[value_column], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        # Aggregate duplicate time points so the trend reflects per-period totals.
        agg = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].sum()
        res = sps.linregress(agg["x"], agg["y"])
        return {
            "dataset_id": dataset_id,
            "time_column": time_column,
            "value_column": value_column,
            "slope": _py(round(float(res.slope), 6)),
            "direction": "increasing" if res.slope > 0 else "decreasing" if res.slope < 0 else "flat",
            "r_squared": _py(round(float(res.rvalue) ** 2, 4)),
            "p_value": _py(round(float(res.pvalue), 6)),
            "n_periods": int(len(agg)),
            "interpretation": _interpret_p(float(res.pvalue)),
        }

    @mcp.tool()
    def distribution_fit(dataset_id: str, column: str) -> dict:
        """Summarize a numeric column's distribution and test for normality.

        Reports skewness, kurtosis, and a Shapiro-Wilk normality test.

        Args:
            dataset_id: Dataset to analyze.
            column: Numeric column.
        """
        s = pd.to_numeric(REGISTRY.get(dataset_id).df[column], errors="coerce").dropna()
        result = {
            "dataset_id": dataset_id,
            "column": column,
            "n": int(len(s)),
            "mean": _py(round(float(s.mean()), 4)),
            "std": _py(round(float(s.std()), 4)),
            "skewness": _py(round(float(sps.skew(s)), 4)),
            "kurtosis": _py(round(float(sps.kurtosis(s)), 4)),
        }
        if 3 <= len(s) <= 5000:
            w, p = sps.shapiro(s)
            result["normality_p"] = _py(round(float(p), 6))
            result["is_normal"] = bool(p >= 0.05)
        else:
            result["normality_p"] = None
            result["is_normal"] = None
        return result
