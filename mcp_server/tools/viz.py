"""Visualization tool: render charts to PNG files under reports/.

Uses matplotlib's non-interactive Agg backend so it renders headless (server /
Streamlit). Returns the saved path plus a base64 data URI so clients can embed
the image without filesystem access.
"""

from __future__ import annotations

import base64
import os
import uuid
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mcp_server.paths import REPORTS_DIR  # noqa: E402
from mcp_server.registry import REGISTRY  # noqa: E402

CHART_KINDS = {"bar", "line", "scatter", "hist", "box", "heatmap"}


def register(mcp) -> None:
    """Attach the visualization tool to the given FastMCP instance."""

    @mcp.tool()
    def make_chart(
        dataset_id: str,
        kind: str,
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: Optional[str] = None,
        bins: int = 20,
    ) -> dict:
        """Render a chart from a dataset and save it as a PNG.

        Args:
            dataset_id: Dataset to plot.
            kind: bar, line, scatter, hist, box, or heatmap.
            x: X-axis column (category for bar; time/x for line/scatter;
                the column for hist/box). Not used by heatmap.
            y: Y-axis column (numeric). Optional for hist/box/heatmap.
            title: Chart title.
            bins: Bin count for histograms.

        Returns the saved file path and a base64 PNG data URI.
        """
        if kind not in CHART_KINDS:
            raise ValueError(f"kind must be one of {sorted(CHART_KINDS)}, got '{kind}'")
        df = REGISTRY.get(dataset_id).df

        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            if kind == "bar":
                data = df.groupby(x)[y].sum().sort_values(ascending=False) if y else df[x].value_counts()
                data.plot.bar(ax=ax)
                ax.set_ylabel(y or "count")
            elif kind == "line":
                d = df[[x, y]].copy()
                d[x] = pd.to_datetime(d[x], errors="ignore")
                d = d.groupby(x)[y].sum()
                d.plot.line(ax=ax, marker="o")
                ax.set_ylabel(y)
            elif kind == "scatter":
                ax.scatter(pd.to_numeric(df[x], errors="coerce"),
                           pd.to_numeric(df[y], errors="coerce"), alpha=0.7)
                ax.set_xlabel(x); ax.set_ylabel(y)
            elif kind == "hist":
                pd.to_numeric(df[x], errors="coerce").dropna().plot.hist(ax=ax, bins=bins)
                ax.set_xlabel(x)
            elif kind == "box":
                cols = [y] if y else [x]
                df[cols].plot.box(ax=ax)
            elif kind == "heatmap":
                numeric = df.select_dtypes(include="number")
                corr = numeric.corr()
                im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns)
                fig.colorbar(im, ax=ax)

            ax.set_title(title or f"{kind} chart")
            fig.tight_layout()

            fname = f"chart_{kind}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(REPORTS_DIR, fname)
            fig.savefig(path, dpi=110)
        finally:
            plt.close(fig)

        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")

        return {
            "dataset_id": dataset_id,
            "kind": kind,
            "path": path,
            "data_uri": f"data:image/png;base64,{b64}",
            "title": title or f"{kind} chart",
        }
