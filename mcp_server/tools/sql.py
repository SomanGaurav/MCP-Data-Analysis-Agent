"""SQL tool: run DuckDB queries directly over registered DataFrames.

Every loaded dataset is exposed to DuckDB as a view named by its ``dataset_id``
(e.g. ``SELECT * FROM ds_001``), so the LLM can write real SQL against CSV/Excel
data without a database server. Optionally the result is registered as a new
dataset for further analysis.
"""

from __future__ import annotations

from typing import Optional

import duckdb

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import _py

MAX_PREVIEW_ROWS = 100


def register(mcp) -> None:
    """Attach the SQL tool to the given FastMCP instance."""

    @mcp.tool()
    def run_sql(query: str, register_as_dataset: bool = False) -> dict:
        """Run a read-only SQL query (DuckDB dialect) over loaded datasets.

        Reference datasets by their ``dataset_id`` as table names, e.g.
        ``SELECT region, SUM(units) AS total FROM ds_001 GROUP BY region``.

        Args:
            query: A SELECT query in DuckDB SQL.
            register_as_dataset: If true, store the full result as a new dataset
                and return its ``dataset_id`` (use for large results you want to
                analyze further). If false, only a row preview is returned.

        Returns column names, a capped row preview, and total row count.
        """
        con = duckdb.connect()
        try:
            for entry in REGISTRY._entries.values():
                con.register(entry.dataset_id, entry.df)
            result_df = con.execute(query).fetch_df()
        finally:
            con.close()

        out: dict = {
            "columns": [str(c) for c in result_df.columns],
            "row_count": int(result_df.shape[0]),
            "rows": [
                {str(k): _py(v) for k, v in row.items()}
                for row in result_df.head(MAX_PREVIEW_ROWS).to_dict(orient="records")
            ],
            "truncated": bool(result_df.shape[0] > MAX_PREVIEW_ROWS),
        }
        if register_as_dataset:
            entry = REGISTRY.add(result_df, source=f"sql:{query[:60]}", prefix="sql")
            out["dataset_id"] = entry.dataset_id
        return out
