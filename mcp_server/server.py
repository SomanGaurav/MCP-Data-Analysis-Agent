"""FastMCP analysis server entrypoint.

Runs over stdio so any MCP client (our agent host, or Claude Desktop) can spawn
it and discover its tools. Later milestones register more tool modules here.

Run directly:  python -m mcp_server.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import clean, eda, load, profile, report, sql, stats, viz

mcp = FastMCP("data-analysis")

# --- Register tool modules (grows each milestone) ---
load.register(mcp)        # M1: load_csv, load_excel, list_datasets
profile.register(mcp)     # M1: profile_dataset
sql.register(mcp)         # M2: run_sql (DuckDB)
eda.register(mcp)         # M2: value_counts, correlations, groupby_aggregate
clean.register(mcp)       # M2: drop_duplicates, handle_missing, cast_types, handle_outliers, rename_columns
stats.register(mcp)       # M3: hypothesis_test, correlation_test, trend_analysis, distribution_fit
viz.register(mcp)         # M3: make_chart
report.register(mcp)      # M3: export_report


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
