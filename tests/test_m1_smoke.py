"""End-to-end M1 smoke test.

Spawns the real MCP server over stdio (exactly how the agent host / Claude
Desktop would), lists its tools, then loads and profiles the sample CSV.
"""

import json
import os
import sys

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO, "data", "sample_sales.csv")


def _content_json(result):
    """Extract the JSON payload from a tool result's text content."""
    text = result.content[0].text
    return json.loads(text)


@pytest.mark.asyncio
async def test_load_and_profile():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = {t.name for t in (await session.list_tools()).tools}
            assert {"load_csv", "load_excel", "list_datasets", "profile_dataset"} <= tools

            loaded = _content_json(await session.call_tool("load_csv", {"path": SAMPLE_CSV}))
            assert loaded["rows"] == 12 and loaded["cols"] == 8
            ds_id = loaded["dataset_id"]

            prof = _content_json(await session.call_tool("profile_dataset", {"dataset_id": ds_id}))
            assert prof["duplicate_rows"] == 1               # row 1007 is duplicated
            age = next(c for c in prof["columns"] if c["name"] == "customer_age")
            assert age["nulls"] == 2                          # two blank ages
            assert "units" in prof["numeric_summary"]
            assert prof["categorical_top"]["region"]         # non-empty top values
