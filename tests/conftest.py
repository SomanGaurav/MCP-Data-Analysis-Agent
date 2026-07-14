"""Shared test fixtures: a live MCP client session over stdio."""

import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO, "data", "sample_sales.csv")


def content_json(result):
    """Extract and parse the JSON payload from a tool result."""
    return json.loads(result.content[0].text)


@asynccontextmanager
async def server_session():
    """Live MCP client session over stdio.

    Used as ``async with server_session() as session:`` directly inside each
    test so the anyio cancel scopes enter and exit in the same task (a
    yield-fixture would tear down in a different task and error).
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
