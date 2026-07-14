"""MCP client: connect to the analysis server, expose its tools to the LLM.

Handles the two translations the agent needs:
  1. MCP tool schemas  -> OpenAI-style ``tools`` (so any LiteLLM provider can call them).
  2. An LLM tool-call  -> an MCP ``call_tool`` and back to a JSON string result.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ToolSession:
    """Thin wrapper over an initialized MCP ClientSession."""

    def __init__(self, session: ClientSession):
        self._session = session
        self._openai_tools: list[dict] | None = None

    async def openai_tools(self) -> list[dict]:
        """List server tools as OpenAI-style function schemas (cached)."""
        if self._openai_tools is None:
            listed = await self._session.list_tools()
            self._openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": t.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in listed.tools
            ]
        return self._openai_tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return its result as a JSON/text string.

        Errors are returned as a JSON ``{"error": ...}`` string rather than
        raised, so the LLM can read the failure and adjust its next step.
        """
        try:
            result = await self._session.call_tool(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash the loop
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

        if getattr(result, "isError", False):
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return json.dumps({"error": " ".join(texts) or "tool error"})

        texts = [c.text for c in result.content if hasattr(c, "text")]
        return texts[0] if len(texts) == 1 else json.dumps(texts)


@asynccontextmanager
async def connect():
    """Spawn the analysis server over stdio and yield a ready ToolSession."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield ToolSession(session)
