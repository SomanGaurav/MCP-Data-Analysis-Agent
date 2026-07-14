"""The autonomous analysis agent loop.

Provider-agnostic: it takes any object with a ``.complete(messages, tools)``
method (our LLMClient, or a fake in tests). The loop is plan -> call MCP tool ->
observe -> repeat, until the model answers in natural language. Every tool call
is recorded as an audit trail and mined for charts/report paths the UI shows.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.mcp_client import ToolSession, connect
from agent.providers import LLMClient

SYSTEM_PROMPT = """You are an autonomous data analyst. You answer questions about \
datasets by calling tools — you must NEVER do arithmetic, statistics, or aggregation \
yourself. The tools compute exact results; your job is to plan which tools to call \
and then interpret their outputs into clear business insight.

Guidelines:
- If the user gives a file path and no dataset is loaded yet, call load_csv/load_excel first.
- Call profile_dataset early to learn columns, dtypes, and data quality.
- Use run_sql, groupby_aggregate, correlations, and the stats tools for analysis.
- Use make_chart to visualize key findings; call export_report if the user wants a report.
- Reference datasets by their dataset_id. Cleaning tools return NEW dataset_ids.
- When you have enough evidence, stop calling tools and give a concise, business-focused \
answer with concrete numbers from the tool results and actionable recommendations."""

MAX_STEPS = 12


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict
    result: Any


@dataclass
class AgentResult:
    answer: str
    provider: str
    model: str
    steps: int
    trace: list[ToolCallRecord] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)      # {title, data_uri, path}
    report_path: Optional[str] = None
    stopped_reason: str = "answered"


def _msg_to_dict(msg: Any) -> dict:
    """Normalize a LiteLLM message object into a plain OpenAI-style dict."""
    if hasattr(msg, "model_dump"):
        d = msg.model_dump()
    elif isinstance(msg, dict):
        d = dict(msg)
    else:  # pragma: no cover - defensive
        d = {"role": "assistant", "content": str(msg)}
    return {k: v for k, v in d.items() if v is not None}


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


class DataAnalysisAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run_async(
        self,
        question: str,
        session: ToolSession,
        preload_context: str = "",
        on_event: Optional[Callable[[str, dict], None]] = None,
        max_steps: int = MAX_STEPS,
    ) -> AgentResult:
        tools = await session.openai_tools()
        user_content = (preload_context + "\n\n" if preload_context else "") + question
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        result = AgentResult(answer="", provider=self.llm.provider, model=self.llm.model, steps=0)

        for step in range(1, max_steps + 1):
            result.steps = step
            response = self.llm.complete(messages, tools=tools)
            msg = response.choices[0].message
            messages.append(_msg_to_dict(msg))

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                result.answer = msg.content or ""
                return result

            for tc in tool_calls:
                name = tc.function.name
                args = _parse_json(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
                if on_event:
                    on_event("tool_call", {"name": name, "arguments": args})

                raw = await session.call_tool(name, args)
                parsed = _parse_json(raw)
                result.trace.append(ToolCallRecord(step, name, args, parsed))
                self._collect_artifacts(name, parsed, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": raw,
                })

        result.stopped_reason = "max_steps"
        result.answer = (
            result.answer
            or "Reached the step limit before finishing. Partial results are in the tool trace."
        )
        return result

    @staticmethod
    def _collect_artifacts(name: str, parsed: Any, result: AgentResult) -> None:
        if not isinstance(parsed, dict):
            return
        if name == "make_chart" and parsed.get("data_uri"):
            result.charts.append({
                "title": parsed.get("title", "chart"),
                "data_uri": parsed["data_uri"],
                "path": parsed.get("path"),
            })
        elif name == "export_report" and parsed.get("path"):
            result.report_path = parsed["path"]

    def run(self, question: str, preload_context: str = "",
            on_event: Optional[Callable[[str, dict], None]] = None) -> AgentResult:
        """Synchronous entrypoint for the UI: opens a server session and runs to completion."""
        async def _go() -> AgentResult:
            async with connect() as session:
                return await self.run_async(question, session,
                                            preload_context=preload_context, on_event=on_event)
        return asyncio.run(_go())
