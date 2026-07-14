"""M5 test: the agent loop wiring, driven by a scripted fake LLM.

No live model is used — a fake returns a fixed plan (load -> aggregate -> chart
-> answer) so we can assert the loop calls real MCP tools, threads results back,
and collects artifacts deterministically. A fresh server makes the first load
return dataset id ``ds_001``.
"""

import json
from types import SimpleNamespace

from agent.agent import DataAnalysisAgent
from agent.mcp_client import connect
from tests.conftest import SAMPLE_CSV


def _tool_call(cid, name, args):
    return SimpleNamespace(
        id=cid,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}


class _Resp:
    def __init__(self, msg):
        self.choices = [SimpleNamespace(message=msg)]


class FakeLLM:
    """Replays a fixed sequence of assistant turns regardless of input."""

    provider = "fake"
    model = "fake/scripted"

    def __init__(self, script):
        self._script = script
        self._i = 0

    def complete(self, messages, tools=None, **kw):
        msg = self._script[self._i]
        self._i += 1
        return _Resp(msg)


async def test_agent_orchestration():
    script = [
        _Msg(tool_calls=[_tool_call("c1", "load_csv", {"path": SAMPLE_CSV})]),
        _Msg(tool_calls=[_tool_call("c2", "groupby_aggregate",
                                    {"dataset_id": "ds_001", "group_by": ["region"],
                                     "metric": "units", "agg": "sum"})]),
        _Msg(tool_calls=[_tool_call("c3", "make_chart",
                                    {"dataset_id": "ds_001", "kind": "bar",
                                     "x": "region", "y": "units", "title": "Units by region"})]),
        _Msg(content="North leads in total units sold; focus inventory there."),
    ]
    agent = DataAnalysisAgent(FakeLLM(script))
    events = []

    async with connect() as session:
        result = await agent.run_async(
            "Which region sells the most units?", session,
            on_event=lambda kind, data: events.append((kind, data["name"])),
        )

    assert result.stopped_reason == "answered"
    assert "North" in result.answer
    # Three tools were executed in order.
    assert [r.name for r in result.trace] == ["load_csv", "groupby_aggregate", "make_chart"]
    # The aggregate result was really computed by the tool, not the LLM.
    agg = result.trace[1].result
    assert agg["rows"][0]["region"] == "North"
    # The chart artifact was collected for the UI.
    assert len(result.charts) == 1 and result.charts[0]["data_uri"].startswith("data:image/png")
    assert events[0] == ("tool_call", "load_csv")
