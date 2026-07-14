"""M3 tests: statistics, visualization, and HTML report generation."""

import os

from tests.conftest import SAMPLE_CSV, content_json, server_session


async def _load(session):
    return content_json(await session.call_tool("load_csv", {"path": SAMPLE_CSV}))["dataset_id"]


async def test_hypothesis_and_correlation():
    async with server_session() as session:
        ds = await _load(session)

        anova = content_json(await session.call_tool(
            "hypothesis_test",
            {"dataset_id": ds, "test": "anova", "value_column": "units", "group_column": "region"},
        ))
        assert "p_value" in anova and isinstance(anova["significant"], bool)

        ctest = content_json(await session.call_tool(
            "correlation_test",
            {"dataset_id": ds, "x": "units", "y": "unit_price"},
        ))
        assert -1.0 <= ctest["correlation"] <= 1.0 and "interpretation" in ctest


async def test_trend_and_distribution():
    async with server_session() as session:
        ds = await _load(session)

        trend = content_json(await session.call_tool(
            "trend_analysis",
            {"dataset_id": ds, "time_column": "date", "value_column": "units"},
        ))
        assert trend["direction"] in {"increasing", "decreasing", "flat"}
        assert trend["n_periods"] > 1

        dist = content_json(await session.call_tool(
            "distribution_fit", {"dataset_id": ds, "column": "unit_price"},
        ))
        assert dist["n"] == 12 and "skewness" in dist


async def test_make_chart():
    async with server_session() as session:
        ds = await _load(session)
        chart = content_json(await session.call_tool(
            "make_chart",
            {"dataset_id": ds, "kind": "bar", "x": "region", "y": "units", "title": "Units by region"},
        ))
        assert os.path.exists(chart["path"])
        assert chart["data_uri"].startswith("data:image/png;base64,")


async def test_export_report():
    async with server_session() as session:
        ds = await _load(session)
        chart = content_json(await session.call_tool(
            "make_chart", {"dataset_id": ds, "kind": "heatmap", "title": "Correlations"},
        ))
        rep = content_json(await session.call_tool(
            "export_report",
            {
                "title": "Sales Analysis",
                "summary": "North leads in unit sales; price and units are weakly related.",
                "sections": [
                    {"heading": "Regional performance", "body": "North sold the most units.",
                     "chart_path": chart["path"]},
                ],
            },
        ))
        assert os.path.exists(rep["path"]) and rep["path"].endswith(".html")
        html = open(rep["path"], encoding="utf-8").read()
        assert "Sales Analysis" in html and "data:image/png;base64," in html
