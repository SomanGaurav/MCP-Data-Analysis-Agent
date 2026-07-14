"""M2 tests: SQL (DuckDB), EDA tools, and non-destructive cleaning."""

from tests.conftest import SAMPLE_CSV, content_json, server_session


async def _load(session):
    loaded = content_json(await session.call_tool("load_csv", {"path": SAMPLE_CSV}))
    return loaded["dataset_id"]


async def test_run_sql():
    async with server_session() as session:
        ds = await _load(session)
        res = content_json(await session.call_tool(
            "run_sql",
            {"query": f"SELECT region, SUM(units) AS total FROM {ds} GROUP BY region ORDER BY total DESC"},
        ))
        assert res["columns"] == ["region", "total"]
        top = res["rows"][0]
        # North: 10+7+10+10(dup) = 37 is the largest regional unit total
        assert top["region"] == "North" and top["total"] == 37


async def test_sql_register_as_dataset():
    async with server_session() as session:
        ds = await _load(session)
        res = content_json(await session.call_tool(
            "run_sql",
            {"query": f"SELECT * FROM {ds} WHERE category = 'Gadgets'", "register_as_dataset": True},
        ))
        assert "dataset_id" in res and res["dataset_id"].startswith("sql_")


async def test_eda_tools():
    async with server_session() as session:
        ds = await _load(session)

        vc = content_json(await session.call_tool("value_counts", {"dataset_id": ds, "column": "region"}))
        counts = {c["value"]: c["count"] for c in vc["counts"]}
        assert counts["North"] == 4

        corr = content_json(await session.call_tool("correlations", {"dataset_id": ds}))
        assert "units" in corr["matrix"]
        assert isinstance(corr["strongest_pairs"], list)

        grp = content_json(await session.call_tool(
            "groupby_aggregate",
            {"dataset_id": ds, "group_by": ["category"], "metric": "units", "agg": "sum"},
        ))
        top = grp["rows"][0]
        assert top["category"] == "Widgets"  # widgets have the most total units


async def test_cleaning_non_destructive():
    async with server_session() as session:
        ds = await _load(session)

        dedup = content_json(await session.call_tool("drop_duplicates", {"dataset_id": ds}))
        assert dedup["parent_id"] == ds
        assert dedup["rows"] == 11  # one duplicate removed from 12

        # Original dataset is untouched.
        orig = content_json(await session.call_tool("profile_dataset", {"dataset_id": ds}))
        assert orig["rows"] == 12

        filled = content_json(await session.call_tool(
            "handle_missing",
            {"dataset_id": ds, "strategy": "median", "columns": ["customer_age"]},
        ))
        age = next(c for c in filled["columns"] if c["name"] == "customer_age")
        assert age["nulls"] == 0

        clipped = content_json(await session.call_tool(
            "handle_outliers",
            {"dataset_id": ds, "column": "units", "method": "iqr", "action": "remove"},
        ))
        assert clipped["operation"].startswith("handle_outliers")
