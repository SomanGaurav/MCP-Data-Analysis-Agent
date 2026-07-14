# Design Plan — Autonomous Data Analysis System (MCP)

> Status: **approved 2026-07-26**. Build order tracked in §7. Defaults chosen:
> LiteLLM for the provider switch · HTML report first · DuckDB-over-files now,
> SQL DBs (SQLite/Postgres/MySQL) later.

## 1. Core idea & the one principle that shapes everything

The LLM **never computes**. It reads schemas, decides *which* tool to call with
*what* arguments, and interprets the returned numbers into plain-language
insight. All actual computation (stats, SQL, cleaning, plotting) happens in
deterministic Python inside **MCP tools**. This keeps results correct,
reproducible, and auditable — the LLM is a planner + narrator, not a calculator.

## 2. High-level architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI  (upload · chat · charts · report · switch) │
└───────────────┬─────────────────────────────────────────┘
                │ calls
┌───────────────▼─────────────────────────────────────────┐
│  Agent Host (orchestration loop)                          │
│   • LLM Provider layer  ← switch: Ollama / Gemini / Groq  │
│   • MCP Client          ← discovers tools, calls them     │
│   • Plan→Act→Observe→Answer loop                          │
└───────────────┬─────────────────────────────────────────┘
                │ MCP protocol (stdio)
┌───────────────▼─────────────────────────────────────────┐
│  MCP Analysis Server (the engine)                         │
│   Tools: load · profile · sql · clean · stats · viz · report │
│   Dataset Registry (in-memory DataFrames keyed by id)     │
│   Backends: Pandas/Polars · DuckDB · matplotlib/plotly    │
└──────────────────────────────────────────────────────────┘
```

## 3. The provider switch

Use **LiteLLM** as the provider abstraction. One OpenAI-style `completion()`
call works across **Ollama (local), Gemini, and Groq** and normalizes their
tool-calling formats. The switch is a single dropdown + config.

| Provider | Role | Tool-calling | Notes |
|----------|------|-------------|-------|
| **Ollama (local)** | Default | Qwen2.5 / Llama3.2 support it | Free, private, fits 4GB VRAM — small models plan tools less reliably |
| **Groq** | Free-tier fallback | Strong (Llama-3.3-70B) | Very fast, generous free tier — best for reliable planning |
| **Gemini** | Free-tier fallback | Strong function-calling | Good free tier, large context |

**Degraded mode:** if a model's native tool-calling is weak, the host falls back
to a structured "plan-as-JSON" prompt instead of native function calls.

## 4. MCP Server — the tool catalog

One MCP server (FastMCP, Python SDK), stateful via a **dataset registry**
(load once → get a `dataset_id` → later tools reference it, so full tables never
pass through the LLM).

**Connectivity**
- `load_csv` · `load_excel` · `connect_sql` → returns `dataset_id` + schema summary

**Profiling / EDA**
- `profile_dataset` — dtypes, null counts, cardinality, numeric summary, sample rows
- `describe`, `value_counts`, `correlations`, `groupby_aggregate`

**SQL**
- `run_sql(query)` — DuckDB over the registered frames

**Cleaning** (non-destructive — each returns a new versioned `dataset_id`)
- `handle_missing` · `drop_duplicates` · `cast_types` · `handle_outliers` · `rename/normalize`

**Statistics / trends**
- `hypothesis_test` (t-test, chi-square, ANOVA) · `correlation_test` · `trend_analysis` · `distribution_fit`

**Visualization**
- `make_chart(kind, x, y, ...)` — saved PNG + spec; bar/line/scatter/hist/box/heatmap

**Reporting**
- `export_report(format)` — assembles findings + charts into HTML (first), then PDF/Markdown

## 5. Agent loop

1. User asks a NL question in Streamlit.
2. Host sends: system prompt + tool schemas + current dataset schema (never full data).
3. LLM emits a tool call → MCP client executes on the server → result returned.
4. Loop (observe → decide next tool) until the LLM can answer.
5. LLM writes the business insight + attaches charts.
6. All tool calls logged → audit trail + downloadable report.

## 6. Repo structure

```
Devops using MCP/
├── mcp_server/        # FastMCP server + tools/ (load, eda, sql, clean, stats, viz, report)
│   ├── server.py
│   ├── registry.py    # dataset registry
│   └── tools/
├── agent/             # host: provider layer (litellm) + mcp client + loop
│   ├── providers.py   # ollama | gemini | groq switch
│   ├── mcp_client.py
│   └── agent.py
├── ui/streamlit_app.py
├── reports/  data/  tests/
├── requirements.txt   .env.example   README.md
```

## 7. Milestones

| # | Milestone | Outcome | State |
|---|-----------|---------|-------|
| **M0** | Repo scaffold + `requirements.txt` + `.env.example` | Project boots | ✅ done |
| **M1** | MCP server: `load_csv` + `profile_dataset` + registry | Load & profile a CSV | ✅ done (smoke test green) |
| **M2** | SQL (DuckDB) + EDA + cleaning tools | Full analytical engine | ✅ done |
| **M3** | Stats + visualization + report tools | Complete tool catalog | ✅ done |
| **M4** | Provider layer (LiteLLM) + Ollama/Gemini/Groq switch | LLM talks, swappable | ✅ done |
| **M5** | Agent loop + MCP client wiring | End-to-end NL → tools → insight | ✅ done |
| **M6** | Streamlit UI: upload, chat, charts, switch, report download | Usable product | ✅ done |
| **M7** | Tests, sample datasets, README, polish | Portfolio-ready | ✅ done (15 tests green) |

## 8. Confirmed tech decisions

- **LiteLLM** for the provider switch.
- **DuckDB** as SQL engine over Pandas frames (uniform CSV/Excel/SQL).
- **Pandas** default frame lib; Polars optional later (M1–M6 Pandas-only).
- **matplotlib** for report charts (clean PNGs); Plotly optional for UI later.
- **Report format order:** HTML first, then PDF, then Markdown.
- **SQL DB connectivity:** DuckDB-over-files now; SQLite/Postgres/MySQL in a later pass.
