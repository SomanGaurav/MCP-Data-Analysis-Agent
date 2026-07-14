# 📊 Autonomous Data Analysis System (MCP)

An AI data analyst that answers natural-language questions over CSV / Excel / SQL
data. The language model **plans and interprets** — all computation (SQL, stats,
cleaning, charts) runs in deterministic **MCP tools**, so results are exact,
reproducible, and auditable.

> **Core principle:** the LLM never does arithmetic. It reads schemas, chooses
> which tool to call with which arguments, and turns the returned numbers into
> business insight.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Streamlit UI   (upload · chat · charts · report · switch) │
└───────────────┬──────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│  Agent host                                                │
│   • Provider switch (LiteLLM): Ollama · Groq · Gemini      │
│   • MCP client: MCP tool schemas ⇆ OpenAI function calls   │
│   • Loop: plan → call tool → observe → answer              │
└───────────────┬──────────────────────────────────────────┘
                │ MCP protocol (stdio)
┌───────────────▼──────────────────────────────────────────┐
│  MCP analysis server                                       │
│   load · profile · sql · eda · clean · stats · viz · report│
│   Dataset registry · Pandas · DuckDB · SciPy · matplotlib  │
└──────────────────────────────────────────────────────────┘
```

See [DESIGN.md](DESIGN.md) for the full design and milestone tracker.

## Features

- **Upload** CSV / Excel; **DuckDB SQL** over your files with no database server.
- **Automated profiling & EDA** — schema, nulls, cardinality, correlations, group aggregates.
- **Non-destructive cleaning** — missing values, duplicates, type casts, outliers (each returns a new versioned dataset).
- **Statistics** — t-test / ANOVA / chi-square, correlation tests, trend analysis, distribution/normality.
- **Charts** — bar, line, scatter, histogram, box, correlation heatmap (PNG).
- **Self-contained HTML reports** with embedded charts.
- **Pluggable LLM** — local **Ollama** by default, switch to **Groq** or **Gemini** free tiers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env               # then fill in the provider you want
```

### Choosing a provider (edit `.env`)

| Provider | Setup | Notes |
|----------|-------|-------|
| **Ollama** (default, local) | `ollama pull qwen2.5:3b` and run the Ollama daemon | Free & private; smaller models plan tools less reliably |
| **Groq** (free tier) | `GROQ_API_KEY=...` from https://console.groq.com/keys | Fast, strong tool-calling — best for reliable planning |
| **Gemini** (free tier) | `GEMINI_API_KEY=...` from https://aistudio.google.com/app/apikey | Large context, strong function-calling |

You can also switch provider live from the sidebar dropdown in the UI.

## Run the app

```bash
streamlit run ui/streamlit_app.py
```

Then upload one of the samples in `data/` (e.g. `sample_sales.csv`) and ask
things like *"Which region sells the most units, and is the trend rising?"* or
*"Which plan has the highest churn — build me a report."*

## Use the MCP server directly (e.g. Claude Desktop)

The server speaks stdio and works with any MCP client:

```bash
python -m mcp_server.server
```

Claude Desktop config:

```json
{ "mcpServers": {
    "data-analysis": { "command": "python", "args": ["-m", "mcp_server.server"] }
} }
```

## MCP tool catalog

| Group | Tools |
|-------|-------|
| Load | `load_csv`, `load_excel`, `list_datasets` |
| Profile | `profile_dataset` |
| SQL | `run_sql` (DuckDB) |
| EDA | `value_counts`, `correlations`, `groupby_aggregate` |
| Clean | `drop_duplicates`, `handle_missing`, `cast_types`, `handle_outliers`, `rename_columns` |
| Stats | `hypothesis_test`, `correlation_test`, `trend_analysis`, `distribution_fit` |
| Viz | `make_chart` |
| Report | `export_report` |

## Testing

```bash
PYTHONPATH="$PWD" pytest -q
```

15 tests spawn the real MCP server over stdio and exercise every tool group;
the agent loop is tested with a scripted fake LLM (deterministic, no network).

## Project structure

```
mcp_server/     FastMCP server + registry + tools/
agent/          provider switch (config, providers) + mcp_client + agent loop
ui/             streamlit_app.py
data/           sample datasets + uploads/
reports/        generated charts and HTML reports
tests/          per-milestone test suites
```
