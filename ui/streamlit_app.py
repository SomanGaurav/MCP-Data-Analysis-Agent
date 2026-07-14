"""Streamlit UI for the Autonomous Data Analysis System.

Upload a CSV/Excel file, pick an LLM provider (Ollama / Groq / Gemini), and ask
questions in natural language. The agent plans and calls MCP tools; this page
renders the answer, the tool trace, any charts, and a downloadable HTML report.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Make the project importable when run via `streamlit run ui/streamlit_app.py`.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agent.agent import DataAnalysisAgent          # noqa: E402
from agent.config import VALID_PROVIDERS            # noqa: E402
from agent.providers import LLMClient               # noqa: E402

UPLOAD_DIR = os.path.join(REPO, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Autonomous Data Analyst (MCP)", page_icon="📊", layout="wide")


def save_upload(uploaded) -> str:
    path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    return path


# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM provider", VALID_PROVIDERS,
                            help="Ollama runs locally; Groq and Gemini use free-tier APIs.")
    client = LLMClient(provider)
    ready, msg = client.is_ready()
    st.caption(f"Model: `{client.model}`")
    if ready:
        st.success("Provider ready ✅")
    else:
        st.warning(msg)

    st.divider()
    st.subheader("📁 Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        st.session_state["data_path"] = save_upload(uploaded)
        st.success(f"Loaded: {uploaded.name}")
    data_path = st.session_state.get("data_path")
    if data_path:
        st.caption(f"Active file: `{os.path.basename(data_path)}`")

    st.divider()
    st.caption("Analytical work runs in MCP tools (Pandas · DuckDB · SciPy · "
               "matplotlib). The model only plans and interprets.")


# ------------------------------- Main --------------------------------------
st.title("📊 Autonomous Data Analyst")
st.write("Ask a question about your dataset in plain English.")

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        for chart in turn["charts"]:
            st.image(chart["data_uri"], caption=chart["title"])
        if turn["report_path"] and os.path.exists(turn["report_path"]):
            with open(turn["report_path"], "rb") as fh:
                st.download_button("⬇️ Download HTML report", fh.read(),
                                   file_name=os.path.basename(turn["report_path"]),
                                   mime="text/html", key=turn["report_path"])
        with st.expander(f"🔧 Tool trace ({len(turn['trace'])} calls · {turn['provider']})"):
            for rec in turn["trace"]:
                st.markdown(f"**{rec['step']}. `{rec['name']}`** — args: `{rec['args']}`")


question = st.chat_input("e.g. Which region sells the most, and is the trend rising?")
if question:
    if not ready:
        st.error(msg)
        st.stop()

    context = ""
    if data_path:
        context = (f"A dataset file is available at this path: {data_path}. "
                   f"Load it with load_csv or load_excel before analyzing.")

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        status = st.status("Analyzing…", expanded=True)

        def on_event(kind: str, data: dict):
            if kind == "tool_call":
                status.write(f"🔧 calling `{data['name']}`")

        agent = DataAnalysisAgent(client)
        try:
            result = agent.run(question, preload_context=context, on_event=on_event)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"Agent error: {exc}")
            st.stop()

        status.update(label=f"Done in {result.steps} step(s)", state="complete")
        st.markdown(result.answer)
        for chart in result.charts:
            st.image(chart["data_uri"], caption=chart["title"])
        if result.report_path and os.path.exists(result.report_path):
            with open(result.report_path, "rb") as fh:
                st.download_button("⬇️ Download HTML report", fh.read(),
                                   file_name=os.path.basename(result.report_path),
                                   mime="text/html", key=result.report_path)

    st.session_state["history"].append({
        "question": question,
        "answer": result.answer,
        "charts": result.charts,
        "report_path": result.report_path,
        "provider": f"{result.provider}/{result.model}",
        "trace": [{"step": r.step, "name": r.name, "args": r.arguments} for r in result.trace],
    })
