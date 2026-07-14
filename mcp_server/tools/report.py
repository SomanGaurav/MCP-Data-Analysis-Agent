"""Reporting tool: assemble findings + charts into a self-contained HTML report.

Charts are embedded as base64 so the report is a single portable file. HTML is
the first supported format (PDF/Markdown can follow the same structure later).
"""

from __future__ import annotations

import base64
import html
import os
import uuid
from datetime import datetime
from typing import Optional

from mcp_server.paths import REPORTS_DIR

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:900px;margin:2rem auto;padding:0 1.25rem;color:#1a1a1a;line-height:1.6}
h1{border-bottom:3px solid #2563eb;padding-bottom:.4rem}
h2{color:#2563eb;margin-top:2rem}
.meta{color:#666;font-size:.9rem}
.summary{background:#f1f5f9;border-left:4px solid #2563eb;padding:1rem 1.25rem;border-radius:4px}
.section{margin:1.5rem 0}
img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin-top:.75rem}
.body{white-space:pre-wrap}
footer{margin-top:3rem;color:#999;font-size:.8rem;border-top:1px solid #eee;padding-top:1rem}
"""


def _img_tag(chart_path: Optional[str]) -> str:
    if not chart_path or not os.path.exists(chart_path):
        return ""
    with open(chart_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="chart"/>'


def register(mcp) -> None:
    """Attach the report tool to the given FastMCP instance."""

    @mcp.tool()
    def export_report(
        title: str,
        summary: str,
        sections: list[dict],
        fmt: str = "html",
    ) -> dict:
        """Assemble an analytical report and write it to reports/.

        Args:
            title: Report title.
            summary: An executive summary / key takeaways paragraph.
            sections: Ordered list of sections. Each is a dict with:
                - "heading" (str)
                - "body" (str: findings, insights, recommendations)
                - "chart_path" (optional str: a PNG path from make_chart)
            fmt: Output format. Only "html" is supported for now.

        Returns the path to the written report file.
        """
        if fmt != "html":
            raise ValueError("Only fmt='html' is supported currently.")

        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p class='meta'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
            f"<div class='summary'><strong>Summary.</strong> {html.escape(summary)}</div>",
        ]
        for sec in sections:
            heading = html.escape(str(sec.get("heading", "")))
            body = html.escape(str(sec.get("body", "")))
            img = _img_tag(sec.get("chart_path"))
            parts.append(
                f"<div class='section'><h2>{heading}</h2>"
                f"<div class='body'>{body}</div>{img}</div>"
            )
        parts.append(
            "<footer>Produced by the Autonomous Data Analysis System (MCP). "
            "All figures computed by deterministic tools, not the language model.</footer>"
            "</body></html>"
        )

        fname = f"report_{uuid.uuid4().hex[:8]}.html"
        path = os.path.join(REPORTS_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(parts))

        return {"path": path, "format": fmt, "sections": len(sections)}
