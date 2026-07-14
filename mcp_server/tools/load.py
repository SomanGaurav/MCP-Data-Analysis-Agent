"""Data loading tools: register CSV/Excel files into the dataset registry."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from mcp_server.registry import REGISTRY
from mcp_server.tools.summary import schema_summary


def register(mcp) -> None:
    """Attach loading tools to the given FastMCP instance."""

    @mcp.tool()
    def load_csv(path: str, delimiter: str = ",") -> dict:
        """Load a CSV file into the dataset registry.

        Args:
            path: Filesystem path to the .csv file.
            delimiter: Column delimiter (default ",").

        Returns a schema summary including the ``dataset_id`` to use in later tools.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path, sep=delimiter)
        entry = REGISTRY.add(df, source=f"csv:{path}")
        return schema_summary(df, entry.dataset_id, entry.source)

    @mcp.tool()
    def load_excel(path: str, sheet: Optional[str] = None) -> dict:
        """Load an Excel (.xlsx) sheet into the dataset registry.

        Args:
            path: Filesystem path to the .xlsx file.
            sheet: Sheet name; defaults to the first sheet when omitted.

        Returns a schema summary including the ``dataset_id`` to use in later tools.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Excel file not found: {path}")
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        src = f"excel:{path}" + (f"#{sheet}" if sheet else "")
        entry = REGISTRY.add(df, source=src)
        return schema_summary(df, entry.dataset_id, entry.source)

    @mcp.tool()
    def list_datasets() -> list[dict]:
        """List every dataset currently loaded in this session, with row/col counts."""
        return REGISTRY.list()
