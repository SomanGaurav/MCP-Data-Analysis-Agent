"""In-memory dataset registry.

Datasets are loaded once and referenced by a short ``dataset_id`` for the rest
of a session, so full tables never have to pass through the LLM. Cleaning tools
(M2+) register *new* versioned datasets rather than mutating in place, which
keeps every transformation auditable and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


@dataclass
class DatasetEntry:
    dataset_id: str
    df: pd.DataFrame
    source: str                       # e.g. "csv:/path/file.csv" or "clean:handle_missing"
    created_at: str
    parent_id: Optional[str] = None   # set when derived from another dataset
    note: str = ""


class DatasetRegistry:
    """Holds all datasets for one server process, keyed by ``dataset_id``."""

    def __init__(self) -> None:
        self._entries: dict[str, DatasetEntry] = {}
        self._counter: int = 0

    def _next_id(self, prefix: str = "ds") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:03d}"

    def add(
        self,
        df: pd.DataFrame,
        source: str,
        parent_id: Optional[str] = None,
        note: str = "",
        prefix: str = "ds",
    ) -> DatasetEntry:
        entry = DatasetEntry(
            dataset_id=self._next_id(prefix),
            df=df,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            parent_id=parent_id,
            note=note,
        )
        self._entries[entry.dataset_id] = entry
        return entry

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self._entries:
            known = ", ".join(self._entries) or "(none loaded)"
            raise KeyError(
                f"Unknown dataset_id '{dataset_id}'. Loaded datasets: {known}"
            )
        return self._entries[dataset_id]

    def list(self) -> list[dict]:
        return [
            {
                "dataset_id": e.dataset_id,
                "source": e.source,
                "rows": int(e.df.shape[0]),
                "cols": int(e.df.shape[1]),
                "parent_id": e.parent_id,
                "created_at": e.created_at,
                "note": e.note,
            }
            for e in self._entries.values()
        ]


# One registry shared by all tools in this server process.
REGISTRY = DatasetRegistry()
