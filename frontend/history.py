"""Session-scoped prediction history helpers (pure functions, easy to test)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

HISTORY_COLUMNS = ["Time", "Source", "Input", "Model", "Prediction", "Confidence (%)"]


def make_entry(result: dict[str, Any], *, input_preview: str, source: str) -> dict[str, Any]:
    """Convert an API prediction into a history row."""
    preview = input_preview.strip().replace("\n", " ")
    return {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Input": preview[:120] + ("…" if len(preview) > 120 else ""),
        "Model": result.get("model_name", result.get("model", "?")),
        "Prediction": result["prediction"],
        "Confidence (%)": result["confidence"],
    }


def to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(history, columns=HISTORY_COLUMNS)


def to_csv_bytes(history: list[dict[str, Any]]) -> bytes:
    """CSV with a UTF-8 BOM so Excel opens non-ASCII text correctly."""
    return to_dataframe(history).to_csv(index=False).encode("utf-8-sig")


def parse_batch_text(raw: str, separator: str = "---") -> list[dict[str, str]]:
    """Split pasted input into batch items.

    * Blocks separated by a line containing only ``---`` become separate articles.
    * A block that is a single http(s) URL is sent as a URL item.
    """
    items: list[dict[str, str]] = []
    block: list[str] = []

    def flush() -> None:
        text = "\n".join(block).strip()
        block.clear()
        if not text:
            return
        if "\n" not in text and text.lower().startswith(("http://", "https://")):
            items.append({"url": text})
        else:
            items.append({"text": text})

    for line in raw.splitlines():
        if line.strip() == separator:
            flush()
        else:
            block.append(line)
    flush()
    return items


def parse_batch_csv(df: pd.DataFrame) -> list[dict[str, str]]:
    """Build batch items from an uploaded CSV with a ``text`` and/or ``url`` column."""
    cols = {c.lower().strip(): c for c in df.columns}
    if "text" not in cols and "url" not in cols:
        raise ValueError("CSV must have a 'text' or 'url' column")
    items = []
    for _, row in df.iterrows():
        url = str(row[cols["url"]]).strip() if "url" in cols and pd.notna(row[cols["url"]]) else ""
        text = (
            str(row[cols["text"]]).strip() if "text" in cols and pd.notna(row[cols["text"]]) else ""
        )
        if text:
            items.append({"text": text})
        elif url:
            items.append({"url": url})
    return items
