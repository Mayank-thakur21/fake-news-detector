"""SQLite-backed prediction log used for ``/stats`` and ``/history``."""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    label        TEXT    NOT NULL,
    is_fake      INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    source       TEXT    NOT NULL,
    url          TEXT,
    title        TEXT,
    text_preview TEXT    NOT NULL,
    text_sha256  TEXT    NOT NULL,
    client_id    TEXT
);
CREATE INDEX IF NOT EXISTS ix_predictions_created ON predictions(created_at);
"""

EXPORT_COLUMNS = (
    "id",
    "created_at",
    "model",
    "label",
    "confidence",
    "source",
    "url",
    "title",
    "text_preview",
)


class PredictionStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # One shared connection guarded by a lock: simple and safe for the
        # modest write volume of this service. WAL lets readers run concurrently.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as cur:
            if self.db_path != ":memory:":
                cur.execute("PRAGMA journal_mode=WAL")
            cur.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        model: str,
        label: str,
        is_fake: bool,
        confidence: float,
        source: str,
        text: str,
        url: str | None = None,
        title: str | None = None,
        client_id: str | None = None,
    ) -> int | None:
        """Persist a prediction. Storage failures are logged, never raised to the caller."""
        try:
            with self._tx() as cur:
                cur.execute(
                    "INSERT INTO predictions (created_at, model, label, is_fake, confidence,"
                    " source, url, title, text_preview, text_sha256, client_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        model,
                        label,
                        int(is_fake),
                        round(confidence, 2),
                        source,
                        url,
                        title,
                        text[:200],
                        hashlib.sha256(text.encode()).hexdigest(),
                        client_id,
                    ),
                )
                return cur.lastrowid
        except sqlite3.Error:
            logger.exception("Failed to record prediction")
            return None

    def stats(self) -> dict[str, Any]:
        since = (datetime.now(UTC) - timedelta(hours=24)).isoformat(timespec="seconds")
        with self._tx() as cur:
            total = cur.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            last_24h = cur.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            avg_conf = cur.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]

            def group(col: str) -> dict[str, int]:
                rows = cur.execute(f"SELECT {col}, COUNT(*) FROM predictions GROUP BY {col}")
                return {str(r[0]): r[1] for r in rows}

            return {
                "total_predictions": total,
                "last_24h": last_24h,
                "average_confidence": round(avg_conf, 2) if avg_conf is not None else None,
                "by_label": group("label"),
                "by_model": group("model"),
                "by_source": group("source"),
            }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._tx() as cur:
            rows = cur.execute(
                f"SELECT {', '.join(EXPORT_COLUMNS)} FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, limit: int = 10_000) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(self.recent(limit))
        return buf.getvalue()
