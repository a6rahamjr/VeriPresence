from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class AttendanceRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record(
        self,
        identity: str | None,
        accepted: bool,
        confidence: float,
        margin: float,
        source: str,
        cooldown_seconds: int,
        captured_at: datetime | None = None,
    ) -> bool:
        captured_at = captured_at or datetime.now(timezone.utc)
        event_type = "attendance" if accepted else "unknown"
        cutoff = captured_at - timedelta(seconds=cooldown_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                """
                SELECT captured_at FROM recognition_events
                WHERE event_type = ? AND COALESCE(identity, '') = COALESCE(?, '')
                    AND source = ?
                ORDER BY captured_at DESC LIMIT 1
                """,
                (event_type, identity, source),
            ).fetchone()
            if previous and datetime.fromisoformat(previous[0]) >= cutoff:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO recognition_events
                    (identity, event_type, confidence, margin, source, captured_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    identity,
                    event_type,
                    confidence,
                    margin,
                    source,
                    captured_at.isoformat(),
                ),
            )
            connection.commit()
            return True

    def list_events(
        self,
        event_date: str | None = None,
        event_type: str | None = None,
        identity: str | None = None,
        source: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT id, identity, event_type, confidence, margin, source, captured_at
            FROM recognition_events
        """
        parameters: list[Any] = []
        conditions = []
        if event_date:
            conditions.append("substr(captured_at, 1, 10) = ?")
            parameters.append(event_date)
        if event_type:
            conditions.append("event_type = ?")
            parameters.append(event_type)
        if identity:
            conditions.append("identity = ?")
            parameters.append(identity)
        if source:
            conditions.append("source = ?")
            parameters.append(source)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY captured_at DESC LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def daily_summary(
        self, event_date: str, source: str | None = None
    ) -> dict[str, Any]:
        source_clause = " AND source = ?" if source else ""
        parameters: list[Any] = [event_date]
        if source:
            parameters.append(source)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT identity,
                       MIN(captured_at) AS first_seen,
                       MAX(captured_at) AS last_seen,
                       COUNT(*) AS sightings,
                       AVG(confidence) AS average_confidence
                FROM recognition_events
                WHERE event_type = 'attendance'
                    AND substr(captured_at, 1, 10) = ?
                    {source_clause}
                GROUP BY identity
                ORDER BY first_seen
                """,
                parameters,
            ).fetchall()
            unknown_row = connection.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM recognition_events
                WHERE event_type = 'unknown'
                    AND substr(captured_at, 1, 10) = ?
                    {source_clause}
                """,
                parameters,
            ).fetchone()

        return {
            "date": event_date,
            "source": source,
            "present_count": len(rows),
            "unknown_events": int(unknown_row["total"]),
            "identities": [dict(row) for row in rows],
        }

    def delete_before(self, cutoff_date: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM recognition_events WHERE captured_at < ?",
                (cutoff_date,),
            )
            connection.commit()
            return int(cursor.rowcount)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT,
                    event_type TEXT NOT NULL CHECK(event_type IN ('attendance', 'unknown')),
                    confidence REAL NOT NULL,
                    margin REAL NOT NULL,
                    source TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recognition_events_time
                ON recognition_events(captured_at DESC)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
