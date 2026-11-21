from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def resolve_sqlite_path(database_url: str) -> str:
    """Convert a SQLAlchemy-style SQLite DATABASE_URL into a filesystem path or ':memory:'."""
    url = database_url.strip()
    if url in {":memory:", "sqlite:///:memory:", "sqlite://"}:
        return ":memory:"

    if url.startswith("sqlite+aiosqlite:///"):
        url = "sqlite:///" + url[len("sqlite+aiosqlite:///") :]

    if url.startswith("sqlite:////"):
        # Four slashes = absolute POSIX path (e.g. sqlite:////data/app.db -> /data/app.db)
        return "/" + url[len("sqlite:////") :]
    if url.startswith("sqlite:///"):
        # Three slashes = relative path (e.g. sqlite:///./data/app.db) or Windows drive path
        return url[len("sqlite:///") :]

    return url


class Database:
    """Thread-safe SQLite storage for sessions and multi-turn message history."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.db_path = resolve_sqlite_path(database_url)
        self._lock = threading.RLock()
        self._memory_conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._memory_conn = conn
            return self._memory_conn

        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_if_not_memory(self, conn: sqlite3.Connection) -> None:
        if self.db_path != ":memory:":
            conn.close()

    def init_schema(self) -> None:
        """Create required SQLite schema automatically on startup."""
        with self._lock:
            conn = self._connect()
            try:
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id  TEXT NOT NULL,
                        role        TEXT NOT NULL,
                        content     TEXT NOT NULL,
                        metadata    TEXT,
                        created_at  TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                        ON messages (session_id, id);

                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id    TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                self._close_if_not_memory(conn)

    def ping(self) -> bool:
        """Return True if SQLite is reachable and responsive."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT 1").fetchone()
                return row is not None and row[0] == 1
            finally:
                self._close_if_not_memory(conn)

    def get_session_system_prompt(self, session_id: str) -> Optional[str]:
        """Return persisted system_prompt for a session, or None if not set."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT system_prompt FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                return row["system_prompt"]
            finally:
                self._close_if_not_memory(conn)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return ordered history for a session (oldest first), without system prompt."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT role, content, metadata, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
                results: list[dict[str, Any]] = []
                for row in rows:
                    item: dict[str, Any] = {
                        "role": row["role"],
                        "content": row["content"],
                    }
                    if row["role"] == "assistant" and row["metadata"]:
                        try:
                            item["metadata"] = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            item["metadata"] = None
                    results.append(item)
                return results
            finally:
                self._close_if_not_memory(conn)

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        assistant_metadata: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Atomically persist both user and assistant messages after a successful reply."""
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(assistant_metadata)

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    existing = conn.execute(
                        "SELECT system_prompt FROM sessions WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()

                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (session_id, system_prompt, now_iso, now_iso),
                        )
                    else:
                        new_prompt = system_prompt if system_prompt is not None else existing["system_prompt"]
                        conn.execute(
                            """
                            UPDATE sessions
                            SET system_prompt = ?, updated_at = ?
                            WHERE session_id = ?
                            """,
                            (new_prompt, now_iso, session_id),
                        )

                    conn.execute(
                        """
                        INSERT INTO messages (session_id, role, content, metadata, created_at)
                        VALUES (?, 'user', ?, NULL, ?)
                        """,
                        (session_id, user_message, now_iso),
                    )
                    conn.execute(
                        """
                        INSERT INTO messages (session_id, role, content, metadata, created_at)
                        VALUES (?, 'assistant', ?, ?, ?)
                        """,
                        (session_id, assistant_reply, metadata_json, now_iso),
                    )
            finally:
                self._close_if_not_memory(conn)

    def close(self) -> None:
        with self._lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None
