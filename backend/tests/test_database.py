from __future__ import annotations

from pathlib import Path

from app.database import Database, resolve_sqlite_path


def test_resolve_sqlite_paths() -> None:
    assert resolve_sqlite_path("sqlite:////data/app.db") == "/data/app.db"
    assert resolve_sqlite_path("sqlite:///./data/app.db") == "./data/app.db"
    assert resolve_sqlite_path("sqlite:///:memory:") == ":memory:"


def test_schema_matches_contract_and_survives_restart(tmp_path: Path) -> None:
    db_file = tmp_path / "subdir" / "app.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    db1 = Database(db_url)
    db1.init_schema()
    assert db1.ping() is True

    # Verify exact required columns on `messages` table
    conn = db1._connect()
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    finally:
        db1._close_if_not_memory(conn)

    assert {"id", "session_id", "role", "content", "metadata", "created_at"}.issubset(cols)

    db1.save_exchange(
        session_id="sess-a",
        user_message="Hello A",
        assistant_reply="Hi A",
        assistant_metadata={"resolution_source": "LIVE_INFERENCE", "input_tokens": 10, "output_tokens": 4},
        system_prompt="Prompt A",
    )
    db1.save_exchange(
        session_id="sess-b",
        user_message="Hello B",
        assistant_reply="Hi B",
        assistant_metadata={"resolution_source": "LIVE_INFERENCE", "input_tokens": 11, "output_tokens": 5},
    )
    db1.close()

    # Simulate container restart with a brand-new Database instance pointing at the same file
    db2 = Database(db_url)
    db2.init_schema()
    msgs_a = db2.get_messages("sess-a")
    msgs_b = db2.get_messages("sess-b")

    assert len(msgs_a) == 2
    assert msgs_a[0]["content"] == "Hello A"
    assert msgs_a[1]["content"] == "Hi A"
    assert db2.get_session_system_prompt("sess-a") == "Prompt A"

    assert len(msgs_b) == 2
    assert msgs_b[0]["content"] == "Hello B"
    assert db2.get_session_system_prompt("sess-b") is None
    db2.close()
