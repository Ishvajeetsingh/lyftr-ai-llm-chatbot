import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config import settings


def get_db_file_path(database_url: str) -> str:
    url = database_url.strip()
    if url.startswith("sqlite:////"):
        # Absolute path on POSIX or container, e.g. sqlite:////data/app.db -> /data/app.db
        path = "/" + url[11:]
    elif url.startswith("sqlite:///"):
        # Relative or standard path, e.g. sqlite:///./data/app.db -> ./data/app.db
        path = url[10:]
    elif url.startswith("sqlite://"):
        path = url[9:]
    else:
        path = url

    if path != ":memory:":
        abs_path = os.path.abspath(path)
        parent_dir = os.path.dirname(abs_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return abs_path
    return path


def get_connection() -> sqlite3.Connection:
    db_path = get_db_file_path(settings.DATABASE_URL)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id);"
        )
        conn.commit()
    finally:
        conn.close()


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, system_prompt, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "session_id": row["session_id"],
                "system_prompt": row["system_prompt"],
                "created_at": row["created_at"],
            }
        return None
    finally:
        conn.close()


def upsert_session(session_id: str, system_prompt: Optional[str] = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            if system_prompt is not None:
                cursor.execute(
                    "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                    (system_prompt, session_id),
                )
        else:
            cursor.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?)",
                (session_id, system_prompt, now_iso),
            )
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Oldest first, excluding system prompt
        cursor.execute(
            """
            SELECT id, session_id, role, content, metadata, created_at
            FROM messages
            WHERE session_id = ? AND role != 'system'
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            meta = None
            if r["metadata"]:
                try:
                    meta = json.loads(r["metadata"])
                except Exception:
                    meta = None
            result.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": meta,
                    "created_at": r["created_at"],
                }
            )
        return result
    finally:
        conn.close()


def save_turn(
    session_id: str,
    system_prompt: Optional[str],
    user_content: str,
    assistant_content: str,
    metadata: Dict[str, Any],
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Upsert session
        cursor.execute(
            "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            if system_prompt is not None and system_prompt != "":
                cursor.execute(
                    "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                    (system_prompt, session_id),
                )
        else:
            cursor.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?)",
                (session_id, system_prompt, now_iso),
            )

        # Insert user message
        cursor.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, "user", user_content, None, now_iso),
        )

        # Insert assistant message
        cursor.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "assistant",
                assistant_content,
                json.dumps(metadata),
                now_iso,
            ),
        )
        conn.commit()
    finally:
        conn.close()
