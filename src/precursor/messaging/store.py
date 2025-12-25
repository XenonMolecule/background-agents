"""
Conversation/message storage (SQLite).

Design goals
------------
- Simple: one table, one row per message.
- Project-scoped: all messages belong to a project_name.
- UI-friendly unread state: `seen_by_user` boolean per row.
- UI visibility: `visible_in_conversation` is interpreted as "visible_to_user".

We intentionally reuse the same SQLite DB file as the scratchpad store so a single
`PRECURSOR_SCRATCHPAD_DB` override controls both.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from precursor.projects.utils import is_valid_project
from precursor.scratchpad.store import get_conn

Role = Literal["user", "agent", "system"]


def init_db() -> None:
    """
    Create the conversation_messages table if it does not already exist.

    Safe to call multiple times.
    """
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            seen_by_user INTEGER NOT NULL DEFAULT 0,
            -- NOTE: this is a user-facing visibility flag (for UI display).
            -- System messages are often hidden from the user but still included
            -- in agent context rendering.
            visible_in_conversation INTEGER NOT NULL DEFAULT 1,
            -- Soft-delete: retain messages for study/audit but hide from both
            -- UI and agent rendering.
            is_deleted INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # lightweight migrations for existing DBs
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversation_messages)").fetchall()}
    if "visible_in_conversation" not in cols:
        conn.execute(
            "ALTER TABLE conversation_messages ADD COLUMN visible_in_conversation INTEGER NOT NULL DEFAULT 1"
        )
    if "is_deleted" not in cols:
        conn.execute("ALTER TABLE conversation_messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
    # Helpful index for UI badge queries + project-scoped fetches.
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_project_created
        ON conversation_messages(project_name, created_at)
        """
    )
    conn.commit()
    conn.close()


def add_message(
    project_name: str,
    role: Role,
    message: str,
    *,
    seen_by_user: Optional[bool] = None,
    visible_to_user: Optional[bool] = None,
    # Backwards-compatible alias (older callers/tests may still pass this name).
    visible_in_conversation: Optional[bool] = None,
) -> int:
    """
    Append a message to a project's conversation history.

    Parameters
    ----------
    project_name:
        Must exist in config/projects.yaml.
    role:
        "user" or "agent".
    message:
        The message text.
    seen_by_user:
        Whether the user has seen this message in the UI yet.
        If omitted:
        - user messages default to True (the user obviously saw what they sent)
        - agent messages default to False (unread badge until displayed)
    visible_in_conversation:
        Deprecated alias for visible_to_user.
    visible_to_user:
        Whether this message should be shown to the user in the UI.
        If omitted:
        - system messages default to False
        - user/agent messages default to True

    Returns
    -------
    int
        Internal database primary key for this message.
    """
    init_db()

    if not is_valid_project(project_name):
        raise ValueError(f"Unknown project: {project_name}")

    msg = str(message or "").strip()
    if not msg:
        raise ValueError("message must be non-empty")

    if role not in ("user", "agent", "system"):
        raise ValueError("role must be 'user', 'agent', or 'system'")

    if seen_by_user is None:
        # system messages shouldn't create "unread badges" by default
        if role == "system":
            seen_by_user = True
        else:
            seen_by_user = role == "user"

    if visible_to_user is None and visible_in_conversation is not None:
        visible_to_user = visible_in_conversation

    if visible_to_user is None:
        visible_to_user = role != "system"

    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO conversation_messages (project_name, role, message, seen_by_user, visible_in_conversation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            project_name,
            role,
            msg,
            1 if seen_by_user else 0,
            1 if visible_to_user else 0,
        ),
    )
    conn.commit()
    mid = cur.lastrowid
    conn.close()
    return mid


def list_messages(
    project_name: str,
    *,
    limit: Optional[int] = None,
    before_id: Optional[int] = None,
    include_invisible: bool = True,
) -> List[Dict[str, object]]:
    """
    List messages for a project ordered by created_at ascending.

    Parameters
    ----------
    limit:
        Optional maximum number of rows to return (most recent within the bound).
    before_id:
        Optional pagination anchor; return messages with id < before_id.
    """
    init_db()

    conn = get_conn()
    params: List[object] = [project_name]

    # Always exclude soft-deleted rows for both UI + agent rendering.
    where = "WHERE project_name = ? AND is_deleted = 0"
    if before_id is not None:
        where += " AND id < ?"
        params.append(int(before_id))
    # "invisible" here means "hidden from user UI"
    if not include_invisible:
        where += " AND visible_in_conversation = 1"

    # If a limit is provided, fetch newest first then reverse in Python so callers
    # still see chronological order.
    if limit is not None:
        params.append(int(limit))
        rows = conn.execute(
            f"""
            SELECT *
            FROM conversation_messages
            {where}
            ORDER BY id DESC
            LIMIT ?
            """
            ,
            tuple(params),
        ).fetchall()
        conn.close()
        rows = list(reversed(rows))
    else:
        rows = conn.execute(
            f"""
            SELECT *
            FROM conversation_messages
            {where}
            ORDER BY id ASC
            """
            ,
            tuple(params),
        ).fetchall()
        conn.close()

    out: List[Dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "project_name": r["project_name"],
                "role": r["role"],
                "message": r["message"],
                "seen_by_user": bool(r["seen_by_user"]),
                "visible_to_user": bool(r["visible_in_conversation"]),
                "created_at": r["created_at"],
            }
        )
    return out


def get_latest_message(
    project_name: str,
    *,
    include_invisible: bool = True,
) -> Optional[Dict[str, object]]:
    """
    Return the most recent message for a project, or None if no messages exist.
    """
    init_db()
    conn = get_conn()
    if include_invisible:
        row = conn.execute(
            """
            SELECT *
            FROM conversation_messages
            WHERE project_name = ?
              AND is_deleted = 0
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_name,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT *
            FROM conversation_messages
            WHERE project_name = ?
              AND visible_in_conversation = 1
              AND is_deleted = 0
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_name,),
        ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "project_name": row["project_name"],
        "role": row["role"],
        "message": row["message"],
        "seen_by_user": bool(row["seen_by_user"]),
        "visible_to_user": bool(row["visible_in_conversation"]),
        "created_at": row["created_at"],
    }


def get_latest_message_by_role(
    project_name: str,
    role: Role,
    *,
    include_invisible: bool = True,
) -> Optional[Dict[str, object]]:
    """
    Return the most recent message for a project with the given role, or None.
    """
    init_db()
    if role not in ("user", "agent", "system"):
        raise ValueError("role must be 'user', 'agent', or 'system'")

    conn = get_conn()
    if include_invisible:
        row = conn.execute(
            """
            SELECT *
            FROM conversation_messages
            WHERE project_name = ?
              AND role = ?
              AND is_deleted = 0
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_name, role),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT *
            FROM conversation_messages
            WHERE project_name = ?
              AND role = ?
              AND visible_in_conversation = 1
              AND is_deleted = 0
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_name, role),
        ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "project_name": row["project_name"],
        "role": row["role"],
        "message": row["message"],
        "seen_by_user": bool(row["seen_by_user"]),
        "visible_to_user": bool(row["visible_in_conversation"]),
        "created_at": row["created_at"],
    }


def count_unseen_by_user(project_name: str) -> int:
    """
    Count agent->user messages that have not been seen by the user.
    """
    init_db()
    conn = get_conn()
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM conversation_messages
        WHERE project_name = ?
          AND role = 'agent'
          AND seen_by_user = 0
          AND is_deleted = 0
        """,
        (project_name,),
    ).fetchone()
    conn.close()
    return int(row["c"] if row else 0)


def mark_seen_by_user(
    project_name: str,
    *,
    up_to_id: Optional[int] = None,
) -> int:
    """
    Mark agent messages as seen_by_user=1.

    Parameters
    ----------
    up_to_id:
        If provided, mark only messages with id <= up_to_id as seen.
        Otherwise mark all unseen agent messages for the project as seen.

    Returns
    -------
    int
        Number of rows updated.
    """
    init_db()
    conn = get_conn()

    if up_to_id is not None:
        cur = conn.execute(
            """
            UPDATE conversation_messages
            SET seen_by_user = 1
            WHERE project_name = ?
              AND role = 'agent'
              AND seen_by_user = 0
              AND is_deleted = 0
              AND id <= ?
            """,
            (project_name, int(up_to_id)),
        )
    else:
        cur = conn.execute(
            """
            UPDATE conversation_messages
            SET seen_by_user = 1
            WHERE project_name = ?
              AND role = 'agent'
              AND seen_by_user = 0
              AND is_deleted = 0
            """,
            (project_name,),
        )

    conn.commit()
    updated = cur.rowcount if cur.rowcount is not None else 0
    conn.close()
    return int(updated)


def trash_project_conversation(project_name: str) -> int:
    """
    Soft-delete all conversation messages for a project.

    Messages remain in the DB for study/audit, but are hidden from:
    - user UI
    - agent context rendering
    """
    init_db()
    if not is_valid_project(project_name):
        raise ValueError(f"Unknown project: {project_name}")

    conn = get_conn()
    cur = conn.execute(
        """
        UPDATE conversation_messages
        SET is_deleted = 1,
            -- Ensure they don't contribute to unread badge counts.
            seen_by_user = 1,
            -- UI-specific visibility off for good measure.
            visible_in_conversation = 0
        WHERE project_name = ?
          AND is_deleted = 0
        """,
        (project_name,),
    )
    conn.commit()
    updated = cur.rowcount if cur.rowcount is not None else 0
    conn.close()
    return int(updated)


