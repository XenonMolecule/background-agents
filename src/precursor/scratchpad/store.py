# src/precursor/scratchpad/store.py
"""
Scratchpad storage layer (SQLite + platformdirs).

This module persists project scratchpad entries in a SQLite database located in
the user's platform-specific data directory under the "precursor" app folder.

Key design points
-----------------
- We store ONE row per scratchpad line (project + section [+ subsection] + message).
- The database has an internal primary key `id` for stability.
- The user / LLM will often refer to entries by *display index* inside a section
  (e.g. "edit Notes index 2"). To support that, we provide helper functions that
  map (project, section, [subsection], display_index) -> real DB row.

This keeps the DB robust while preserving the very convenient "indexing by [0], [1]"
behavior you had before in the in-memory version.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from platformdirs import user_data_dir

from precursor.scratchpad.schema import PROJECT_SECTIONS, PROJECT_RESOURCE_SUBSECTIONS
from precursor.projects.utils import is_valid_project
import os


# ============================================================================
# Paths / connection helpers
# ============================================================================

def _get_data_dir() -> Path:
    """
    Return the platform-specific user data directory for the 'precursor' app.

    Examples
    --------
    macOS:   ~/Library/Application Support/precursor/
    Linux:   ~/.local/share/precursor/
    Windows: C:\\Users\\<user>\\AppData\\Local\\precursor\\
    """
    data_dir = Path(user_data_dir(appname="precursor"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_db_path() -> Path:
    """
    Return the full path to the scratchpad SQLite database.
    """
    env_path = os.getenv("PRECURSOR_SCRATCHPAD_DB")
    if env_path:
        return Path(env_path)
    return _get_data_dir() / "scratchpad.db"


def get_conn() -> sqlite3.Connection:
    """
    Open a connection to the scratchpad database.

    Returns
    -------
    sqlite3.Connection
        A connection with row_factory set to sqlite3.Row for dict-like access.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# DB schema / init
# ============================================================================

def init_db() -> None:
    """
    Create the scratchpad_entries table if it does not already exist.

    This is safe to call multiple times.
    """
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scratchpad_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            section TEXT NOT NULL,
            subsection TEXT,
            message TEXT NOT NULL,
            confidence INTEGER DEFAULT 0,
            -- Stable, user-controlled ordering within (project, section, subsection).
            -- Used by the macOS UI for drag-reorder; older DBs may have this added via migration.
            sort_order INTEGER,
            -- Track the source of the last edit for post-hoc analysis.
            -- Expected values: "system" (default) or "user" (manual UI edits).
            last_edited_by TEXT DEFAULT 'system',
            status TEXT DEFAULT 'active',
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Lightweight migration: add sort_order for older DBs, and initialize it so ordering is stable.
    _ensure_sort_order_column(conn)
    _ensure_last_edited_by_column(conn)
    conn.commit()
    conn.close()

def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for r in rows:
        # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
        if r[1] == column:
            return True
    return False


def _ensure_sort_order_column(conn: sqlite3.Connection) -> None:
    """
    Ensure the scratchpad_entries.sort_order column exists and is initialized.

    We initialize per (project_name, section, subsection) group based on (created_at, id)
    so existing DBs preserve their original display order.
    """
    if not _column_exists(conn, "scratchpad_entries", "sort_order"):
        conn.execute("ALTER TABLE scratchpad_entries ADD COLUMN sort_order INTEGER;")

    # If sort_order is already populated, do nothing.
    row = conn.execute(
        """
        SELECT 1
        FROM scratchpad_entries
        WHERE sort_order IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if row is not None:
        return

    groups = conn.execute(
        """
        SELECT DISTINCT project_name, section, COALESCE(subsection, '') AS subsection_norm
        FROM scratchpad_entries
        WHERE status = 'active'
        """
    ).fetchall()

    for project_name, section, subsection_norm in groups:
        if subsection_norm == "":
            ids = conn.execute(
                """
                SELECT id
                FROM scratchpad_entries
                WHERE project_name = ?
                  AND section = ?
                  AND subsection IS NULL
                  AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC
                """,
                (project_name, section),
            ).fetchall()
        else:
            ids = conn.execute(
                """
                SELECT id
                FROM scratchpad_entries
                WHERE project_name = ?
                  AND section = ?
                  AND COALESCE(subsection, '') = ?
                  AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC
                """,
                (project_name, section, subsection_norm),
            ).fetchall()

        conn.executemany(
            "UPDATE scratchpad_entries SET sort_order = ? WHERE id = ?",
            [(idx, row[0]) for idx, row in enumerate(ids)],
        )

def _ensure_last_edited_by_column(conn: sqlite3.Connection) -> None:
    """
    Ensure the scratchpad_entries.last_edited_by column exists and is initialized.

    We default to "system" for all existing rows.
    """
    if not _column_exists(conn, "scratchpad_entries", "last_edited_by"):
        conn.execute("ALTER TABLE scratchpad_entries ADD COLUMN last_edited_by TEXT DEFAULT 'system';")
    conn.execute(
        """
        UPDATE scratchpad_entries
        SET last_edited_by = 'system'
        WHERE last_edited_by IS NULL
        """
    )


# ============================================================================
# Section normalization
# ============================================================================

def _classify_resource_section(raw_section: str) -> tuple[str, str]:
    """
    Map a free-form resource-ish label to our structured resource section.

    Returns
    -------
    (section, subsection)
        section    -> always "Project Resources"
        subsection -> one of PROJECT_RESOURCE_SUBSECTIONS
    """
    low = raw_section.lower()

    if "file" in low:
        return ("Project Resources", "Files")
    if "repo" in low or "git" in low:
        return ("Project Resources", "Repos")
    if "folder" in low or "dir" in low:
        return ("Project Resources", "Folders")
    if "collaborator" in low or "person" in low or "contact" in low:
        return ("Project Resources", "Core Collaborators")

    return ("Project Resources", "Other")


def _normalize_section_and_subsection(
    section: str,
    subsection: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Normalize an incoming section name (possibly messy / LLM-generated) into one
    of our canonical sections and, if relevant, subsections.

    - If it's one of the known top-level sections, return as-is.
    - If it smells like a resource label, map to ("Project Resources", <sub>).
    - Otherwise, put it under "Notes".
    """
    if section in PROJECT_SECTIONS:
        # but if it's Project Resources, validate subsection
        if section == "Project Resources":
            if subsection not in PROJECT_RESOURCE_SUBSECTIONS:
                subsection = "Other"
        return section, subsection

    # resource-ish?
    if any(
        key in section.lower()
        for key in ("file", "repo", "folder", "collaborator", "relevant resource")
    ):
        return _classify_resource_section(section)

    # fallback
    return "Notes", None


# ============================================================================
# Core CRUD (id-based)
# ============================================================================

def _safe_parse_metadata(metadata_json: Optional[str]) -> Dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        parsed = json.loads(metadata_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _row_to_public_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    mj = d.get("metadata_json")
    d["metadata"] = _safe_parse_metadata(mj)
    if "metadata_json" in d:
        del d["metadata_json"]
    return d

def add_entry(
    project_name: str,
    section: str,
    message: str,
    confidence: int = 0,
    subsection: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    last_edited_by: str = "system",
) -> int:
    """
    Append a new line to a project's scratchpad.

    Parameters
    ----------
    project_name : str
        Must exist in config/projects.yaml.
    section : str
        Desired section name (will be normalized).
    message : str
        Text of the entry.
    confidence : int
        Confidence score to display alongside the message.
    subsection : Optional[str]
        For "Project Resources", the finer-grained bucket.
    metadata : Optional[Dict[str, Any]]
        metadata is optional, stored as JSON, and not rendered.
        Use it for hidden details like long summaries, URIs, or task data.

    Returns
    -------
    int
        Internal database primary key for this entry.
    """
    if not is_valid_project(project_name):
        raise ValueError(f"Unknown project: {project_name}")

    section, subsection = _normalize_section_and_subsection(section, subsection)

    conn = get_conn()
    # Assign a stable sort_order within this (project, section, subsection) bucket.
    if section == "Project Resources":
        # Project Resources always has a subsection (validated/normalized above).
        next_sort = conn.execute(
            """
            SELECT COALESCE(MAX(sort_order), -1) + 1
            FROM scratchpad_entries
            WHERE project_name = ?
              AND section = ?
              AND COALESCE(subsection, '') = COALESCE(?, '')
              AND status = 'active'
            """,
            (project_name, section, subsection),
        ).fetchone()[0]
    else:
        next_sort = conn.execute(
            """
            SELECT COALESCE(MAX(sort_order), -1) + 1
            FROM scratchpad_entries
            WHERE project_name = ?
              AND section = ?
              AND status = 'active'
            """,
            (project_name, section),
        ).fetchone()[0]

    cur = conn.execute(
        """
        INSERT INTO scratchpad_entries (project_name, section, subsection, message, confidence, sort_order, last_edited_by, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_name,
            section,
            subsection,
            message,
            confidence,
            int(next_sort) if next_sort is not None else 0,
            (last_edited_by or "system"),
            json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
        ),
    )
    conn.commit()
    entry_id = cur.lastrowid
    conn.close()
    return entry_id


def list_entries(
    project_name: str,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List active entries for a project, optionally filtered by section.

    Returns rows ordered by created_at ascending.
    """
    conn = get_conn()
    if section:
        rows = conn.execute(
            """
            SELECT * FROM scratchpad_entries
            WHERE project_name = ? AND section = ? AND status = 'active'
            ORDER BY sort_order ASC, datetime(created_at) ASC, id ASC
            """,
            (project_name, section),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM scratchpad_entries
            WHERE project_name = ? AND status = 'active'
            ORDER BY section ASC, sort_order ASC, datetime(created_at) ASC, id ASC
            """,
            (project_name,),
        ).fetchall()
    conn.close()
    return [_row_to_public_dict(r) for r in rows]


def list_resource_entries(
    project_name: str,
) -> List[Dict[str, Any]]:
    """
    Convenience: list only "Project Resources" entries.
    """
    return list_entries(project_name, section="Project Resources")


def update_entry(
    entry_id: int,
    new_message: str,
    new_confidence: Optional[int] = None,
    new_metadata: Optional[Dict[str, Any]] = None,
    last_edited_by: str = "system",
) -> None:
    """
    Update an entry by its internal database id.
    metadata is optional, stored as JSON, and not rendered.
    Use it for hidden details like long summaries, URIs, or task data.
    """
    conn = get_conn()
    last_edited_by = (last_edited_by or "system").strip() or "system"
    if new_confidence is not None and new_metadata is not None:
        conn.execute(
            """
            UPDATE scratchpad_entries
            SET message = ?, confidence = ?, metadata_json = ?, last_edited_by = ?
            WHERE id = ?
            """,
            (
                new_message,
                new_confidence,
                json.dumps(new_metadata, ensure_ascii=False),
                last_edited_by,
                entry_id,
            ),
        )
    elif new_confidence is not None:
        conn.execute(
            """
            UPDATE scratchpad_entries
            SET message = ?, confidence = ?, last_edited_by = ?
            WHERE id = ?
            """,
            (new_message, new_confidence, last_edited_by, entry_id),
        )
    elif new_metadata is not None:
        conn.execute(
            """
            UPDATE scratchpad_entries
            SET message = ?, metadata_json = ?, last_edited_by = ?
            WHERE id = ?
            """,
            (new_message, json.dumps(new_metadata, ensure_ascii=False), last_edited_by, entry_id),
        )
    else:
        conn.execute(
            """
            UPDATE scratchpad_entries
            SET message = ?, last_edited_by = ?
            WHERE id = ?
            """,
            (new_message, last_edited_by, entry_id),
        )
    conn.commit()
    conn.close()


def delete_entry(entry_id: int) -> None:
    """
    Soft-delete an entry by id.
    """
    conn = get_conn()
    conn.execute(
        """
        UPDATE scratchpad_entries
        SET status = 'deleted', last_edited_by = 'system'
        WHERE id = ?
        """,
        (entry_id,),
    )
    conn.commit()
    conn.close()


# ============================================================================
# Display-index <-> PK bridge
# ============================================================================

def get_entry_by_display_index(
    project_name: str,
    section: str,
    display_index: int,
    subsection: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Map a user/LLM-facing index inside a section to the actual DB row.

    This is the crucial adapter that preserves your old workflow where the LLM
    says "edit Notes index 2".
    """
    # normalize section first (so "files" → ("Project Resources", "Files"))
    norm_section, norm_subsection = _normalize_section_and_subsection(section, subsection)

    conn = get_conn()
    if norm_section == "Project Resources":
        rows = conn.execute(
            """
            SELECT * FROM scratchpad_entries
            WHERE project_name = ?
              AND section = 'Project Resources'
              AND status = 'active'
            ORDER BY sort_order ASC, datetime(created_at) ASC, id ASC
            """,
            (project_name,),
        ).fetchall()

        # filter by subsection in Python to keep display order *within bucket*
        filtered: List[sqlite3.Row] = []
        for r in rows:
            sub = r["subsection"] or "Other"
            if sub == norm_subsection:
                filtered.append(r)
        rows = filtered
    else:
        rows = conn.execute(
            """
            SELECT * FROM scratchpad_entries
            WHERE project_name = ?
              AND section = ?
              AND status = 'active'
            ORDER BY sort_order ASC, datetime(created_at) ASC, id ASC
            """,
            (project_name, norm_section),
        ).fetchall()
    conn.close()

    if display_index < 0 or display_index >= len(rows):
        return None

    return _row_to_public_dict(rows[display_index])


def update_entry_by_display_index(
    project_name: str,
    section: str,
    display_index: int,
    new_message: str,
    new_confidence: Optional[int] = None,
    subsection: Optional[str] = None,
    new_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update an entry by its human/LLM-facing index.

    Returns True if an entry was updated, False if the index was invalid.
    """
    row = get_entry_by_display_index(
        project_name, section, display_index, subsection=subsection
    )
    if not row:
        return False
    update_entry(row["id"], new_message, new_confidence, new_metadata)
    return True


def delete_entry_by_display_index(
    project_name: str,
    section: str,
    display_index: int,
    subsection: Optional[str] = None,
) -> bool:
    """
    Delete an entry by its human/LLM-facing index.

    Returns True if deleted, False if index out of range.
    """
    row = get_entry_by_display_index(
        project_name, section, display_index, subsection=subsection
    )
    if not row:
        return False
    delete_entry(row["id"])
    return True