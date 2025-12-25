"""
Render conversation history for a project into a compact, LLM-friendly format.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from precursor.messaging import store


_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _ordinal_suffix(day: int) -> str:
    # 11th/12th/13th are special cases
    if 11 <= (day % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _format_human_timestamp(created_at: object) -> str:
    """
    Convert a SQLite `created_at` value into a very human-readable timestamp.

    Expected SQLite format: "YYYY-MM-DD HH:MM:SS" (CURRENT_TIMESTAMP).
    """
    if created_at is None:
        return "Unknown time"
    raw = str(created_at).strip()
    if not raw:
        return "Unknown time"

    # Python's fromisoformat supports both "T" and " " between date/time.
    try:
        dt_utc_naive = datetime.fromisoformat(raw)
    except Exception:
        # Best-effort fallback: surface raw value rather than dropping it.
        return raw

    # Storage best practice: treat SQLite CURRENT_TIMESTAMP as UTC, then display in
    # the local timezone of the running process.
    dt_local = dt_utc_naive.replace(tzinfo=timezone.utc).astimezone()

    day = dt_local.day
    year = dt_local.year
    # Locale-stable English names (strftime can vary by locale).
    weekday = _WEEKDAYS[dt_local.weekday()]
    month = _MONTHS[dt_local.month]
    # Locale-stable 12-hour clock.
    hour24 = dt_local.hour
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    time_part = f"{hour12}:{dt_local.minute:02d} {ampm}"
    return f"{weekday} {month} {day}{_ordinal_suffix(day)}, {year} ({time_part})"


def render_project_conversation_markdown(
    project_name: str,
    *,
    limit: Optional[int] = 200,
    audience: Literal["agent", "user"] = "agent",
) -> str:
    """
    Render a project's conversation as markdown suitable for passing into an agent.

    Output format
    -------------
    # Conversation: <project>
    [1] user: ...
    [2] agent: ...

    Notes
    -----
    - We intentionally do not include `seen_by_user` in the rendered history.
      That's a UI concern, not something the agent needs.
    - For audience="agent": include system messages (even if hidden from user UI).
    - For audience="user": exclude messages hidden from the user UI (e.g. system).
    - We use stable 1-based display indices for readability.
    """
    store.init_db()
    include_invisible = audience == "agent"
    rows = store.list_messages(project_name, limit=limit, include_invisible=include_invisible)

    lines: List[str] = [f"# Conversation: {project_name}", ""]
    if not rows:
        lines.append("(no messages)")
        return "\n".join(lines).rstrip() + "\n"

    for i, r in enumerate(rows, start=1):
        role = str(r.get("role") or "").strip()
        msg = str(r.get("message") or "").strip()
        ts = _format_human_timestamp(r.get("created_at"))
        if not role:
            role = "unknown"
        lines.append(f"[{i}] [{ts}] {role}: {msg}")

    return "\n".join(lines).rstrip() + "\n"


