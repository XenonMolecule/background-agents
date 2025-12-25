"""
Time utilities for conversation/message timestamps.

SQLite `CURRENT_TIMESTAMP` yields a UTC timestamp like: "YYYY-MM-DD HH:MM:SS".
These helpers are intentionally dependency-light and tolerant of bad inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def is_recent_sqlite_utc_timestamp(
    created_at: object,
    *,
    now_utc: Optional[datetime] = None,
    window_minutes: float = 15.0,
) -> bool:
    """
    Return True if `created_at` is within the last `window_minutes` minutes.

    Parameters
    ----------
    created_at:
        Typically a SQLite `CURRENT_TIMESTAMP` string: "YYYY-MM-DD HH:MM:SS" (UTC).
    now_utc:
        Optional override for testing.
    window_minutes:
        Age threshold in minutes.
    """
    if created_at is None:
        return False

    raw = str(created_at).strip()
    if not raw:
        return False

    try:
        dt_utc_naive = datetime.fromisoformat(raw)
    except Exception:
        return False

    dt_utc = dt_utc_naive.replace(tzinfo=timezone.utc)
    now = now_utc or datetime.now(timezone.utc)
    age_seconds = (now - dt_utc).total_seconds()
    return 0 <= age_seconds <= (window_minutes * 60)


