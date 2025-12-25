from __future__ import annotations

from datetime import datetime, timezone

def test_pick_no_followup_ack_contains_no_question_mark():
    from precursor.agents.utils.no_followup_ack import NO_FOLLOWUP_ACKS

    # This message is explicitly meant to be a non-question acknowledgement.
    assert NO_FOLLOWUP_ACKS, "Expected at least one acknowledgement template."
    assert all("?" not in s for s in NO_FOLLOWUP_ACKS)


def test_is_recent_sqlite_utc_timestamp_within_window_true():
    from precursor.messaging.time_utils import is_recent_sqlite_utc_timestamp

    now = datetime(2025, 1, 1, 12, 15, 0, tzinfo=timezone.utc)
    # SQLite CURRENT_TIMESTAMP format
    created_at = "2025-01-01 12:05:00"
    assert is_recent_sqlite_utc_timestamp(created_at, now_utc=now) is True


def test_is_recent_sqlite_utc_timestamp_outside_window_false():
    from precursor.messaging.time_utils import is_recent_sqlite_utc_timestamp

    now = datetime(2025, 1, 1, 12, 16, 0, tzinfo=timezone.utc)
    created_at = "2025-01-01 12:00:00"
    assert is_recent_sqlite_utc_timestamp(created_at, now_utc=now) is False


