from __future__ import annotations


def test_render_skips_invisible_system_messages(scratchpad_test_env):
    from precursor.messaging import render as mr
    from precursor.messaging import store as ms

    project = "Test Project Alpha"
    ms.init_db()

    ms.add_message(project, role="user", message="u1")
    ms.add_message(project, role="system", message="s1", visible_to_user=False)
    ms.add_message(project, role="agent", message="a1")

    txt_user = mr.render_project_conversation_markdown(project, limit=50, audience="user")
    assert "system:" not in txt_user
    assert "s1" not in txt_user
    assert "user: u1" in txt_user
    assert "agent: a1" in txt_user

    txt_agent = mr.render_project_conversation_markdown(project, limit=50, audience="agent")
    assert "system: s1" in txt_agent
    assert "user: u1" in txt_agent
    assert "agent: a1" in txt_agent


def test_render_includes_human_readable_timestamps(scratchpad_test_env):
    from precursor.messaging import render as mr
    from precursor.messaging import store as ms
    from precursor.scratchpad.store import get_conn
    from datetime import datetime, timezone

    project = "Test Project Alpha"
    ms.init_db()

    # Insert with a deterministic created_at.
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO conversation_messages
            (project_name, role, message, seen_by_user, visible_in_conversation, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project, "user", "u1", 1, 1, "2025-12-23 16:22:00"),
    )
    conn.commit()
    conn.close()

    txt = mr.render_project_conversation_markdown(project, limit=50, audience="user")
    # created_at is stored in UTC; renderer displays in local timezone.
    dt_local = datetime(2025, 12, 23, 16, 22, 0, tzinfo=timezone.utc).astimezone()
    day = dt_local.day
    year = dt_local.year
    weekday = mr._WEEKDAYS[dt_local.weekday()]
    month = mr._MONTHS[dt_local.month]
    hour24 = dt_local.hour
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    expected = f"{weekday} {month} {day}{mr._ordinal_suffix(day)}, {year} ({hour12}:{dt_local.minute:02d} {ampm})"
    assert expected in txt
    assert "user: u1" in txt


def test_render_timestamp_ordinal_suffix_11th(scratchpad_test_env):
    from precursor.messaging import render as mr
    from precursor.messaging import store as ms
    from precursor.scratchpad.store import get_conn
    from datetime import datetime, timezone

    project = "Test Project Alpha"
    ms.init_db()

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO conversation_messages
            (project_name, role, message, seen_by_user, visible_in_conversation, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project, "agent", "a1", 0, 1, "2025-12-11 09:05:00"),
    )
    conn.commit()
    conn.close()

    txt = mr.render_project_conversation_markdown(project, limit=50, audience="agent")
    dt_local = datetime(2025, 12, 11, 9, 5, 0, tzinfo=timezone.utc).astimezone()
    day = dt_local.day
    year = dt_local.year
    weekday = mr._WEEKDAYS[dt_local.weekday()]
    month = mr._MONTHS[dt_local.month]
    hour24 = dt_local.hour
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    expected = f"{weekday} {month} {day}{mr._ordinal_suffix(day)}, {year} ({hour12}:{dt_local.minute:02d} {ampm})"
    assert expected in txt
    assert "agent: a1" in txt


def test_trash_conversation_hides_messages_from_user_and_agent(scratchpad_test_env):
    from precursor.messaging import render as mr
    from precursor.messaging import store as ms

    project = "Test Project Alpha"
    ms.init_db()

    ms.add_message(project, role="user", message="u1")
    ms.add_message(project, role="system", message="s1", visible_to_user=False)
    ms.add_message(project, role="agent", message="a1")

    # Trash everything (soft-delete).
    updated = ms.trash_project_conversation(project)
    assert updated >= 3

    # User render should be empty.
    txt_user = mr.render_project_conversation_markdown(project, limit=50, audience="user")
    assert "(no messages)" in txt_user
    assert "u1" not in txt_user
    assert "a1" not in txt_user
    assert "s1" not in txt_user

    # Agent render should also be empty (even though it normally includes system).
    txt_agent = mr.render_project_conversation_markdown(project, limit=50, audience="agent")
    assert "(no messages)" in txt_agent
    assert "u1" not in txt_agent
    assert "a1" not in txt_agent
    assert "s1" not in txt_agent

