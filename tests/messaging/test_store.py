from __future__ import annotations


def test_add_and_list_messages_order(scratchpad_test_env):
    from precursor.messaging import store as mstore

    project = "Test Project Alpha"

    mstore.init_db()
    mstore.add_message(project, role="user", message="hello from user")
    mstore.add_message(project, role="agent", message="hello from agent")
    mstore.add_message(project, role="system", message="internal trigger", visible_to_user=False)

    rows = mstore.list_messages(project)
    assert len(rows) == 3
    assert rows[0]["role"] == "user"
    assert rows[0]["message"] == "hello from user"
    assert rows[0]["seen_by_user"] is True
    assert rows[1]["role"] == "agent"
    assert rows[1]["message"] == "hello from agent"
    assert rows[1]["seen_by_user"] is False
    assert rows[2]["role"] == "system"
    assert rows[2]["visible_to_user"] is False


def test_unseen_count_and_mark_seen(scratchpad_test_env):
    from precursor.messaging import store as mstore

    project = "Test Project Alpha"

    mstore.init_db()
    id1 = mstore.add_message(project, role="agent", message="a1")
    id2 = mstore.add_message(project, role="agent", message="a2")
    _id3 = mstore.add_message(project, role="user", message="u1")
    _id4 = mstore.add_message(project, role="system", message="s1", visible_to_user=False)

    assert mstore.count_unseen_by_user(project) == 2

    updated = mstore.mark_seen_by_user(project, up_to_id=id1)
    assert updated == 1
    assert mstore.count_unseen_by_user(project) == 1

    updated2 = mstore.mark_seen_by_user(project)
    assert updated2 == 1
    assert mstore.count_unseen_by_user(project) == 0

    rows = mstore.list_messages(project)
    by_id = {r["id"]: r for r in rows}
    assert by_id[id1]["seen_by_user"] is True
    assert by_id[id2]["seen_by_user"] is True


