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


