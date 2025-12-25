# tests/scratchpad/test_render.py
from __future__ import annotations


def test_render_simple_scratchpad(scratchpad_test_env):
    import precursor.scratchpad.store as store
    import precursor.scratchpad.render as render

    project = "Test Project Alpha"
    store.init_db()

    store.add_entry(project, "Ongoing Objectives", "Finish demo for seminar", confidence=9)
    store.add_entry(project, "Notes", "User is reviewing layout.tsx", confidence=4)
    store.add_entry(
        project,
        "Project Resources",
        "test-repo (uri: https://example.com/repo.git)",
        confidence=8,
        subsection="Repos",
    )

    text = render.render_project_scratchpad(project)

    # header shape
    assert "# Test Project Alpha" in text
    assert "## Ongoing Objectives" in text
    assert "## Notes" in text
    assert "## Project Resources" in text
    assert "### Repos" in text
    assert "test-repo (uri: https://example.com/repo.git)" in text


def test_render_unknown_project_is_graceful(scratchpad_test_env):
    import precursor.scratchpad.render as render

    text = render.render_project_scratchpad("Does Not Exist")
    assert "(Project not found in config/projects.yaml)" in text


def test_render_all_sections_present_even_if_empty(scratchpad_test_env):
    import precursor.scratchpad.store as store
    import precursor.scratchpad.render as render

    project = "Test Project Alpha"
    store.init_db()

    # no entries
    text = render.render_project_scratchpad(project)

    assert "## Ongoing Objectives" in text
    assert "## Completed Objectives" in text
    assert "## Suggestions" in text
    assert "## Notes" in text
    assert "## Project Resources" in text
    # Next Steps is being phased out; renderer may omit it.


def test_render_ignores_metadata_content(scratchpad_test_env):
    import precursor.scratchpad.store as store
    import precursor.scratchpad.render as render

    project = "Test Project Alpha"
    store.init_db()

    # Add entry with hidden metadata; renderer should not include metadata values
    store.add_entry(
        project,
        "Notes",
        "Created release doc",
        7,
        metadata={"uri": "drive://abc", "long_summary": "Outlined results"},
    )

    text = render.render_project_scratchpad(project)
    assert "Created release doc" in text
    # Ensure metadata values are not rendered
    assert "drive://abc" not in text
    assert "Outlined results" not in text