from __future__ import annotations

def test_interviewer_agent_excludes_write_capable_mcp_tools(monkeypatch):
    """
    The interviewer agent should not be able to modify local files or Google Docs.
    This test targets the dependency-light filter helper (so it does not require
    DSPy or MCP runtime dependencies).
    """
    from precursor.agents.utils.tool_filters import filter_interviewer_mcp_tools

    class FakeTool:
        def __init__(self, name: str):
            self.name = name

    tools = [
        FakeTool("filesystem.read_file"),
        FakeTool("filesystem.list_files"),
        FakeTool("filesystem.write_file"),
        FakeTool("filesystem.edit_file"),
        FakeTool("filesystem.create_directory"),
        FakeTool("filesystem.move_file"),
        FakeTool("drive.search_files"),
        FakeTool("drive.get_file_as_text"),
        FakeTool("drive.create_google_doc"),
        FakeTool("drive.suggest_edit"),
        FakeTool("core.search_folders_fast"),  # should be removed by namespace restriction
        FakeTool("other.anything"),            # should be removed by namespace restriction
    ]

    filtered = filter_interviewer_mcp_tools(tools)
    tool_names = {t.name for t in filtered}

    excluded = {
        "drive.create_google_doc",
        "drive.suggest_edit",
        "filesystem.write_file",
        "filesystem.edit_file",
        "filesystem.create_directory",
        "filesystem.move_file",
    }
    assert excluded.isdisjoint(tool_names), f"Excluded tools still present: {excluded & tool_names}"

    # Sanity: some read-only tools should still be present.
    assert "filesystem.read_file" in tool_names
    assert "filesystem.list_files" in tool_names
    assert "drive.search_files" in tool_names
    assert "drive.get_file_as_text" in tool_names


