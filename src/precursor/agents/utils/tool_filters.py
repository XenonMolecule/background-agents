"""
Agent tool filtering utilities.

These helpers are intentionally dependency-light so they can be unit tested
without importing heavy optional runtime dependencies (e.g., DSPy).
"""

from __future__ import annotations

from typing import Iterable, List, Set, TypeVar

T = TypeVar("T")


def filter_interviewer_mcp_tools(mcp_tools: Iterable[T]) -> List[T]:
    """
    Return a filtered list of MCP tools appropriate for the interviewer agent.

    Rules:
    - Only allow tool names in the `filesystem.*` or `drive.*` namespaces.
    - Exclude write/capability-escalation tools (local writes, doc creation/edits).

    The objects in `mcp_tools` are expected to have a `.name` attribute.
    """
    excluded: Set[str] = {
        "drive.create_google_doc",
        "drive.suggest_edit",
        "filesystem.write_file",
        "filesystem.edit_file",
        "filesystem.create_directory",
        "filesystem.move_file",
    }

    out: List[T] = []
    for t in mcp_tools:
        name = getattr(t, "name", "")
        if not (name.startswith("filesystem.") or name.startswith("drive.")):
            continue
        if name in excluded:
            continue
        out.append(t)
    return out


