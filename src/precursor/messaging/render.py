"""
Render conversation history for a project into a compact, LLM-friendly format.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from precursor.messaging import store


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
        if not role:
            role = "unknown"
        lines.append(f"[{i}] {role}: {msg}")

    return "\n".join(lines).rstrip() + "\n"


