# precursor/core_tools/send_user_message_tool.py
"""
One-way user messaging core tool.

This tool is designed for *brief*, *non-interactive* communication from an agent
to a user. It is typically used at the **start of an interaction** (before tool calls)
to acknowledge the user's latest update and state what the agent plans to do next.

It may also be used later in the same interaction to confirm completed work,
but ONLY if those actions have already occurred in the current run.

IMPORTANT: This tool must NEVER be used to ask questions.
All questions must be asked via the agent's output field (e.g., next_clarifying_question)
and sent by the outer loop.
"""

from __future__ import annotations

from typing import Optional

from precursor.messaging import store as message_store


def send_user_message(project_name: str, message: str) -> None:
    """
    Send a short, one-way message to the user scoped to a project.

    Primary purpose
    ---------------
    This tool is used at the **start of an interaction** (before tool calls) to:
    - Acknowledge the user's most recent update, AND
    - Briefly state what you are about to do next (your plan).

    It may also be used later in the same interaction to confirm completed work,
    but ONLY if those actions have already occurred in the current run.

    Parameters
    ----------
    project_name : str
        Project context for this message. Must match an existing project name.
        Use this so the messaging system can route/log messages per project.
    message : str
        The message to send.

    Critical rules
    --------------
    - The message must be **no more than 2–3 sentences**.
    - **NEVER ask questions using this tool.** Questions must be returned via
      an output field such as `next_clarifying_question`.
    - Do NOT claim you updated files, searched documents, or edited the scratchpad
      unless you have already done so (via tool calls) in the current run.
      If you have not done it yet, phrase it as intent: “I’m going to…”, “Next I’ll…”.

    Good examples (pre-action / plan)
    ---------------------------------
    - "I’m going to review the scratchpad and your latest notes, then propose concrete next steps."
    - "Next I’ll check the project resources (files and Drive) and update the scratchpad if needed."
    - "I’ll reconcile this update with the current objectives and see what needs adjusting."

    Good examples (post-action / only if already done this run)
    ----------------------------------------------------------
    - "I’ve updated the scratchpad with the new objective and added a next step to draft the evaluation plan."
    - "I found the relevant document and added it under Project Resources; next I’ll outline the immediate next steps."

    Bad examples
    ------------
    - "I’ve updated the scratchpad..."  (if you have not actually done it yet)
    - Any message containing a question mark '?'

    Returns
    -------
    None
        This tool does not expect a response from the user.
    """
    msg = str(message or "").strip()
    if not msg:
        return None

    # Persist as an unread agent->user message for UI badge support.
    message_store.add_message(
        project_name=project_name,
        role="agent",
        message=msg,
        seen_by_user=False,
    )
    return None