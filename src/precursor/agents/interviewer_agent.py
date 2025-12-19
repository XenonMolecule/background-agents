# components/hitl_interviewer.py
"""
Human-in-the-loop interviewer agent (single-file module).

- Updates a project scratchpad using scratchpad tools.
- Gathers context via filesystem + Google Drive MCP tools.
- Strongly prefers `search_folders_fast` over `filesystem.search_files` for locating folders/repos.
- Sends short acknowledgement/plan messages to the user via send_user_message(project_name, ...), NEVER questions.
- Returns at most ONE clarifying question via next_clarifying_question (or None).

Conversation history
--------------------
This module does NOT manage persistence; it expects the caller to provide a
markdown-formatted transcript via `conversation_history_markdown`.

For the default DB-backed implementation, see:
`precursor.cli.interviewer_agent_cli` + `precursor.messaging`.
"""

from __future__ import annotations

from typing import Optional
import dspy

from precursor.config.loader import get_user_profile

from precursor.scratchpad import store
from precursor.scratchpad.render import render_project_scratchpad
from precursor.scratchpad.scratchpad_tools import (
    append_to_scratchpad,
    edit_in_scratchpad,
    remove_from_scratchpad,
    get_refreshed_scratchpad,
)

from precursor.mcp_loader.loader import load_selected_mcp_servers
from precursor.toolset.builder import build_toolset

# Fast folder search core tool (preferred for locating repos/folders by name).
from precursor.core_tools.fast_find_tool import search_folders_fast

# One-way user messaging core tool (ack/plan only; never questions).
from precursor.core_tools.send_user_message_tool import send_user_message


class HumanInTheLoopInterviewerSignature(dspy.Signature):
    """
You are a user-in-the-loop interviewer.

Your role is to progressively establish shared understanding with the user about
the project’s goals, direction, and current state, while keeping the project
scratchpad accurate, current, and useful.

Primary responsibilities
------------------------
1) Maintain the project scratchpad using the scratchpad tools:
   - append_to_scratchpad
   - edit_in_scratchpad
   - remove_from_scratchpad
2) Gather relevant context from the local filesystem or Google Drive when helpful.
3) Communicate clearly with the user using send_user_message(project_name, ...) for
   acknowledgements and planned actions.
4) Ask clarifying questions ONLY when they meaningfully improve shared understanding,
   via next_clarifying_question.

Tool usage rules (important)
----------------------------
- If you need to locate a repository, project directory, or folder by name:
  **ALWAYS use `search_folders_fast` first.**
  This tool is significantly faster and more targeted than filesystem search.

- Avoid `filesystem.search_files` for folder discovery.
  It should only be used when:
  - You already know the approximate directory structure and need to enumerate files, or
  - You are working within a known directory and need to locate specific files by name.

- Use filesystem tools such as `list_files` or `read_file` only after the
  correct directory has been identified.

Communication rules
-------------------
- You MAY use send_user_message(project_name, ...) to acknowledge the user or state your plan.
- Messages sent via send_user_message must:
  - Be no more than 2–3 sentences.
  - Contain no questions.
  - Describe intent (“I’m going to…”, “Next I’ll…”) unless the action has already
    been completed in the current run.

- You must NEVER ask questions via send_user_message.
- All questions must be returned via next_clarifying_question.

Question discipline
-------------------
Questions are a high-leverage tool for establishing shared understanding.

Ask a clarifying question only when the answer would materially improve alignment
between you and the user about the project’s direction, goals, or current state.

Appropriate reasons to ask a question include (but are not limited to):
- Clarifying high-level project goals, scope, or direction.
- Resolving ambiguity about what you are working toward together.
- Discovering important files, folders, repositories, or resources that are not yet visible.
- Correcting, removing from, or cleaning up outdated or incorrect items in the scratchpad.
- Understanding which objectives are still active versus no longer relevant.
- Determining what kind of help or progress would be most valuable next.

Structure of questions
----------------------
- It is acceptable for a question to have multiple parts if they are tightly related
  and easy to answer together.
- Multi-part questions should feel like a single coherent thought, not a checklist.
- When asking about objectives, goals, or direction, **explicitly state your current
  understanding first** so the user can confirm or correct it.

Good examples:
- "I currently understand the main goal to be building a personalized reward model for
  preference data — is that accurate, and do you have a planning doc that outlines
  these objectives?"
- "It looks like the focus has been on evaluation metrics so far; is that still the
  primary objective, or has the direction shifted recently?"

Bad examples:
- "What are your goals? Where are your collaborators? What help do you want?"
- "Is the current objective still active?"  (without stating what you believe it is)

Restraint
---------
- Ask at most ONE clarifying question per run.
- Prefer questions that guide multiple future actions over narrow, tactical details.
- If you can proceed with a reasonable assumption without risking misalignment,
  proceed without asking.

General constraints
-------------------
- You may make multiple tool calls per run.
- You may make multiple scratchpad edits per run.
- Never fabricate facts, files, folders, or collaborators.

Goal
----
The goal of asking questions is not merely to unblock actions,
but to converge on a shared, durable understanding of where the project is headed
and what you are working on together.
"""

    user_profile: str = dspy.InputField(
        description="High-level user preferences and collaboration goals."
    )
    project_name: str = dspy.InputField(
        description="Name of the project being discussed."
    )
    project_scratchpad: str = dspy.InputField(
        description="Rendered scratchpad text for the current project."
    )
    conversation_history: str = dspy.InputField(
        description="Markdown-formatted transcript of the conversation so far."
    )

    next_clarifying_question: Optional[str] = dspy.OutputField(
        description=(
            "A single clarifying question to ask the user next, "
            "or None if no clarification is needed to proceed."
        )
    )


class HumanInTheLoopInterviewerAgent:
    """
    Human-in-the-loop interviewer agent.

    Integration notes:
    - Caller is responsible for loading and persisting conversation history (JSON on disk).
    - This agent expects conversation_history as a markdown string.
    - Caller decides whether/when to send the returned next_clarifying_question to the user.
    """

    def __init__(self, model: dspy.LM | None = None) -> None:
        self.model = model or dspy.settings.lm

        # Load MCP tools (filesystem + drive).
        bundle = load_selected_mcp_servers(["filesystem", "drive"])
        mcp_tools = build_toolset(bundle)

        # Keep only filesystem.* and drive.* tool names from MCP toolset.
        mcp_tools = [
            t
            for t in mcp_tools
            if t.name.startswith("filesystem.") or t.name.startswith("drive.")
        ]

        # Assemble full tool list:
        # - fast folder search (preferred for locating repos/folders)
        # - scratchpad tools
        # - send_user_message(project_name, ...) for acknowledgement/plan (never questions)
        self.tools = mcp_tools + [
            search_folders_fast,
            get_refreshed_scratchpad,
            append_to_scratchpad,
            edit_in_scratchpad,
            remove_from_scratchpad,
            send_user_message,
        ]

        self.react = dspy.ReAct(
            HumanInTheLoopInterviewerSignature,
            tools=self.tools,
            max_iters=25,
        )

    def run(
        self,
        project_name: str,
        conversation_history_markdown: str,
        *,
        user_profile: Optional[str] = None,
        project_scratchpad: Optional[str] = None,
    ) -> Optional[str]:
        """
        Run one interviewer turn.

        Parameters
        ----------
        project_name : str
            Project to update.
        conversation_history_markdown : str
            Markdown transcript of conversation so far. Caller loads from JSON and formats.
        user_profile : str, optional
            Override loaded user profile if desired.
        project_scratchpad : str, optional
            Override scratchpad text if already loaded; otherwise rendered from store.

        Returns
        -------
        Optional[str]
            next_clarifying_question (one question) or None.
        """
        # Ensure scratchpad DB exists before any tool calls.
        store.init_db()

        profile = user_profile or get_user_profile()
        scratchpad = project_scratchpad or render_project_scratchpad(project_name)

        with dspy.context(lm=self.model):
            res = self.react(
                user_profile=profile,
                project_name=project_name,
                project_scratchpad=scratchpad,
                conversation_history=conversation_history_markdown,
            )

        return res.next_clarifying_question


# (No CLI entrypoint in this module. Use `python -m precursor.cli.interviewer_agent_cli`.)