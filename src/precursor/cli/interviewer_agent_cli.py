"""
CLI entrypoint for the HumanInTheLoopInterviewerAgent.

Behavior
--------
- Loads conversation history from the DB (excluding system/invisible messages).
- Runs the interviewer agent *only if* the most recent DB message is from:
  - user, or
  - system (system messages can trigger updates without user input)
- Persists the next clarifying question (if any) as an unread agent message and exits.

Example:
    python -m precursor.cli.interviewer_agent_cli \
        --project "AutoMetrics Release" \
        --model openai/gpt-5
"""

from __future__ import annotations

import argparse
import logging

import dspy
from dotenv import load_dotenv

from precursor.agents.interviewer_agent import HumanInTheLoopInterviewerAgent
from precursor.messaging import render as convo_render
from precursor.messaging import store as convo_store


load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HumanInTheLoopInterviewerAgent for a project")
    ap.add_argument("--project", required=True, help="Project name (must exist in projects.yaml)")
    ap.add_argument("--model", default="openai/gpt-5", help="DSPy model id (e.g., openai/gpt-5)")
    ap.add_argument(
        "--max-history",
        type=int,
        default=200,
        help="Max number of (visible) conversation messages to include in the markdown transcript",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Run even if the latest DB message is not from user/system (default: skip).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("precursor.tools").setLevel(logging.INFO)

    # Configure DSPy
    lm = dspy.LM(args.model, temperature=1.0, max_tokens=24000)
    dspy.configure(lm=lm)

    # Gate execution on the last message role.
    latest = convo_store.get_latest_message(args.project, include_invisible=True)
    if latest is None:
        logging.info("No messages found for project=%s; skipping.", args.project)
        return

    latest_role = str(latest.get("role") or "").strip().lower()
    if not args.force and latest_role not in ("user", "system"):
        logging.info("Latest message role=%s (not user/system); skipping.", latest_role or "(missing)")
        return

    # Render DB conversation transcript for the agent (system/invisible messages excluded).
    conversation_history_markdown = convo_render.render_project_conversation_markdown(
        args.project, limit=args.max_history, audience="agent"
    )

    agent = HumanInTheLoopInterviewerAgent(model=lm)
    next_q = agent.run(
        project_name=args.project,
        conversation_history_markdown=conversation_history_markdown,
    )

    if next_q:
        # Persist the question as the next agent message so the UI can display it.
        convo_store.add_message(
            project_name=args.project,
            role="agent",
            message=next_q,
            seen_by_user=False,
            visible_to_user=True,
        )
        print(next_q)


if __name__ == "__main__":
    main()


