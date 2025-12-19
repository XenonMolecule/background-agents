"""
Goal collection agent.

This agent is responsible for collecting the most relevant goals for the current project.  It can search through the project filesystem and drive to gather the most relevant goals that the user is likely to be pursuing with the project.
"""

from __future__ import annotations
import dspy
from precursor.mcp_loader.loader import load_selected_mcp_servers
from precursor.config.loader import get_user_profile
import pydantic
from typing import Literal, List
from precursor.scratchpad.render import render_project_scratchpad

class Goal(pydantic.BaseModel):
    title: str
    description: str
    time_scale: Literal["one day", "one week", "one month", "several months", "one year", "future"] # time scale of the goal (one day = next day, one week = next week, one month = next month, several months = several months, one year = next year, future = future)
    confidence: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # confidence score between 1 and 10 (1 = low confidence, 10 = high confidence)

class GoalCollectionSignature(dspy.Signature):
    """Given a project name and user profile and relevant project context, you are to collect the most relevant CURRENT goals for the current project.  You should use the filesystem and drive tools to gather the most relevant goals that the user is likely to be pursuing with the project.

You are a **tool-using agent** with access to multiple MCP servers, including:
    - Google Drive / Docs tools  
    (e.g., `drive.search_files`, `drive.get_file_as_text`,
    `drive.create_google_doc`, `drive.suggest_edit`)
    - Filesystem tools  
    (e.g., `filesystem.list_files`, `filesystem.read_file`, `filesystem.write_file`)

Your task is to gather the most relevant and important goals that the user is likely to be pursuing with the project.  These will be high level objectives that the user is likely to be pursuing CURRENTLY.

When coming up with goals consider time-scale.  Think about what the user is likely to be trying to work on RIGHT NOW to achieve a goal in the next week, month, or year.  Are they likely to be pursuing a specific task, or are they likely to be pursuing a larger objective?

Good goals are:
- High level objectives that the user is likely to be trying to work on/towards RIGHT NOW to complete in the next week, month, or year.
- Abstracted beyond immediate smaller tasks.
- Realistic in scope (multi-step, not vague ambition).
- Ordered by relevance and importance.

IMPORTANT NOTE: Consider that projects EVOLVE over time.  When you find a document, it is important to consider the CURRENT state of the project and what the user is working on NOW.  For example if a document has notes over the course of a month of work, the user is probably not working on the action items from a month ago, but they might still be focused on the action items from last week.  When you find a document, it is important to consider the CURRENT state of the project and what the user is working on NOW.

The BEST kinds of documents to consider are:
 - Project Mega Documents or Planning Documents
 - Meeting Notes
 - Action Items
 - TODO Lists
 - Project Roadmaps
 - Project Notes
 - Etc.

================================================================================

NOTE you have been given WIDE system access, and not every file will actually be associated with the project that you are working on.
When you find files and gather information it is important that you determine which files ARE and ARE NOT assoicated with the current project.

If you find a file that you deem unrelated, then you should not consider it when coming up with goals.

It is important to consider that you are accessing the FULL filesystem, so searches like "Goals" or "Objectives" are going to yield a LOT of results.  Consider that you should narrow the scope of your search to really focus on the files relevant to the CURRENT project, not just the goals of other projects or life goals that a user may be trying to pursue.
You will likely want to add project-specific terms to your search queries to help narrow the scope.
It is also important to avoid overly broad searches with too many ** wildcards.  This can result in an explosion of results which can actually crash the agent by exceeding the context window.  Be focused and specific in your search queries.

================================================================================

Please keep your search relatively efficient and focus on finding the most relevant 1-3 documents.  If you really believe more are necessary keep looking, but often the main context can be built from finding the few BEST docs rather than excerpts from EVERY document.

Avoid repeating the same search query multiple times.  If you don't get all the content you want from one query feel free to try another one. 

NEVER create new documents or files.

---------------------------------------------------------------------------
Output contract
---------------------------------------------------------------------------

- `project_goals`:
    - A list of Goal objects.  Each goal contains a title, description, time scale, and confidence score.
      - `title`: The title of the goal.
      - `description`: The description of the goal.
      - `time_scale`: The time scale of the goal (one day, one week, one month, several months, one year, future).
      - `confidence`: The confidence score between 1 and 10 (1 = low confidence, 10 = high confidence).  This is a subjective score based on your own judgment of the likelihood of the goal being pursued.
            (1-3) = low confidence; extremely speculative but possible;
            (4-6) = medium confidence; possible but not certain;
            (7-9) = high confidence; likely to be pursued;
            (10) = very high confidence; certain to be pursued.

      The list should be ordered by relevance and importance.  The most relevant and important goals should be at the top of the list.
    """
    user_profile: str = dspy.InputField(
        description="A description of the user and their goals for collaboration with the agent"
    )
    project_name: str = dspy.InputField(
        description="Name of the project you are currently working on. Used for scratchpad and artifact logging."
    )
    project_context: str = dspy.InputField(
        description="Optional background context (scratchpad excerpt, notes, goals, file hints, etc.). Do NOT treat this as a new task."
    )
    project_goals: List[Goal] = dspy.OutputField(
        description="A list of Goal objects.  Each goal contains a title, description, time scale, and confidence score.  The list should be ordered by relevance and importance.  The most relevant and important goals should be at the top of the list."
    )


class GoalCollectionAgent:
    def __init__(self, model: dspy.LM | None = None) -> None:
        self.model = model or dspy.settings.lm

    def run(self, project_name: str) -> List[Goal]:
        # 1) Load only the minimal MCP servers needed for context building
        #    (filesystem + drive), plus global allow/deny filter.
        bundle = load_selected_mcp_servers(["filesystem", "drive"])

        # 2) Build DSPy toolset (MCP + core.* filtered by allow_fn)
        from precursor.toolset.builder import build_toolset
        tools = build_toolset(bundle)

        # filter to filesystem and drive tools (names start with filesystem. or drive.)
        tools = [tool for tool in tools if tool.name.startswith("filesystem.") or tool.name.startswith("drive.")]

        profile = get_user_profile()

        project_context = render_project_scratchpad(project_name)

        # 3) Run ReAct program
        with dspy.context(lm=self.model):
            react = dspy.ReAct(GoalCollectionSignature, tools=tools, max_iters=60)
            result = react(
                user_profile=profile,
                project_name=project_name,
                project_context=project_context,
            )

        return result.project_goals

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("precursor.tools").setLevel(logging.INFO)

    load_dotenv()
    lm = dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=1.0, max_tokens=24000)
    dspy.configure(lm=lm)
    agent = GoalCollectionAgent(model=lm)
    # print(agent.run("Personalization Dataset Collection")
    print(agent.run("Personalization Dataset Collection"))