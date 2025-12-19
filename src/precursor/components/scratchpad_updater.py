# components/scratchpad_updater.py
from __future__ import annotations

from typing import Optional, List, Dict, Any

import dspy

from precursor.scratchpad import render, store
from precursor.scratchpad.scratchpad_tools import (
    append_to_scratchpad,
    edit_in_scratchpad,
    remove_from_scratchpad,
    get_refreshed_scratchpad,
)

from precursor.config.loader import get_project_names
from precursor.config.loader import get_user_profile

import pydantic

class ProjectResource(pydantic.BaseModel):
    """Lightweight resource schema used by the extractor."""
    name: str
    description: Optional[str] = None
    uri: Optional[str] = None


class ExtractProjectResourcesSignature(dspy.Signature):
    """
Extract the **project resources** visible in the user’s current context and scratchpad.  

Your goal is to identify **files, folders, repositories, documents, or collaborators** that are *actually visible or referenced* in the current view.  
Each resource should include:
- a **name** (short label),  
- a **description** (optional, summarizing what it is or why it matters),  
- and a **resource identifier** such as a URL, file path, folder path, repository name, or collaborator email.

========================
Rules
========================
- Do **not** invent or speculate about unseen resources.  
- If nothing relevant is visible, return `NULL` or state that no resources were found.  
- Prefer high-level, persistent entities (repositories, shared docs, project folders) over ephemeral ones (temporary files, single lines).  

========================
How to Think Hierarchically
========================
- In a **code editor**, identify the repository or root project folder first.  
  - Then list notable files or directories underneath it.  
- In **cloud-based tools** (e.g., Google Docs, Overleaf, Notion), record:  
  - The document or workspace title,  
  - Its shareable link (if visible),  
  - And any collaborator names or emails shown in the interface or comment threads.  
  - People leaving comments, edits, or suggestions are likely collaborators — record them.  
- In **communication or coordination views**, list shared workspaces or people explicitly involved.

========================
Heuristic Questions
========================
Ask yourself:
- What would another person need to locate or resume this project later?  
- Which entities define *where*, *in what medium*, and *with whom* the work is happening?

========================
Special Emphasis
========================
If the context resembles a workspace or editor:
- Always capture the repository, folder, or workspace name (shown in the title bar or root path).  
- Then include relevant files, cloud documents, and collaborators.  
- Repository, document, and collaborator identifiers are *critical* for long-term traceability.
"""
    # we keep these general (str) so this file doesn’t have to import TRUE_PROJECTS
    current_project_name: str = dspy.InputField(
        description="The project we think the user is currently working on."
    )
    current_project_scratchpad: str = dspy.InputField(
        description="Full rendered scratchpad for the current project."
    )
    user_context: str = dspy.InputField(
        description="What the user is currently doing / looking at / working on."
    )
    # we re-add the screenshot input you originally had
    current_screenshot: dspy.Image = dspy.InputField(
        description="The screenshot of the user's current workspace (use to detect repos, docs, collaborators)."
    )
    # we no longer force user_profile here — the extractor should stay unopinionated
    project_resources: List[ProjectResource] = dspy.OutputField(
        description="Up to ~10 resources actually seen in the context/screenshot and not obviously already in the scratchpad."
    )


class ExtractProjectResources(dspy.Module):
    """Thin wrapper around the extractor signature."""
    def __init__(self) -> None:
        self.extract = dspy.ChainOfThought(ExtractProjectResourcesSignature)

    def forward(
        self,
        current_project_name: str,
        current_project_scratchpad: str,
        user_context: str,
        current_screenshot: dspy.Image,
    ) -> List[Dict[str, Any]]:
        res = self.extract(
            current_project_name=current_project_name,
            current_project_scratchpad=current_project_scratchpad,
            user_context=user_context,
            current_screenshot=current_screenshot,
        )
        # convert to simple dicts for passing into the editor as candidates
        return [
            {"name": r.name, "description": r.description, "uri": r.uri}
            for r in res.project_resources
        ]


class EditProjectScratchpadSignature(dspy.Signature):
    """
Edit the project scratchpad based on new information.  
The scratchpad is an ongoing, structured log of the user’s progress and context for this project.

========================
Guidelines by Section
========================
- **Ongoing Objectives** → Goals currently being worked on. Add only when there is clear evidence of active progress.  
- **Completed Objectives** → Tasks that appear finished or replaced. Move them here when appropriate.  
- **Suggestions** → Helpful ideas, tools, or workflow improvements worth trying later.  
- **Notes** → Observations, context, or clarifications that help interpret other sections.  
- **Project Resources** → Tangible items such as Files, Repos, Folders, Core Collaborators, or Other.  
  - This section is *critical* for reconstructing the project later.  
  - Include repository names, file paths, collaborator emails, or shared doc links when visible.  
  - Prefer higher-level resources over small files. Mark uncertain URIs as guesses.  
- **Next Steps** → Specific, *constructive*, and *autonomously actionable* tasks that a background agent could complete or prepare without explicit approval.  
  - These should never commit the user socially (e.g., no scheduling meetings or sending messages), but they **should** take creative, technical, or organizational initiative.  
  - Great next steps are the ones that save future effort or push the project meaningfully forward — not just clerical tasks.  
  - Examples of strong next steps include:  
    - “Refactor `data_loader.py` to reduce startup latency by caching preprocessed inputs.”  
    - “Turn the recent model-comparison results into a one-slide summary for tomorrow’s check-in.”  
    - “Write a concise README section explaining how to reproduce the latest experiment.”  
    - “Add an evaluation script to benchmark response time across different models.”  
    - “Integrate the new paper citations into `related_work.tex`.”  
    - “Create a quick plot comparing the last three experiment runs in `results.csv`.”  
    - “Summarize insights from yesterday’s discussion into a ‘Future Work’ paragraph.”  
    - “Identify which dataset subsets contribute most to variance in the current metric.”  
    - “Outline a short plan for automating the recurring data-upload process.”  
  - Focus on steps that increase clarity, reduce friction, or meaningfully advance the research or engineering process — ideally things that a human would appreciate having ‘already done.’

========================
Confidence Management (Important; Please Read Carefully)
========================
- Every entry should include a **confidence score** from 0–10.  
- **Start low (1–3)** when adding new items — this reflects an initial observation.  
- If you later see the same fact or behavior again, **update confidence upward** (e.g., 5–7).  
- Only mark high confidence (8–10) when a pattern has been repeatedly confirmed.  
- This gradual reinforcement helps the agent learn which propositions are stable.
- It is also possible that we are updating the incorrect project scratchpad.  This is why it is so important to start with low confidence because it allows us to later remove low confident/incorrect entries.

========================
General Guidance
========================
- Keep entries concise and avoid duplicates.  
- Aim for diversity across sections when updating.  
- If no new information is available, it’s fine to only add a short Note or even skip editing entirely.
- Move outdated or finished Ongoing Objectives to Completed Objectives as evidence builds.  
- **Add only one proposition per edit call.** For multiple changes, call the tools separately.  


========================
Keeping it clean
========================
- The scratchpad can often grow a bit cluttered with all the contents over time.
- It is useful to periodically clean up the scratchpad by removing outdated or finished objectives, suggestions, notes, and next steps.
- Additionally if there are many files that seem to be irrelevant, you should remove them from the scratchpad.
- Sometimes several files in the same folder are added to the scratchpad because they were individually relevant, but when considered together it makes more sense to consolidate to mention them as a single folder.
- Focus on reducing clutter and redundancy in the scratchpad, and KEEPING it UP TO DATE!

========================
When Adding Project Resources
========================
- Prioritize high-level, persistent entities (repos, docs, folders, collaborators).  
- Include links (URLs, file paths, or emails) when known.  
- Never fabricate URIs — explicitly note if uncertain.  
"""
    # restore your original, richer context
    current_project_name: str = dspy.InputField(
        description="The name of the project that the user is currently working on (may be inaccurate)."
    )
    current_project_scratchpad: str = dspy.InputField(
        description="The current rendered project scratchpad."
    )
    speculated_current_objectives: List[str] = dspy.InputField(
        description="Objectives we think the user is currently working on (may be inaccurate)."
    )
    speculated_former_objectives: List[str] = dspy.InputField(
        description="Objectives we think the user worked on recently (may be inaccurate)."
    )
    calendar_events: List[str] = dspy.InputField(
        description="Upcoming or recent calendar events that *might* relate to this project."
    )
    full_project_list: List[str] = dspy.InputField(
        description="All projects the user is tracking; may be useful for disambiguation."
    )
    user_context: str = dspy.InputField(
        description="Short description of what the user is currently doing/looking at."
    )
    # candidates from the extractor (stringified list) — LLM decides to add or not
    potential_resources: str = dspy.InputField(
        description="Candidate project resources detected from context/screenshot. Decide which, if any, to add."
    )
    current_screenshot: dspy.Image = dspy.InputField(
        description="Screenshot of the user's current workspace (for layout/context cues)."
    )
    # we can optionally pass user_profile too, since you added config/user.yaml
    user_profile: Optional[str] = dspy.InputField(
        description="Optional high-level user profile/preferences for better suggestions.",
        default=None,
    )
    summary_of_edits: str = dspy.OutputField(
        description="A short natural-language summary of the edits you made to the scratchpad via tool calls."
    )


class ScratchpadUpdater(dspy.Module):
    """
    LLM-driven scratchpad updater.

    Responsibilities:
    - initialize the backing store (SQLite in user_data_dir via store.init_db())
    - (optionally) run resource extraction to get candidate resources from the current view
    - pass those candidates, plus all the rich context, into a DSPy ReAct editor
    - let the editor decide which of the scratchpad tools to call:
        • append_to_scratchpad(...)
        • edit_in_scratchpad(...)
        • remove_from_scratchpad(...)
        • get_refreshed_scratchpad(...)
    - return the summary + refreshed scratchpad text

    NOTE: we do NOT auto-write extracted resources anymore. We only supply them as
    potential_resources, and the LLM decides whether to add them. This preserves
    your original workflow.
    """

    def __init__(self, run_resource_extraction: bool = True) -> None:
        self.run_resource_extraction = run_resource_extraction
        self.editor = dspy.ReAct(
            EditProjectScratchpadSignature,
            tools=[
                get_refreshed_scratchpad,
                append_to_scratchpad,
                edit_in_scratchpad,
                remove_from_scratchpad,
            ],
            max_iters=20,
        )
        self.resource_extractor = (
            ExtractProjectResources() if run_resource_extraction else None
        )

    def forward(
        self,
        project_name: str,
        user_context: str,
        current_screenshot: dspy.Image,
        *,
        user_profile: Optional[str] = None,
        current_scratchpad: Optional[str] = None,
        speculated_current_objectives: Optional[List[str]] = None,
        speculated_former_objectives: Optional[List[str]] = None,
        calendar_events: Optional[List[str]] = None,
        full_project_list: Optional[List[str]] = None,
    ):
        # make sure DB exists before any tool call
        store.init_db()

        # render current scratchpad text if not supplied
        pad = current_scratchpad or render.render_project_scratchpad(project_name)

        # by default, we pass empty candidates
        potential_resources_str = ""

        # optional resource extraction step (non-destructive)
        if self.run_resource_extraction and self.resource_extractor is not None:
            extracted = self.resource_extractor(
                current_project_name=project_name,
                current_project_scratchpad=pad,
                user_context=user_context,
                current_screenshot=current_screenshot,
            )
            # turn into a bullet-y string for the LLM
            lines: List[str] = []
            for r in extracted:
                parts = [r["name"]]
                if r.get("description"):
                    parts.append(r["description"])
                if r.get("uri"):
                    parts.append(f"(uri: {r['uri']})")
                lines.append(" - " + " - ".join(parts))
            potential_resources_str = "\n".join(lines)

        # call the editor with full context
        res = self.editor(
            current_project_name=project_name,
            current_project_scratchpad=pad,
            speculated_current_objectives=speculated_current_objectives or [],
            speculated_former_objectives=speculated_former_objectives or [],
            calendar_events=calendar_events or [],
            full_project_list=full_project_list or list(get_project_names(only_enabled=False)),
            user_context=user_context,
            potential_resources=potential_resources_str,
            current_screenshot=current_screenshot,
            user_profile=user_profile or get_user_profile(),
        )

        # re-render after edits
        refreshed = render.render_project_scratchpad(project_name)
        return res.summary_of_edits, refreshed