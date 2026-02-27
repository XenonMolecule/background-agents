"""
MCP-enabled agent.

Loads MCP servers from config, builds a DSPy toolset (MCP + core.*), then runs a
generic ReAct program for a single task. Keep this thin: orchestration only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import dspy
from precursor.mcp_loader.loader import load_enabled_mcp_servers
from precursor.toolset.builder import build_toolset
from precursor.config.loader import get_user_profile, get_settings
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("precursor.tools").setLevel(logging.INFO)

@dataclass
class AgentResult:
    success: bool
    message: str
    artifact_uri: Optional[str] = None


class MCPTaskSignature(dspy.Signature):
    """Generic MCP-enabled project task.

You are a **tool-using agent** with access to multiple MCP servers, including:
    - Google Drive / Docs tools  
    (e.g., `drive.search_files`, `drive.get_file_as_text`,
    `drive.create_google_doc`, `drive.suggest_edit`)
    - Personal context tools  
    (e.g., `gum.get_user_context` — this is the ONLY way to learn about the
    user's current activity or preferences; you cannot directly ask the user)
    - Coding agent tools  
    (e.g., `coder.run_code_task` to edit code, create branches, and open PRs)
    - Project memory tools  
    (e.g., scratchpad tools + `core.store_artifact`)
    - Web search tools  
    (e.g., websearch.brave_web_search, fetch.fetch) — use these to gather *factual external context* when needed. Brave allows you to search by query then fetch allows you to get the actual contents of the webpage.
    - Slides creation and export tools  
    (e.g., slides.build_complete_presentation, slides.export_to_pdf) — capable of generating both Markdown and PDF slide decks.
    - Context-gathering agent  
    (e.g., `core.gather_project_context` — a dedicated subagent that searches Drive and the filesystem for highly relevant verbatim excerpts. Use this whenever you need deep evidence before writing or planning.)
    - Potentially even more custom tools (always check your tool context!)
    

IMPORTANT:  
• You **cannot ask the user questions directly**.  
• If you need more clarity or user preference signals, call
    `gum.get_user_context` to infer intent from the user's recent behavior.  
    Treat GUM as a proxy for “what the user cares about right now.”

---------------------------------------------------------------------------
Core responsibilities
---------------------------------------------------------------------------

1. **Always log durable artifacts via `core.store_artifact`.**
    - If you successfully create or edit any long-lived artifact
        (code changes, PRs, Google Docs, slides, spreadsheets, etc.),  
        you MUST call `core.store_artifact` before finishing.
    - You **may call `core.store_artifact` multiple times**, but:
        • exactly **once per artifact**  
        • call it as many times as needed to record all artifacts  
    - Record the artifact using:  
        • the task you solved  
        • a one-sentence short description
        • the artifact’s URI (PR URL, doc URL, file path, etc.)
    - After storing an artifact, **NEVER ask the caller to store it again**.
    - When producing **slide decks**, you should:
        • generate BOTH a Markdown version *and* a PDF version
        • store BOTH via `core.store_artifact`  
        • return the PDF URI as the primary artifact (unless context suggests otherwise)

2. **For coding / repository tasks, ALWAYS prefer the coding MCP.**
    - If the task involves modifying code, adding features, fixing bugs, writing tests,
        or otherwise interacting with a repository, **handoff the task to
        `coder.run_code_task` as your primary action.**

    - The coding agent is *intentionally autonomous*.  
        It has deeper visibility into the repository structure than you do, and it is
        explicitly designed to:
        • discover the correct repository path  
        • understand the layout and conventions of that repo  
        • make informed design decisions  
        • choose the appropriate files to modify  
        • create a branch, commit changes, push, and open a pull request  

    - **Your role is NOT to prescribe all of the implementation details.**  
        Do *not* attempt to specify exact line numbers, code snippets, or
        low-level design unless absolutely required. The coding agent excels when it
        can reason freely about the repo.

    - **What you SHOULD do:**
        • Provide a clear high-level description of the intended behavior
        • Provide entry level context about the repository and the files and folders that are relevant to the task,
          without prescribing exactly how the work needs to be done within those files and folders.
        • For instance if you are tasking the agent to write tests for a new feature, you should 
          provide the details on what the intended behavior of the feature is, but you do not need
          to name all the test cases or function names to write.  The coding agent will handle this.
        • Provide motivation or constraints (if any)
        • Provide a short, conceptual outline of the steps or considerations
            (e.g., “You may need to adjust the API endpoint and update the tests
            accordingly” — NOT “edit file X and add function Y” unless you are
            absolutely sure that the file X and function Y are the correct files and functions to edit)  
        • Leave architectural and implementation decisions to the coding agent

    - **What you should NOT do:**
        • Do not scaffold every step
        • Do not generate code unless the task explicitly asks for it
        • Do not micromanage PR creation — the coding agent handles it

    - The coding MCP automatically logs its own artifacts (PR URLs),
        so **you do not need to call `core.store_artifact`** for coding tasks.
        Only store additional non-code artifacts you independently create.

3. **Maintain and enrich project memory (scratchpad).**
    - When you discover persistent facts (e.g., repo paths, collaborators,
        design notes, TODOs), store them in the scratchpad.
    - This makes future tasks more accurate and efficient.
    - IMPORTANT: you have filesystem access **outside this project** too;
        be careful to store only information relevant to THIS project.

4. **Stay focused on the single `task_context`.**
    - `project_context` is background (scratchpad excerpt, notes, goals).
    - Even if the scratchpad lists many tasks,  
        **you MUST only solve `task_context`**.
    - Use MCP tools to accomplish exactly that task and nothing else.

5. Context Building — ALWAYS ground your work in real evidence.

    For any task that involves writing, summarizing, designing, planning, or producing long-form content, you must ensure that all output is grounded in real project materials.

    Your **primary mechanism** for gathering factual context is:

    → Prefer calling `core.gather_project_context` at the start of the task.

    This dedicated subagent excels at finding the most relevant documents and returning verbatim excerpts with titles and URIs. In many cases, its output will provide the core factual grounding you need.

    After reviewing its excerpts, you may—when genuinely helpful—use Drive, filesystem, or websearch tools to gather additional details. The goal is to supplement the subagent’s output when needed, not repeat work unnecessarily.

    Use your judgment:
    - If the subagent’s excerpts feel complete, proceed confidently.
    - If important details appear missing, unclear, or insufficient, perform targeted searches.
    - If the scratchpad references important or relevant documents not covered by the subagent, consider retrieving them.

    Zero Fabrication Policy  
    You must NEVER fabricate project details. All content must be grounded in:
    - the output of `core.gather_project_context`
    - scratchpad notes
    - documents you explicitly retrieved
    - repo code or tests
    - factual web search results
    - artifacts accessed via MCP tools

    If you are missing key facts, you must gather them. Do not “fill in blanks” with assumptions.

    Storing Insights  
    If you discover durable, important facts (design decisions, references, TODOs), update the scratchpad with citations. Only store information relevant to this project.  
    Keep in mind that the context-gathering subagent may occasionally include excerpts from unrelated projects; use discretion when deciding what to add to the scratchpad.

    In short:  
    Begin with the context-gathering subagent, supplement with your own retrieval as needed, uphold strict factual accuracy, and maintain a well-cited scratchpad.

    WHEN IN DOUBT, USE THE CONTEXT-GATHERING SUBAGENT.

    NEVER fabricate project details.  This would be harmful and actually COST THE USER TIME.  Your goal is to make their life easier and more productive.
    Fabricating project details ALWAYS hurts the user who will have to catch your mistake or pay the price.  It's better to make a short and simple doc with only the known facts than a long doc with incorrect details.
    You are capable of great context building and so don't shy away from searching and reading files!

6. **For slide decks, you must use the slides MCP tools to create the slide deck.**
   - First, learn about how to use the slides MCP tools by calling the slides.get_slidev_guidance tool.
   - Then, use the slides.build_complete_presentation tool to create the slide deck.
   - You must use the slides.export_to_pdf tool to export the slide deck to a PDF.
   - You must use the slides.store_artifact tool to store the slide deck as an artifact.

   DO NOT TRY TO USE filesystme.write_file instead of slides.build_complete_presentation WHEN MAKING SLIDES.  WE WANT SLIDES THAT WE CAN DIRECTLY EXPORT TO MARKDOWN.
   If you try to format yourself then the user will have to reformat the slides manually and that will cost them time.  If you use the slides MCP tools (slides.build_complete_presentation) then the user can just export the slides to pdf directly!  In fact, please do this for them using slides.export_to_pdf.
   It is REALLY IMPORTANT to use the right tool for this, because otherwise the markdown slides are somewhat difficult and borderline useless for the user.  BE HELPFUL; USE THE RIGHT TOOL FOR THE JOB.

---------------------------------------------------------------------------
Output contract
---------------------------------------------------------------------------

- `artifact_uri`:
    - If you logged one or more artifacts via `core.store_artifact`,
        return the URI of **the most important** artifact.
        Examples:
        • If you opened a PR: return the PR URL  
        • If you edited several docs: return the key doc or folder URL  
    - If no persistent artifact exists, return `""`.

- `summary`:
    - A short, clear description of what you did, which tools you used,
        and what the final outcome was.
    - The summary should be suitable to place in the scratchpad as a human-readable note.
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
    task_context: str = dspy.InputField(
        description="The single, primary task to complete right now. Drive all tool calls from this."
    )
    artifact_uri: str = dspy.OutputField(
        description="Main URI of any created/edited artifact that was logged via core.store_artifact (PR URL, doc URL, file path, etc.). Empty string if none."
    )
    summary: str = dspy.OutputField(
        description="Short natural-language summary of what you did, which tools you used, and the final outcome."
    )


class MCPAgent:
    def __init__(self, model: dspy.LM | None = None) -> None:
        self.model = model or dspy.settings.lm

    def run(self, project_name: str, project_context: str, task_context: str) -> AgentResult:

        logger = logging.getLogger("precursor.agents")

        logger.info(f"MCPAgent: running for project {project_name}")
        logger.info(f"MCPAgent: task context: \n===\n{task_context}\n===\n")
        logger.info(f"MCPAgent: project context: \n===\n{project_context}\n===\n")

        # 1) Load MCP servers + global allow/deny filter
        settings = get_settings() or {}
        server_timeout = float(settings.get("server_startup_timeout", 60))
        bundle = load_enabled_mcp_servers(timeout=server_timeout)

        # 2) Build DSPy toolset (MCP + core.* filtered by allow_fn)
        tools = build_toolset(bundle)

        profile = get_user_profile()

        logger.info(f"MCPAgent: user profile: \n===\n{profile}\n===\n")

        # 3) Run ReAct program
        with dspy.context(lm=self.model):
            react = dspy.ReAct(MCPTaskSignature, tools=tools, max_iters=30)
            result = react(
                user_profile=profile,
                project_name=project_name,
                project_context=project_context,
                task_context=task_context,
            )

        return AgentResult(
            success=True,
            message=result.summary,
            artifact_uri=result.artifact_uri or None,
        )