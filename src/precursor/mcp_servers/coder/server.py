"""
MCP server: coder

Exposes one tool `run_code_task(project_name, task)` that mirrors the old CodeAgent:
  • Renders the project's scratchpad (no project_context arg needed).
  • Uses your existing repo-finding stack (dspy Identify/Select + find_folders).
  • Runs OpenHands headlessly to complete the task and create a PR.
  • Summarizes the trajectory (short + long) via a DSPy call.
  • Logs a concise artifact line in the scratchpad with hidden metadata.

Keep docstrings exactly as authored; DSPy/tool callers rely on them.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import json
import platform
import subprocess
import time

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

# Match your original import structure (you said these are available):
from precursor.mcp_servers.coder.fast_find import find_folders
from precursor.mcp_servers.coder.get_git_repo import get_repo_full_name

# Scratchpad rendering + artifact logging (your existing modules)
from precursor.scratchpad import render as scratchpad_render
from precursor.core_tools.artifacts import store_artifact

# OpenHands runner colocated with this server package per your note
from precursor.mcp_servers.coder.openhands_tool import run_openhands_task_with_pr_async

# dspy + your minimal agent scaffolding (kept as in your old file)
import dspy


mcp = FastMCP("coder")


# ---------------------------------------------------------------------------
# Docker readiness helpers
# ---------------------------------------------------------------------------
def _docker_info_ok() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _start_docker_background() -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            # Launch Docker Desktop without stealing focus
            subprocess.Popen(
                ["/usr/bin/open", "-g", "-a", "Docker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Linux":
            # Best-effort attempts without sudo (environment may not allow)
            for cmd in (
                ["systemctl", "--user", "start", "docker"],
                ["systemctl", "start", "docker"],
                ["service", "docker", "start"],
            ):
                try:
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5,
                    )
                except subprocess.SubprocessError:
                    pass
        elif system == "Windows":
            # Start Docker Desktop; ignore errors if not installed
            subprocess.Popen(
                'cmd /c start "" "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        # Swallow errors; readiness will be checked via _docker_info_ok
        pass


def ensure_docker_running(timeout_seconds: int = 180) -> bool:
    """
    Ensure Docker engine is running. Attempt to start it in the background
    if it is not yet running, then wait until ready or timeout.
    """
    if _docker_info_ok():
        return True

    _start_docker_background()

    deadline = time.time() + max(1, timeout_seconds)
    # Initial short delay to give the daemon a moment to spawn
    time.sleep(2)
    backoff = 1.0

    while time.time() < deadline:
        if _docker_info_ok():
            return True
        time.sleep(min(5.0, backoff))
        backoff = min(5.0, backoff * 1.5)

    return _docker_info_ok()


# ---------------------------------------------------------------------------
# DSPy agent code (unchanged except: project_context now comes from scratchpad)
# ---------------------------------------------------------------------------

class IdentifyRepositoryName(dspy.Signature):
    """Identify the possible name of the repository that we are working on.  Take in the project context and return a list of possible repository names.  If the project context contains a repository name, return that first."""
    project_name: str = dspy.InputField(description="The name of the project that we are working on")
    task_context: str = dspy.InputField(description="Detailed context about the task that we are working on.")
    project_context: str = dspy.InputField(description="Detailed context about the project that we are working on.  The Files section may contain repository names.")
    potential_repository_names: list[str] = dspy.OutputField(description="A list of possible repository names.  Only return repository names that are likely to be the repository that we are working on.  If the repo name seems like a guess or has spaces you should suggest variations of the name to help identify the true repository name as a folder name (the true repo name is unlikely to have spaces).  Variations may include removing spaces, adding hyphens, adding underscores, lowercasing, etc.")


class SelectRepositoryName(dspy.Signature):
    """Select the repository name that we are working on. Given a list of actual files and folders on the local machine, select the single path that is most likely to be the repository that we are working on.  Note that if you find subfiles of the repository name, you should select the parent folder of the subfiles as the repository path."""
    project_name: str = dspy.InputField(description="The name of the project that we are working on")
    task_context: str = dspy.InputField(description="Detailed context about the task that we are working on.")
    project_context: str = dspy.InputField(description="Detailed context about the project that we are working on.  The Files section may contain repository names.")
    actual_files_and_folders: list[str] = dspy.InputField(description="A list of actual files and folders on the local machine.")
    repository_path: str = dspy.OutputField(description="The single path that is most likely to be the repository that we are working on.  Should be a global path to the repository on the local machine.  If you have low confidence in the repository path, return None.")


class FindRepository(dspy.Module):
    def __init__(self):
        self.identify_repository_name = dspy.ChainOfThought(IdentifyRepositoryName)
        self.select_repository_name = dspy.ChainOfThought(SelectRepositoryName)

    def forward(self, project_name: str, project_context: str, task_context: str) -> str:
        potential_repository_names = self.identify_repository_name(
            project_name=project_name,
            task_context=task_context,
            project_context=project_context
        ).potential_repository_names

        actual_files_and_folders: list[str] = []
        for potential_repository_name in potential_repository_names:
            # Use your existing fast_find helper exactly as before
            hits = find_folders(potential_repository_name, max_results=5, timeout=5, backend_timeout=5)
            actual_files_and_folders.extend([str(p) for p in hits])

        repository_path = self.select_repository_name(
            project_name=project_name,
            task_context=task_context,
            project_context=project_context,
            actual_files_and_folders=actual_files_and_folders
        ).repository_path
        return repository_path


# ---------------------------------------------------------------------------
# NEW: Trajectory summarizer (one DSPy call)
# ---------------------------------------------------------------------------

class SummarizeTrajectory(dspy.Signature):
    """Given project and task context plus the **true contents of the OpenHands trajectory file**, produce:
- short_summary: a single sentence (layperson-friendly) describing what was achieved.
- full_summary: a step-by-step description of what the agent did (suitable for logs or hidden metadata)."""
    project_name: str = dspy.InputField(description="The name of the project we are working on.")
    task_context: str = dspy.InputField(description="The exact coding task that was requested.")
    project_context: str = dspy.InputField(description="Rendered scratchpad for the project (used as broader context).")
    trajectory_json: str = dspy.InputField(description="The raw JSON string contents of the OpenHands trajectory file.")
    full_summary: str = dspy.OutputField(description="A detailed, step-by-step summary of what the agent accomplished.")
    short_summary: str = dspy.OutputField(description="A single sentence in layman's terms describing what was achieved.  Should be only 1-2 sentences long.")


class CodeAgent:
    def __init__(self, model: dspy.LM):
        self.model = model or dspy.settings.lm
        self.find_repository = FindRepository()
        self.summarize = dspy.ChainOfThought(SummarizeTrajectory)

    async def run(self, project_name: str, task_context: str) -> Dict[str, Any]:
        """
        This mirrors your original async flow, except we render project_context
        from the scratchpad inside this method and summarize the trajectory.
        """
        # Render scratchpad to feed repo finder (acts as project_context)
        project_context = scratchpad_render.render_project_scratchpad(project_name)

        with dspy.context(lm=self.model):
            repository_path = self.find_repository(
                project_name=project_name,
                project_context=project_context,
                task_context=task_context
            )

        repo_full_name = get_repo_full_name(repository_path)

        full_task = (
            f"We are working on the {repo_full_name} repository.  "
            f"The broader project is {project_name}. Some broader details about the project are shared below ===\n"
            f"{project_context}\n===\n\n"
            f"HOWEVER I want you to focus only on this specific task: ===\n"
            f"{task_context}\n===\n\n"
            f"Please follow the following steps to complete the task: ===\n"
            f"1. Make a branch in the repository called `precursor-<task> where <task> is a single word identifying the task."
            f"2. Check out the branch."
            f"3. Investigate the repository to understand the codebase and the task."
            f"4. Edit the code in the branch to complete the task."
            f"5. Commit the changes to the branch."
            f"6. Push the changes to the branch."
            f"7. Create a pull request to the repository.  It's fine to point to the url that will create the pull request as the pull request url.  Be clear about what this url is though!"
            f"You may wish to add more detailed steps to the task as you need for certain more specific tasks.  Be sure to ALWAYS create a branch and a pull request for the task."
        )

        result = await run_openhands_task_with_pr_async(
            project_name=project_name,
            repo=repo_full_name,
            task=full_task,
            github_token=os.getenv("GITHUB_TOKEN")
        )

        # Load trajectory file contents (string) for summarization
        traj_path = result.get("trajectory_path", "")
        traj_json_str = "{}"
        if traj_path:
            try:
                with open(traj_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                # Reduce to highlights to avoid context overflow while keeping truthful content
                traj_json_str = self._shrink_trajectory_json(raw, max_items=40, max_message_chars=2000)
            except Exception:
                # Leave as "{}" if we can't read it
                pass

        # Summarize (short + long) with one DSPy call
        with dspy.context(lm=self.model):
            summ = self.summarize(
                project_name=project_name,
                task_context=task_context,
                project_context=project_context,
                trajectory_json=traj_json_str,
            )
        short_summary = (summ.short_summary or "").strip() or "Code change attempt recorded."
        full_summary = (summ.full_summary or "").strip() or "No detailed steps available."

        # Decide URI preference (numbered PR first, then create-PR link)
        pr_url = result.get("pr_url")
        pr_create = result.get("pr_create_url")
        uri = pr_url or pr_create or ""

        # Log artifact (concise visible line + hidden metadata)
        try:
            store_artifact(
                project_name=project_name,
                task=task_context,
                short_description=short_summary,
                uri=uri,
                step_by_step_summary=full_summary,
            )
            artifact_recorded = True
        except Exception:
            artifact_recorded = False

        final_state = result.get("final_state")

        return {
            "project_name": project_name,
            "repo_path": repository_path,
            "repo_full_name": repo_full_name,
            "final_state": final_state,
            "pr_url": pr_url,
            "pr_create_url": pr_create,
            "trajectory_path": traj_path,
            "short_summary": short_summary,
            "full_summary": full_summary,
            "artifact_recorded": artifact_recorded,
        }

    @staticmethod
    def _shrink_trajectory_json(raw: str, *, max_items: int = 40, max_message_chars: int = 2000) -> str:
        """
        Create a compact JSON string from the trajectory:
        - Keep only the last `max_items` entries
        - Keep lightweight keys: id, timestamp, source, message
        - Truncate very long messages
        If parsing fails, return the original string.
        """
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return raw
            # Take last N entries
            tail = data[-max_items:]
            compact: list[dict] = []
            for entry in tail:
                if not isinstance(entry, dict):
                    continue
                msg = str(entry.get("message", ""))
                if len(msg) > max_message_chars:
                    msg = msg[: max_message_chars] + "...(truncated)"
                compact.append({
                    "id": entry.get("id"),
                    "timestamp": entry.get("timestamp"),
                    "source": entry.get("source"),
                    "message": msg,
                })
            return json.dumps(compact, ensure_ascii=False)
        except Exception:
            return raw


# ---------------------------------------------------------------------------
# MCP Tool (returns a STRING)
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_code_task(
    project_name: str,
    task: str,
) -> str:
    """
    Trigger the coding agent to implement a code change and open a PR.

    NOTE: All coding related tasks should be handled by this tool.  This is a highly capable and specialized agent that is capable of handling a wide range of coding tasks.

    The tool will:
      1) Render the project's scratchpad to provide broader context.
      2) Identify the likely project repository and resolve it to a concrete path.
      3) Run OpenHands (Coding Agent) headlessly to make a branch, edit, commit, push, and open a PR.
      4) Read the **true contents** of the OpenHands trajectory file and summarize it via
         a DSPy call into:
         • short_summary: a single sentence, layperson-friendly description of what was achieved
         • full_summary: a step-by-step summary of what the agent accomplished
      5) Record a concise artifact entry in the scratchpad under
         "Agent Completed Tasks (Pending Review)" with hidden metadata that
         stores the long summary.

    Parameters
    ----------
    project_name : str
        The Precursor project name.
    task : str
        A concrete coding task for the repository.

    Returns
    -------
    str
        A concise human-readable status message including the PR URL (if any),
        the one-sentence short summary, and a reminder:
        **do not resubmit the artifact if it was already recorded successfully.**
    """
    # Ensure Docker is running for OpenHands
    if not ensure_docker_running(timeout_seconds=180):
        return (
            "[DOCKER_NOT_RUNNING] Project: "
            f"{project_name}\nRepo: UNKNOWN (path: UNKNOWN)\nPR: N/A\n"
            "Summary: Docker engine is not running and could not be started automatically. "
            "Please start Docker Desktop (macOS/Windows) or the docker service (Linux), "
            "then retry.\n"
            "Artifact could not be recorded automatically; if you retry, take care not to duplicate entries."
        )

    # Minimal, explicit model selection (as in your old demo)
    model = dspy.LM('openai/gpt-4o-mini-2024-07-18')

    agent = CodeAgent(model)
    out = await agent.run(project_name=project_name, task_context=task)

    pr_hint = out.get("pr_url") or out.get("pr_create_url") or "N/A"
    short_summary = out.get("short_summary", "No summary.")
    final_state = out.get("final_state") or "UNKNOWN"
    artifact_recorded = out.get("artifact_recorded", False)

    note = (
        "Artifact recorded in scratchpad. **Do NOT resubmit this artifact.**"
        if artifact_recorded else
        "Artifact could not be recorded automatically; if you retry, take care not to duplicate entries."
    )

    return (
        f"[{final_state}] Project: {project_name}\n"
        f"Repo: {out.get('repo_full_name', 'UNKNOWN')} (path: {out.get('repo_path', 'UNKNOWN')})\n"
        f"PR: {pr_hint}\n"
        f"Summary: {short_summary}\n"
        f"{note}"
    )


if __name__ == "__main__":
    mcp.run()