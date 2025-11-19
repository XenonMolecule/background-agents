# src/precursor/managers/agent_manager.py
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
import subprocess
import sys

from precursor.scratchpad import render as scratchpad_render
from precursor.components.task_proposer.task_proposer_pipeline import (
    TaskProposerPipeline,
)
import precursor.config.loader as config_loader
from precursor.components.task_proposer.task_scorer import TaskAssessment

logger = logging.getLogger(__name__)

# Import sinks for optional injection
try:
    from precursor.testing.sinks import TelemetrySink
except ImportError:
    # Define minimal interface if testing module not available
    from abc import ABC, abstractmethod
    
    class TelemetrySink(ABC):
        @abstractmethod
        def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
            pass


class AgentManager:
    """
    Given a project name:
      1. load its scratchpad
      2. run TaskProposerPipeline to propose tasks and score them
      3. return / log high-priority candidate actions (filtered/sorted)
      4. (future) dispatch agents for those actions

    This is intentionally small and opinionated.
    """

    def __init__(
        self,
        *,
        task_pipeline: Optional[TaskProposerPipeline] = None,
        deploy_enabled: bool = False,
        telemetry_sink: Optional[TelemetrySink] = None,
    ) -> None:
        # dspy.Module – creates goals, milestones, tasks, and assessments
        self.task_pipeline = task_pipeline or TaskProposerPipeline()
        # load settings for scoring/selection
        settings = config_loader.get_settings() or {}
        self.value_weight: float = float(settings.get("value_weight", 2.0))
        self.feasibility_weight: float = float(settings.get("feasibility_weight", 1.5))
        self.user_pref_alignment_weight: float = float(
            settings.get("user_preference_alignment_weight", 0.5)
        )
        self.safety_threshold: int = int(settings.get("safety_threshold", 7))
        self.deployment_threshold: float = float(
            settings.get("deployment_threshold", 0.9)
        )
        self.max_deployed_tasks: int = int(settings.get("max_deployed_tasks", 3))
        # runtime toggle: actually dispatch MCP agents for selected candidates
        self.deploy_enabled: bool = deploy_enabled
        # optional telemetry sink for testing
        self.telemetry_sink = telemetry_sink

    def _refresh_settings(self) -> None:
        """
        Reload weights/thresholds from settings.yaml to adapt to live edits.
        """
        settings = config_loader.get_settings() or {}
        self.value_weight = float(settings.get("value_weight", self.value_weight))
        self.feasibility_weight = float(
            settings.get("feasibility_weight", self.feasibility_weight)
        )
        self.user_pref_alignment_weight = float(
            settings.get(
                "user_preference_alignment_weight", self.user_pref_alignment_weight
            )
        )
        self.safety_threshold = int(settings.get("safety_threshold", self.safety_threshold))
        self.deployment_threshold = float(
            settings.get("deployment_threshold", self.deployment_threshold)
        )
        self.max_deployed_tasks = int(
            settings.get("max_deployed_tasks", self.max_deployed_tasks)
        )

    def compute_true_score(self, a: TaskAssessment) -> float:
        value = float(a.value_score or 0)
        feas = float(a.feasibility_score or 0)
        align = float(a.user_preference_alignment_score or 0)
        return (
            value * self.value_weight
            + feas * self.feasibility_weight
            + align * self.user_pref_alignment_weight
        )

    def run_for_project(
        self,
        project_name: str,
        *,
        user_profile: str = "",
        project_description: Optional[str] = None,
        user_agent_goals: Optional[str] = None,
    ) -> dict:
        """
        Main entrypoint: propose and score background-agent tasks for this project.
        """
        logger.info("agent_manager: running for project %s", project_name)

        # Always refresh settings so live edits to YAML take effect
        self._refresh_settings()

        # 1) get latest scratchpad
        scratchpad_text = scratchpad_render.render_project_scratchpad(project_name)
        if not scratchpad_text.strip():
            logger.warning(
                "agent_manager: project %s has empty scratchpad, skipping task proposal",
                project_name,
            )
            return {
                "project": project_name,
                "future_goals": [],
                "goal_to_milestones": {},
                "agent_tasks": [],
                "task_assessments": [],
                "candidates": [],
            }

        # 2) run the task proposer pipeline
        
        pipeline_out: Dict[str, Any] = self.task_pipeline(
            user_profile=user_profile,
            project_name=project_name,
            project_scratchpad=scratchpad_text,
            project_description=project_description,
            user_agent_goals=user_agent_goals,
        )
        
        future_goals: List[str] = list(pipeline_out.get("future_goals", []) or [])
        goal_to_milestones: Dict[str, List[str]] = dict(
            pipeline_out.get("goal_to_milestones", {}) or {}
        )
        agent_tasks: List[str] = list(pipeline_out.get("agent_tasks", []) or [])
        # assessments may be pydantic models or plain dicts depending on caller
        assessments: List[TaskAssessment] = list(
            pipeline_out.get("task_assessments", []) or []
        )

        # 3) weighted scoring + selection per settings.yaml
        #    - compute true score and ratio
        #    - filter by safety threshold
        #    - filter by ratio >= deployment_threshold
        #    - sort by true score desc; tie-break by weight order (highest weight first)
        max_score = 10.0 * (
            self.value_weight + self.feasibility_weight + self.user_pref_alignment_weight
        )

        # Determine tie-break order dynamically by descending weight
        tie_break_fields: List[str] = []
        weights: List[tuple[str, float]] = [
            ("value_score", self.value_weight),
            ("feasibility_score", self.feasibility_weight),
            ("user_preference_alignment_score", self.user_pref_alignment_weight),
        ]
        for field, _ in sorted(weights, key=lambda x: -x[1]):
            tie_break_fields.append(field)

        # filter + annotate
        filtered: List[Dict[str, Any]] = []
        for a in assessments:
            if a.safety_score < self.safety_threshold:
                continue
            ts = self.compute_true_score(a)
            ratio = ts / max_score if max_score > 0 else 0.0
            if ratio < self.deployment_threshold:
                continue
            b = dict(a)
            b["_true_score"] = ts
            b["_score_ratio"] = ratio
            filtered.append(b)

        # sort by true score desc, then tie-break by the configured order
        def sort_key(a: Dict[str, Any]) -> tuple:
            tie = tuple(-(a.get(f) or 0) for f in tie_break_fields)
            return (-(a.get("_true_score") or 0.0),) + tie

        candidates = sorted(filtered, key=sort_key)

        # final cap
        if self.max_deployed_tasks > 0 and len(candidates) > self.max_deployed_tasks:
            candidates = candidates[: self.max_deployed_tasks]

        # 4) log all assessments for observability
        for a in assessments:
            logger.debug(
                "agent_manager: task=%r value=%s safety=%s feasibility=%s align=%s",
                a.task_description,
                a.value_score,
                a.safety_score,
                a.feasibility_score,
                a.user_preference_alignment_score,
            )

        # 5) emit telemetry events
        if self.telemetry_sink:
            self.telemetry_sink.emit("batch_processed", {
                "count": len(assessments),
                "project": project_name,
                "candidates_selected": len(candidates)
            })

        # 6) future: actually dispatch here
        # Optionally spawn separate MCP agent processes for each candidate.
        if self.deploy_enabled and candidates:
            self._deploy_candidates(project_name, candidates)

        # return structured so tests / callers can inspect
        return {
            "project": project_name,
            "future_goals": future_goals,
            "goal_to_milestones": goal_to_milestones,
            "agent_tasks": agent_tasks,
            "task_assessments": assessments,
            "candidates": candidates,
        }

    # ---------------------------------------------------------------------
    # Deployment helpers
    # ---------------------------------------------------------------------
    def _deploy_candidates(self, project_name: str, candidates: List[Dict[str, Any]]) -> None:
        """
        Spawn a background MCP Agent process for each candidate task.
        Uses the CLI entrypoint `python -m precursor.cli.mcp_agent_cli`.
        """
        for c in candidates:
            task_desc = (c.get("task_description") or "").strip()
            if not task_desc:
                continue
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "precursor.cli.mcp_agent_cli",
                    "--project",
                    project_name,
                    "--task",
                    task_desc,
                ]
                logger.info("agent_manager: deploying MCPAgent for project=%r task=%r", project_name, task_desc)
                # Run detached/background; do not wait
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                logger.exception("agent_manager: failed to spawn MCPAgent for task %r", task_desc)