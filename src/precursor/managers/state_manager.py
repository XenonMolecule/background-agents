# src/precursor/managers/state_manager.py
"""
StateManager: the central, synchronous pipeline for
    ContextEvent -> objectives -> project -> scratchpad update -> history
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any

import dspy

from precursor.context.events import ContextEvent
from precursor.context.project_history import ProjectHistory
from precursor.components.objective_inducer import ObjectivesInducer
from precursor.components.current_project_classifier import CurrentProjectClassifier
from precursor.components.scratchpad_updater import ScratchpadUpdater
from precursor.config.loader import get_project_names, get_user_profile
from precursor.managers.utils import (
    goals_to_text,
    goals_to_objective_strings,
    ensure_screenshot_image,
)
from precursor.testing.telemetry import emit_telemetry

logger = logging.getLogger(__name__)


class StateManager:
    """
    Orchestrates the core pipeline on every incoming ContextEvent:

        event
          -> induce objectives
          -> classify project
          -> update scratchpad
          -> record in history

    This class stays “LM-aware” but not “observer-aware”; observers produce
    ContextEvent objects, and we just consume them.
    """

    def __init__(
        self,
        *,
        history: Optional[ProjectHistory] = None,
        objectives_inducer: Optional[ObjectivesInducer] = None,
        project_classifier: Optional[CurrentProjectClassifier] = None,
        scratchpad_updater: Optional[ScratchpadUpdater] = None,
    ) -> None:
        self.history = history or ProjectHistory()
        self.objectives_inducer = objectives_inducer or ObjectivesInducer()
        self.project_classifier = project_classifier or CurrentProjectClassifier(
            include_scratchpads=True,
            max_scratchpad_chars=1200,
        )
        self.scratchpad_updater = scratchpad_updater or ScratchpadUpdater()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def process_event(self, event: ContextEvent) -> dict:
        """
        Run the full pipeline on a single context event and return a dict
        with the useful outputs for whoever called us (observer, UI, etc.).
        """
        logger.info("processing context event at %s", event.timestamp.isoformat())
        
        # Emit telemetry for observation queued (we don't have queue size here, but we can track processing)
        emit_telemetry("observation_processed", {
            "timestamp": event.timestamp.isoformat(),
            "has_screenshot": event.screenshot is not None,
            "context_length": len(event.context_update) if event.context_update else 0
        })

        try:
            # 1) induce objectives from the rich context
            context_for_inducer = self._build_context_for_inducer(event)
            goals, reasoning = self.objectives_inducer(
                context=context_for_inducer,
                limit=3,
                screenshot=event.screenshot,
            )
            logger.debug("induced %d goals", len(goals))

            # 2) classify current project
            #    we pass a small history of project predictions to help continuity
            recent_preds = [r.project for r in self.history.recent(5)]
            screenshot_img = ensure_screenshot_image(event.screenshot)

            clf_res = self.project_classifier(
                recent_objectives=goals_to_text(goals),
                # observer should have placed gum "recent" here
                recent_propositions=event.recent_propositions or "",
                calendar_events=event.calendar_events or "",
                screenshot=screenshot_img,
                recent_project_predictions=recent_preds,
            )
            current_project = clf_res.project
            logger.debug("classified current project as %s", current_project)

            # 3) build objective strings for scratchpad (current)
            current_objectives_rich: List[str] = goals_to_objective_strings(goals)

            # 4) get former objectives from *history for the same project*
            #    we scan a small window of recent entries and pick the latest one
            #    that matches this project
            recent_entries = [
                e for e in reversed(self.history.recent(20))
                if e.project == current_project
            ]
            former_objectives_rich: List[str] = (
                recent_entries[0].objectives if recent_entries else []
            )

            # 5) update the scratchpad for the current project
            edits_summary, refreshed_scratchpad = self.scratchpad_updater(
                project_name=current_project,
                user_context=event.context_update,
                current_screenshot=screenshot_img,
                user_profile=get_user_profile(),
                current_scratchpad=None,
                speculated_current_objectives=current_objectives_rich,
                speculated_former_objectives=former_objectives_rich,
                # classifier got a single string; updater likes a list
                calendar_events=(event.calendar_events or "").splitlines(),
                full_project_list=list(get_project_names(only_enabled=False)),
            )
            logger.debug(
                "scratchpad updated for %s: %s", current_project, edits_summary
            )

            # 6) record in history (we store the goals so future edits can see them)
            self.history.append(
                timestamp=event.timestamp,
                project=current_project,
                objectives=current_objectives_rich,
            )

            result = {
                "project": current_project,
                "induced_goals": [g.model_dump() for g in goals],
                "induction_reasoning": reasoning,
                "scratchpad_edits_summary": edits_summary,
                "scratchpad_text": refreshed_scratchpad,
            }

            logger.info(
                "event processed → project=%s objectives=%d",
                current_project,
                len(goals),
            )
            return result

        except Exception:
            logger.exception("failed to process context event")
            raise

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_context_for_inducer(self, event: ContextEvent) -> str:
        """
        Turn a ContextEvent into exactly the kind of rich string
        the objectives inducer expects.
        """
        parts: List[str] = []
        if event.user_name:
            parts.append(f"User: {event.user_name}")
        if event.user_description:
            parts.append(f"User Description: {event.user_description}")
        if getattr(event, "user_agent_goals", None):
            parts.append(f"Agent Goals (Things this user wants the agent to focus on; not exhaustive): {event.user_agent_goals}")
        # this already contains the gum "recent" output per your latest note
        if event.recent_propositions:
            parts.append(f"User Details / Recent Propositions:\n{event.recent_propositions}")
        if event.calendar_events:
            parts.append(f"Calendar Events: {event.calendar_events}")
        parts.append(f"Current Context Update: {event.context_update}")
        return "\n".join(parts)