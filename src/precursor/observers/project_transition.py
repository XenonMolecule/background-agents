# src/precursor/observers/project_transition.py
"""
ProjectTransitionObserver

This observer is for the *departure* case:

- user was on PROJECT_A for a good stretch
- then user switched to PROJECT_B
- → we trigger background work for PROJECT_A (the one we just left)

Key detail (to match tests):
We measure the duration of the segment we just left as:
    duration = (start of CURRENT segment) - (start of PREVIOUS segment)

i.e. "we were on Alpha from when we first saw Alpha until we started Beta".
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional, List, Tuple, Callable
import logging

from precursor.context.project_history import ProjectHistory, ProjectReading
from precursor.managers.base import Manager
from precursor.testing.telemetry import emit_telemetry

logger = logging.getLogger(__name__)


class ProjectActivityObserver:
    """
    Unified, configurable observer for project activity transitions.
    Modes:
      - 'departure': trigger for the project we just left after minimum duration
      - 'arrival': trigger when returning to current project after an absence
    """

    def __init__(
        self,
        *,
        history: ProjectHistory,
        agent_manager: Manager,
        mode: str,
        window_size: int = 20,
        min_entries_current_segment: int = 1,
        min_entries_previous_segment: int = 3,
        time_threshold: timedelta = timedelta(minutes=10),
        on_trigger: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        if mode not in ("departure", "arrival"):
            raise ValueError("mode must be 'departure' or 'arrival'")
        self.history = history
        self.agent_manager = agent_manager
        self.mode = mode
        self.window_size = window_size
        self.min_entries_current_segment = min_entries_current_segment
        self.min_entries_previous_segment = min_entries_previous_segment
        self.time_threshold = time_threshold
        self.on_trigger = on_trigger
        self._last_triggered_key: Optional[str] = None

    def handle_processed(self) -> None:
        entries: List[ProjectReading] = self.history.recent(self.window_size)
        if len(entries) < 1:
            return

        cur_seg, prev_seg = self._last_two_segments(entries)

        if self.mode == "departure":
            if prev_seg is None:
                return
            (
                cur_project,
                cur_start,
                _cur_end,
                _cur_count,
            ) = cur_seg
            (
                prev_project,
                prev_start,
                _prev_end,
                prev_count,
            ) = prev_seg

            if prev_count < self.min_entries_previous_segment:
                return

            gap = cur_start - prev_start
            if gap < self.time_threshold:
                return

            seg_key = f"{prev_project}:{prev_start.isoformat()}:{cur_start.isoformat()}"
            if self._last_triggered_key == seg_key:
                return

            logger.info(
                "project transition detected: left %s (entries=%d, duration=%s) → triggering manager",
                prev_project,
                prev_count,
                gap,
            )
            # Emit telemetry for project transition
            emit_telemetry("project_transition_detected", {
                "from": prev_project,
                "to": cur_project,
                "duration_seconds": gap.total_seconds(),
                "entries": prev_count
            })
            result = self.agent_manager.run_for_project(prev_project)
            if self.on_trigger is not None:
                self.on_trigger(prev_project, result)
            self._last_triggered_key = seg_key
            return

        # arrival mode
        (
            cur_project,
            cur_start,
            _cur_end,
            cur_count,
        ) = cur_seg
        if cur_count < self.min_entries_current_segment:
            return
        # find last time we were in this project before current segment
        last_seen_end = self._last_seen_end_for_project(entries, cur_project, cur_start)
        if last_seen_end is None:
            return
        absence = cur_start - last_seen_end
        if absence < self.time_threshold:
            return

        seg_key = f"{cur_project}:{cur_start.isoformat()}:{last_seen_end.isoformat()}"
        if self._last_triggered_key == seg_key:
            return

        logger.info(
            "project return detected: returned to %s after %s (entries_in_current=%d) → triggering manager",
            cur_project,
            absence,
            cur_count,
        )
        result = self.agent_manager.run_for_project(cur_project)
        if self.on_trigger is not None:
            self.on_trigger(cur_project, result)
        self._last_triggered_key = seg_key

    # helpers reused from prior implementations
    def _last_two_segments(
        self, entries: List[ProjectReading]
    ) -> Tuple[
        Tuple[str, object, object, int],
        Optional[Tuple[str, object, object, int]],
    ]:
        n = len(entries)
        idx = n - 1

        # current segment
        cur_project = entries[idx].project
        cur_end = entries[idx].timestamp
        cur_start = entries[idx].timestamp
        cur_count = 1
        idx -= 1
        while idx >= 0 and entries[idx].project == cur_project:
            cur_start = entries[idx].timestamp
            cur_count += 1
            idx -= 1
        current_segment = (cur_project, cur_start, cur_end, cur_count)
        if idx < 0:
            return current_segment, None

        prev_project = entries[idx].project
        prev_end = entries[idx].timestamp
        prev_start = entries[idx].timestamp
        prev_count = 1
        idx -= 1
        while idx >= 0 and entries[idx].project == prev_project:
            prev_start = entries[idx].timestamp
            prev_count += 1
            idx -= 1
        previous_segment = (prev_project, prev_start, prev_end, prev_count)
        return current_segment, previous_segment

    def _last_seen_end_for_project(
        self,
        entries: List[ProjectReading],
        project: str,
        current_segment_start: object,
    ) -> Optional[object]:
        """
        Scan entries before the current segment to find the most recent segment end
        for `project` (end timestamp of that older segment).
        """
        # find the index immediately before current segment by walking back from end
        n = len(entries)
        idx = n - 1
        while idx >= 0 and entries[idx].timestamp != current_segment_start:
            idx -= 1
        # at this point, idx points to first element of current segment
        idx -= 1
        while idx >= 0:
            if entries[idx].project == project:
                return entries[idx].timestamp
            idx -= 1
        return None
