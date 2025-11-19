# src/precursor/observers/csv_simulator.py
from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Any, List

from PIL import Image as PILImage
import dspy

from precursor.context.events import ContextEvent
from precursor.config.loader import get_user_name, get_user_description, get_user_agent_goals

logger = logging.getLogger(__name__)


@dataclass
class CSVSimulatorConfig:
    csv_path: str = "dev/survey/context_log.csv"
    # how to pace replay:
    # - "interval": sleep interval_seconds between rows
    # - "asap": no sleep, emit as fast as possible
    # - "step": deterministic step mode for testing (no async sleep)
    mode: str = "interval"
    interval_seconds: float = 180.0  # 3 minutes
    # optional description to attach (e.g. from user.yaml)
    user_description: Optional[str] = None
    # optional name to attach (e.g. from user.yaml)
    user_name: Optional[str] = None
    # optional agent-goals to attach (e.g. from user.yaml)
    user_agent_goals: Optional[str] = None
    # max rows to process (for testing)
    max_rows: Optional[int] = None


class CSVSimulatorObserver:
    """
    Replays recorded context rows (like your old logger produced) as ContextEvent
    objects and feeds them to a callback — usually the StateManager.
    """

    def __init__(self, config: Optional[CSVSimulatorConfig] = None) -> None:
        self.config = config or CSVSimulatorConfig()
        # fill defaults from YAML if not provided
        if not self.config.user_name:
            self.config.user_name = get_user_name()
        if not self.config.user_description:
            self.config.user_description = get_user_description()
        if not self.config.user_agent_goals:
            self.config.user_agent_goals = get_user_agent_goals()

    async def run(self, handler: Callable[[ContextEvent], Any]) -> None:
        """
        Main entrypoint: iterate rows and call `handler(event)` for each.
        """
        rows = self._load_rows(self.config.csv_path)
        
        # Apply max_rows limit if specified (useful for testing)
        if self.config.max_rows is not None:
            rows = rows[:self.config.max_rows]
        
        logger.info(
            "csv simulator starting with %d rows from %s",
            len(rows),
            self.config.csv_path,
        )

        for row in rows:
            event = self._row_to_event(row)
            # hand off to whoever is orchestrating
            try:
                handler(event)
            except StopIteration:
                logger.info("csv simulator received early-stop signal; exiting replay loop")
                break

            # pacing
            if self.config.mode == "interval":
                await asyncio.sleep(self.config.interval_seconds)
            elif self.config.mode == "asap":
                # no sleep
                pass
            elif self.config.mode == "step":
                # deterministic step mode - no async sleep at all
                pass
            else:
                # unknown mode -> treat like interval
                await asyncio.sleep(self.config.interval_seconds)

        logger.info("csv simulator finished replaying all rows")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load_rows(self, path_str: str) -> List[dict]:
        path = Path(path_str)
        text = path.read_text(encoding="utf-8")
        # rely on csv module; your file is comma-delimited in spirit,
        # even though the pasted sample showed tabs
        reader = csv.DictReader(text.splitlines())
        return list(reader)

    def _row_to_event(self, row: dict) -> ContextEvent:
        # timestamp like "20251020_144855"
        ts_raw = row.get("timestamp", "").strip()
        if ts_raw:
            ts = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        screenshot_img: Optional[dspy.Image] = None
        screenshot_path = (row.get("screenshot_path") or "").strip()
        if screenshot_path:
            img = PILImage.open(screenshot_path).convert("RGB")
            screenshot_img = dspy.Image.from_PIL(img)

        # user_details in your logger was JSON-serialized list/dict
        user_details_raw = row.get("user_details") or ""
        if user_details_raw:
            user_details = json.loads(user_details_raw)
        else:
            user_details = None

        calendar_events = (row.get("calendar_events") or "").strip()
        context_update = row.get("context_update") or ""

        # in your current world: user_details == recent_propositions
        event = ContextEvent(
            timestamp=ts,
            context_update=context_update,
            screenshot=screenshot_img,
            user_name=(row.get("user_name") or "").strip() or self.config.user_name,
            user_description=self.config.user_description,
            user_agent_goals=self.config.user_agent_goals,
            recent_propositions=user_details,  # ← single source of truth
            calendar_events=calendar_events or None,
            raw=row,
        )
        return event