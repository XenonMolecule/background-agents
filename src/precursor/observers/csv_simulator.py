# src/precursor/observers/csv_simulator.py
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Any, List, Union

from PIL import Image as PILImage
import dspy

from precursor.context.events import ContextEvent
from precursor.config.loader import get_user_name, get_user_description, get_user_agent_goals

logger = logging.getLogger(__name__)


class Clock(ABC):
    """Abstract clock interface for time management."""
    
    @abstractmethod
    def now(self) -> float:
        """Return current timestamp as seconds since epoch."""
        pass


class RealClock(Clock):
    """Real-time clock using system time."""
    
    def now(self) -> float:
        return time.time()


class FakeClock(Clock):
    """Fake clock for deterministic testing."""
    
    def __init__(self, initial_time: float = 0.0):
        self._current_time = initial_time
    
    def now(self) -> float:
        return self._current_time
    
    def advance(self, delta: float) -> None:
        """Advance the clock by delta seconds."""
        self._current_time += delta


@dataclass
class CSVSimulatorConfig:
    csv_path: str = "dev/survey/context_log.csv"
    # how to pace replay:
    # - "interval": sleep interval_seconds between rows
    # - "asap": no sleep, emit as fast as possible
    # - "deterministic": use timestamps and injectable clock for deterministic replay
    mode: str = "interval"
    interval_seconds: float = 180.0  # 3 minutes
    # optional description to attach (e.g. from user.yaml)
    user_description: Optional[str] = None
    # optional name to attach (e.g. from user.yaml)
    user_name: Optional[str] = None
    # optional agent-goals to attach (e.g. from user.yaml)
    user_agent_goals: Optional[str] = None
    # injectable clock for deterministic mode (defaults to real time)
    clock: Optional[Clock] = None


class CSVSimulatorObserver:
    """
    Replays recorded context rows (like your old logger produced) as ContextEvent
    objects and feeds them to a callback — usually the StateManager.
    
    Supports deterministic mode for testing via injectable clock and timestamp-based
    replay using run_once() and step() methods.
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
        
        # Set up clock
        self.clock = self.config.clock or RealClock()
        
        # For deterministic mode: preload and sort rows by timestamp
        self._rows: Optional[List[dict]] = None
        self._current_index = 0
        self._start_time: Optional[float] = None

    async def run(self, handler: Callable[[ContextEvent], Any]) -> None:
        """
        Main entrypoint: iterate rows and call `handler(event)` for each.
        """
        if self.config.mode == "deterministic":
            # Use deterministic mode - step through all rows
            self._prepare_deterministic_mode()
            while self._current_index < len(self._rows):
                events = self.step(float('inf'))  # Process all remaining rows
                for event in events:
                    try:
                        handler(event)
                    except StopIteration:
                        logger.info("csv simulator received early-stop signal; exiting replay loop")
                        return
            logger.info("csv simulator finished replaying all rows")
            return
        
        # Original behavior for interval/asap modes
        rows = self._load_rows(self.config.csv_path)
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
            else:
                # unknown mode -> treat like interval
                await asyncio.sleep(self.config.interval_seconds)

        logger.info("csv simulator finished replaying all rows")

    def run_once(self, handler: Callable[[ContextEvent], Any]) -> bool:
        """
        Deterministic mode: process at most one eligible row given the current clock.
        
        Returns:
            True if a row was processed, False if no row was eligible.
        """
        if self.config.mode != "deterministic":
            raise ValueError("run_once() only available in deterministic mode")
        
        self._prepare_deterministic_mode()
        
        if self._current_index >= len(self._rows):
            return False
        
        current_time = self.clock.now()
        row = self._rows[self._current_index]
        row_time = self._get_row_timestamp(row)
        
        # Check if this row is eligible (its time has come)
        if current_time >= row_time:
            event = self._row_to_event(row)
            try:
                handler(event)
            except StopIteration:
                logger.info("csv simulator received early-stop signal in run_once")
                return False
            
            self._current_index += 1
            return True
        
        return False
    
    def step(self, clock_tick: float) -> List[ContextEvent]:
        """
        Deterministic mode: advance the clock by clock_tick and process all newly eligible rows.
        
        Args:
            clock_tick: Amount to advance the clock (in seconds)
            
        Returns:
            List of ContextEvents that were emitted in this step
        """
        if self.config.mode != "deterministic":
            raise ValueError("step() only available in deterministic mode")
        
        if not isinstance(self.clock, FakeClock):
            raise ValueError("step() requires a FakeClock instance")
        
        self._prepare_deterministic_mode()
        
        # Advance the clock
        self.clock.advance(clock_tick)
        current_time = self.clock.now()
        
        # Process all eligible rows
        events = []
        while self._current_index < len(self._rows):
            row = self._rows[self._current_index]
            row_time = self._get_row_timestamp(row)
            
            if current_time >= row_time:
                event = self._row_to_event(row)
                events.append(event)
                self._current_index += 1
            else:
                break
        
        return events

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
    
    def _prepare_deterministic_mode(self) -> None:
        """Prepare for deterministic mode by loading and sorting rows."""
        if self._rows is None:
            self._rows = self._load_rows(self.config.csv_path)
            # Sort by timestamp to ensure deterministic ordering
            self._rows.sort(key=lambda row: self._get_row_timestamp(row))
            
            # Set start time based on first row's timestamp
            if self._rows and self._start_time is None:
                first_row_time = self._get_row_timestamp(self._rows[0])
                self._start_time = first_row_time
                
                # If using FakeClock, initialize it to the start time
                if isinstance(self.clock, FakeClock) and self.clock.now() == 0.0:
                    self.clock._current_time = first_row_time
            
            logger.info(
                "csv simulator prepared deterministic mode with %d rows from %s",
                len(self._rows),
                self.config.csv_path,
            )
    
    def _get_row_timestamp(self, row: dict) -> float:
        """Convert row timestamp to seconds since epoch."""
        ts_raw = row.get("timestamp", "").strip()
        if ts_raw:
            ts = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            return ts.timestamp()
        else:
            # If no timestamp, use current time
            return datetime.now(timezone.utc).timestamp()