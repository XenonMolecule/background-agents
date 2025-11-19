# src/precursor/observers/gum_source.py
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from dotenv import load_dotenv
from gum import gum
# Guard Calendar import to handle environments where gum.observers.Calendar is unavailable
try:
    from gum.observers import Calendar
except ImportError:
    Calendar = None
from gum.observers import Screen

from precursor.context.events import ContextEvent
from precursor.context.utils import grab_screen_dspy_image
from precursor.config.loader import get_user_name, get_user_description, get_user_agent_goals

logger = logging.getLogger(__name__)
load_dotenv()


def _is_calendar_disabled() -> bool:
    """Check if Calendar observer should be disabled via feature flag."""
    flag_value = os.getenv("PRECURSOR_DISABLE_CALENDAR", "").lower()
    return flag_value in ("1", "true")


class GumSource:
    """
    Real-time observer that listens to gum updates and converts them into
    `ContextEvent`s for the rest of the pipeline.

    Usage:
        async def handle(evt: ContextEvent):
            state_manager.process_event(evt)

        src = GumSource()
        await src.run(handle)
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        poll_calendar_days: int = 3,
        max_batch_size: int = 15,
        capture_screenshot: bool = True,
        on_event: Optional[Callable[[ContextEvent], Any] | Callable[[ContextEvent], Awaitable[Any]]] = None,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.user_name = get_user_name()
        self.user_description = get_user_description()
        self.user_agent_goals = get_user_agent_goals()
        self.model = model or os.getenv("GUM_MODEL") or "gpt-4o-mini-2024-07-18"
        self.poll_calendar_days = poll_calendar_days
        self.max_batch_size = max_batch_size
        self.capture_screenshot = capture_screenshot
        self.on_event = on_event
        self.cooldown_seconds = float(cooldown_seconds or 0.0)
        self._last_sent_at: Optional[datetime] = None

    def _serialize_recent_propositions(self, items: Any) -> str:
        """
        Convert gum.recent() output (often a list of Proposition objects)
        into a newline-delimited string suitable for LLM inputs.
        """
        if not items:
            return ""
        # If it's already a string, pass through
        if isinstance(items, str):
            return items
        # If it's a list/iterable, try to extract `text` and optional reasoning
        try:
            lines = []
            for idx, item in enumerate(items, start=1):
                # Prefer explicit fields commonly present on gum Proposition
                text = getattr(item, "text", None)
                reasoning = getattr(item, "reasoning", None)
                if isinstance(text, str) and text.strip():
                    line = f"[{idx}] {text.strip()}"
                    if isinstance(reasoning, str) and reasoning.strip():
                        line = f"{line} — {reasoning.strip()}"
                    lines.append(line)
                else:
                    # Fallback to string representation
                    lines.append(f"[{idx}] {str(item)}")
            return "\n".join(lines)
        except Exception:
            # Final fallback
            return str(items)

    async def run(
        self,
        handler: Optional[Callable[[ContextEvent], Any] | Callable[[ContextEvent], Awaitable[Any]]] = None,
    ) -> None:
        """
        Open gum, register an update handler, and run forever.
        """
        effective_handler = handler or self.on_event
        if effective_handler is None:
            raise ValueError("GumSource.run requires a handler (or provide on_event in __init__).")
        
        # Initialize Calendar observer if available and not disabled
        cal = None
        if Calendar is not None and not _is_calendar_disabled():
            cal = Calendar()
            logger.info("Calendar observer enabled")
        else:
            if Calendar is None:
                logger.info("Calendar observer unavailable (gum.observers.Calendar import failed)")
            else:
                logger.info("Calendar observer disabled via PRECURSOR_DISABLE_CALENDAR flag")
        
        screen = Screen(self.model)
        logger.info("Starting GumSource for user=%s, model=%s", self.user_name, self.model)

        # Build observers list, only including calendar if available
        observers = [screen]
        if cal is not None:
            observers.append(cal)
        
        async with gum(
            self.user_name,
            self.model,
            *observers,
            max_batch_size=self.max_batch_size,
        ) as gum_instance:

            async def _on_update(observer, update):
                # Cooldown gating: drop updates that arrive too soon
                now = datetime.now(timezone.utc)
                if self.cooldown_seconds > 0 and self._last_sent_at is not None:
                    if (now - self._last_sent_at) < timedelta(seconds=self.cooldown_seconds):
                        return
                # === Context ===
                context_update = update.content
                recent_list = await gum_instance.recent()
                recent_propositions = self._serialize_recent_propositions(recent_list)
                
                # Query calendar events only if calendar is available
                calendar_events = ""
                if cal is not None:
                    calendar_events = cal.query_str(
                        start_delta=timedelta(days=0),
                        end_delta=timedelta(days=self.poll_calendar_days),
                    )
                
                screenshot = grab_screen_dspy_image() if self.capture_screenshot else None

                # === Package event ===
                event = ContextEvent(
                    timestamp=now,
                    context_update=context_update,
                    screenshot=screenshot,
                    user_name=self.user_name,
                    user_description=self.user_description,
                    user_agent_goals=self.user_agent_goals,
                    recent_propositions=recent_propositions,  # same as user_details
                    calendar_events=calendar_events,
                    raw=update,
                )

                result = effective_handler(event)
                if asyncio.iscoroutine(result):
                    await result
                self._last_sent_at = now

            gum_instance.register_update_handler(_on_update)
            await asyncio.Future()


__all__ = ["GumSource"]