from __future__ import annotations
from typing import List

from gum import gum
from gum.observers import Calendar

import dspy
import pydantic

# pip install mss pillow pynput
from typing import Optional, Tuple
import mss
from PIL import Image
from pynput.mouse import Controller as MouseController
from datetime import timedelta
import os
from datetime import datetime
import csv
import json

def _point_in_rect(x: int, y: int, rect: dict) -> bool:
    return (rect["left"] <= x < rect["left"] + rect["width"] and
            rect["top"]  <= y < rect["top"]  + rect["height"])

def _monitor_for_mouse(mons: list[dict], x: int, y: int) -> dict:
    """
    Return the monitor dict that contains (x, y). Fallback to first physical monitor.
    `mons` is expected to be mss.monitors[1:] (physical monitors only).
    """
    for m in mons:
        if _point_in_rect(x, y, m):
            return m
    return mons[0]  # fallback

def grab_screen_at_mouse(region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """
    Capture the screen of the monitor currently under the mouse cursor.

    Args:
        region: Optional (left, top, width, height) within the detected monitor.

    Returns:
        PIL.Image.Image (RGB).
    """
    mouse = MouseController()
    x, y = mouse.position  # current global mouse position

    with mss.mss() as sct:
        # mss.monitors[0] = virtual full desktop; [1:] = physical monitors
        physical = sct.monitors[1:]
        mon = _monitor_for_mouse(physical, int(x), int(y))

        if region is not None:
            rx, ry, rw, rh = region
            mon = {"left": mon["left"] + rx,
                   "top":  mon["top"]  + ry,
                   "width": rw,
                   "height": rh}

        frame = sct.grab(mon)  # BGRA
        return Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)

GOAL_INDUCTION__PROMPT = """I have the attached a CONTEXT that a current user is working on:

Now, employ the following reasoning framework when inferring the goals. 
0. If there is an attached screenshot, use context clues to infer what application the user is viewing and what they might be doing in that application. Are they the direct author of the text, or are they viewing it as a reader? Are they actively editing the text, providing feedback, or synthesizing the content?
1. Identify the genre of what the user is working on and their stage of completion. Map the content's genre and completion stage to common goals users of these genre and stages may have and form an initial hypothesis of what the user's goals may be.
2. Infer who the intended audience of the content is. Based on how you think the user wants their audience to receive their content, update your goal hypothesis.
3. Think about what an ideal version of the user's current content would look like and identify what is missing. Then, use this to update your goal hypothesis.
4. Simulate what the user's reaction would be to possible tools generated (e.g. grammar checker, style reviser, high-level structure advisor, new content generator, etc.). Use the user's responses to update your goal hypothesis.

For each step in your reasoning, briefly write out your thought process, your current hypothesis of the goals as a numbered list, and what the updated list would be after your reasoning.

After you are done, finalize the [[limit]] most important goals. Make sure these goals are distinct and have minimal overlap. """

class Goal(pydantic.BaseModel):
    name: str
    description: str
    weight: pydantic.conint(ge=1, le=10)  # weight must be between 1 and 10 inclusive

class InduceObjectivesWithScreenshot(dspy.Signature):
    context:str = dspy.InputField(description="The context that the user is working on")
    screenshot: dspy.Image = dspy.InputField(description="The screenshot of the user's current workspace")
    limit: int = dspy.InputField(description="The number of goals to induce")
    goals: List[Goal] = dspy.OutputField(description="The goals that the user is working on")

class InduceObjectives(dspy.Signature):
    context:str = dspy.InputField(description="The context that the user is working on")
    limit: int = dspy.InputField(description="The number of goals to induce")
    goals: List[Goal] = dspy.OutputField(description="The goals that the user is working on")

class ObjectiveInducer():

    def __init__(self, gum: gum, calendar: Calendar):
        self.gum = gum
        self.calendar = calendar

        self._induce_with_screenshot = dspy.ChainOfThought(InduceObjectivesWithScreenshot.with_instructions(GOAL_INDUCTION__PROMPT))
        self._induce_without_screenshot = dspy.ChainOfThought(InduceObjectives.with_instructions(GOAL_INDUCTION__PROMPT))

    def _to_plain(self, v):
        try:
            # Normalize datetimes early
            if isinstance(v, datetime):
                return v.isoformat()
            if hasattr(v, "model_dump"):
                return v.model_dump()
            if hasattr(v, "dict"):
                return v.dict()
            if isinstance(v, list):
                return [self._to_plain(x) for x in v]
            if isinstance(v, dict):
                return {k: self._to_plain(val) for k, val in v.items()}
            if hasattr(v, "__dict__"):
                return {k: self._to_plain(val) for k, val in v.__dict__.items()}
        except Exception:
            pass
        return str(v)

    def _format_user_details(self, user_details) -> str:
        def _extract(item):
            # Pull only the fields we care about
            def _getattr(obj, name, default=""):
                try:
                    return getattr(obj, name, default)
                except Exception:
                    return default

            if isinstance(item, dict):
                created = item.get("created_at", "")
                if isinstance(created, datetime):
                    created = created.isoformat()
                return {
                    "id": item.get("id", ""),
                    "text": item.get("text", str(item)),
                    "confidence": item.get("confidence", ""),
                    "decay": item.get("decay", ""),
                    "reasoning": item.get("reasoning", ""),
                    "created_at": created,
                }
            created_attr = _getattr(item, "created_at", "")
            if isinstance(created_attr, datetime):
                created_attr = created_attr.isoformat()
            return {
                "id": _getattr(item, "id", ""),
                "text": _getattr(item, "text", str(item)),
                "confidence": _getattr(item, "confidence", ""),
                "decay": _getattr(item, "decay", ""),
                "reasoning": _getattr(item, "reasoning", ""),
                "created_at": created_attr,
            }

        lines: list[str] = []
        try:
            items = user_details if isinstance(user_details, list) else [user_details]
            for raw in items:
                data = _extract(raw)
                if not data.get("text"):
                    continue
                lines.append(f"- [id={data.get('id')}] {data.get('text')}")
                meta_bits = []
                if data.get("confidence") != "":
                    meta_bits.append(f"confidence: {data['confidence']}")
                if data.get("decay") != "":
                    meta_bits.append(f"decay: {data['decay']}")
                if data.get("created_at"):
                    meta_bits.append(f"created_at: {data['created_at']}")
                if meta_bits:
                    lines.append("  - " + " | ".join(meta_bits))
                if data.get("reasoning"):
                    lines.append(f"  - reasoning: {data['reasoning']}")
        except Exception:
            lines.append(str(user_details))
        return "\n".join(lines)

    async def _get_context(self, context: str) -> str:
        user_name = self.gum.user_name

        user_details = await self.gum.recent()
        user_details_str = self._format_user_details(user_details)
        calendar_events = self.calendar.query_str(start_delta=timedelta(days=0), end_delta=timedelta(days=1))

        context_str = (
            f"User: {user_name}\n"
            f"User Details:\n{user_details_str}\n"
            f"Calendar Events: {calendar_events}\n"
            f"Current Context Update: {context}"
        )

        return context_str

    async def induce_with_screenshot(self, context: str, limit: int = 3) -> Tuple[List[Goal], str]:
        screenshot = dspy.Image.from_PIL(grab_screen_at_mouse())
        context = await self._get_context(context)
        res = self._induce_with_screenshot(context=context, screenshot=screenshot, limit=limit)
        return res.goals, res.reasoning

    async def induce_without_screenshot(self, context: str, limit: int) -> Tuple[List[Goal], str]:
        context = await self._get_context(context)
        res = self._induce_without_screenshot(context=context, limit=limit)
        return res.goals, res.reasoning

    async def induce_and_log(self, context: str, limit: int = 3, csv_path: Optional[str] = None, screenshot_directory: Optional[str] = None) -> Tuple[List[Goal], str]:

        # Take the screenshot and save it to the screenshot directory
        screenshot = grab_screen_at_mouse()
        screenshot_dspy = dspy.Image.from_PIL(screenshot)
        screenshot_path: Optional[str] = None
        if screenshot_directory:
            os.makedirs(screenshot_directory, exist_ok=True)
            screenshot_path = os.path.join(screenshot_directory, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        # Build the full context
        user_name = self.gum.user_name

        user_details = await self.gum.recent()
        calendar_events = self.calendar.query_str(start_delta=timedelta(days=0), end_delta=timedelta(days=1))
        full_context = f"User: {user_name}\nUser Details: {user_details}\nCalendar Events: {calendar_events}\nCurrent Context Update: {context}"

        # Induce the goals
        res = self._induce_with_screenshot(context=full_context, screenshot=screenshot_dspy, limit=limit)

        # Log the screenshot, context, and goals to the CSV
        if screenshot_path:
            screenshot.save(screenshot_path)

        if csv_path:
            # Ensure parent directory exists and write header if new file
            parent = os.path.dirname(csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

            # Desired full schema (adds recent_observations)
            columns_full = [
                "timestamp",
                "screenshot_path",
                "user_name",
                "user_details",
                "calendar_events",
                "recent_observations",
                "context_update",
                "goals",
                "reasoning",
            ]

            # If file exists but lacks the new column, migrate in-place by adding a blank column
            if not write_header and os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', newline='', encoding='utf-8') as rf:
                        reader = csv.reader(rf)
                        existing_header = next(reader, None)
                    if existing_header and "recent_observations" not in existing_header:
                        tmp_path = csv_path + ".tmp"
                        # Read original rows as dicts
                        with open(csv_path, 'r', newline='', encoding='utf-8') as rf, \
                             open(tmp_path, 'w', newline='', encoding='utf-8') as wf:
                            dict_reader = csv.DictReader(rf)
                            dict_writer = csv.DictWriter(wf, fieldnames=columns_full)
                            dict_writer.writeheader()
                            for row in dict_reader:
                                row.setdefault("recent_observations", "")
                                dict_writer.writerow({k: row.get(k, "") for k in columns_full})
                        os.replace(tmp_path, csv_path)
                except Exception:
                    # If migration fails, proceed without crashing; we'll still append using existing header
                    pass

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns_full, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                # Safe serialization helpers
                def _to_plain(v):
                    try:
                        if hasattr(v, "model_dump"):
                            return v.model_dump()
                        if hasattr(v, "dict"):
                            return v.dict()
                        if isinstance(v, list):
                            return [_to_plain(x) for x in v]
                        if isinstance(v, dict):
                            return {k: _to_plain(val) for k, val in v.items()}
                    except Exception:
                        pass
                    return str(v)

                goals_plain = [self._to_plain(g) for g in res.goals]

                # Filter user_details for CSV to only the wanted fields
                def _filter_for_csv(details):
                    def _one(item):
                        if isinstance(item, dict):
                            created = item.get("created_at")
                            if isinstance(created, datetime):
                                created = created.isoformat()
                            return {
                                "id": item.get("id"),
                                "text": item.get("text"),
                                "confidence": item.get("confidence"),
                                "decay": item.get("decay"),
                                "reasoning": item.get("reasoning"),
                                "created_at": created,
                            }
                        # object with attrs
                        created_attr = getattr(item, "created_at", None)
                        if isinstance(created_attr, datetime):
                            created_attr = created_attr.isoformat()
                        return {
                            "id": getattr(item, "id", None),
                            "text": getattr(item, "text", None),
                            "confidence": getattr(item, "confidence", None),
                            "decay": getattr(item, "decay", None),
                            "reasoning": getattr(item, "reasoning", None),
                            "created_at": created_attr,
                        }

                    if isinstance(details, list):
                        return [_one(x) for x in details]
                    return [_one(details)]

                user_details_plain = _filter_for_csv(user_details)
                calendar_events_plain = self._to_plain(calendar_events)

                # Add last 5 recent observations for CSV only (do not modify AI context)
                recent_obs_serialized = []
                try:
                    recent_obs = await self.gum.recent_observations(limit=5)
                    for o in recent_obs:
                        created = getattr(o, "created_at", None)
                        if isinstance(created, datetime):
                            created = created.isoformat()
                        recent_obs_serialized.append({
                            "observer_name": getattr(o, "observer_name", ""),
                            "content": getattr(o, "content", ""),
                            "content_type": getattr(o, "content_type", ""),
                            "created_at": created if created is not None else "",
                        })
                except Exception:
                    recent_obs_serialized = []

                writer.writerow({
                    "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                    "screenshot_path": screenshot_path or "",
                    "user_name": user_name,
                    "user_details": json.dumps(user_details_plain, ensure_ascii=False),
                    "calendar_events": json.dumps(calendar_events_plain, ensure_ascii=False),
                    "recent_observations": json.dumps(recent_obs_serialized, ensure_ascii=False),
                    "context_update": context,
                    "goals": json.dumps(goals_plain, ensure_ascii=False),
                    "reasoning": res.reasoning
                })
        
        
        return res.goals, res.reasoning