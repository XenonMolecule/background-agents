from __future__ import annotations
from typing import List

from gum import gum
from cal import Calendar

import dspy
import pydantic

# pip install mss pillow pynput
from typing import Optional, Tuple
import mss
from PIL import Image
from pynput.mouse import Controller as MouseController
from datetime import timedelta

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

import pydantic
import dspy


class EventListener(pydantic.BaseModel):
    """A minimal event listener used to START work or REPORT results.

    type: one of
        - OnApplicationOpen
        - OnObserverState
        - OnDocumentOpen
        - OnGoogleDocumentOpen
        - AtTime
        - AtCalendarEvent

    match: a short, human-readable condition (e.g., "GitHub repo page visible",
           "User opens Resume in Google Docs", "10 minutes before Weekly Sync").
    """
    type: str
    match: str


class SelectAgentAndTriggers(dspy.Signature):
    """You are the **JiT Orchestrator**.

Given the **user context**, the **current screenshot**, and the **objectives**, your job is to decide:
1. Which **agent** should quietly handle the background task  
2. When to **start** that work safely  
3. When to **report** the results naturally  

---

## 🎯 Core Principles

- These agents perform **background side tasks**, not the user’s main objective.  
  They act silently in support — never in place of the user.  
  Examples include:
  - Reformatting a document for clarity  
  - Updating a README in a code repo  
  - Refreshing meeting notes before a meeting  

- **Never overwrite user work.** Always duplicate, copy, or branch before editing.

- **Wait for safe, idle moments.**  
  Run only when:
  - A document is opened and stable  
  - The user is not mid-typing or coding  
  - A meeting is about to start or end  
  Background assistance should feel seamless — never interruptive.

---

## 🤖 Agents

| Agent | Description | Example |
|--------|--------------|----------|
| **FileAgent** | Duplicates a local file and edits the copy. | Reformat or polish a `.docx` or `.txt` file. |
| **CodeAgent** | Clones a repo and edits the clone (opens a PR). | Update or fix documentation in a codebase. |
| **GoogleDriveAgent** | Copies a Google Doc/Sheet/Slide and edits the copy. | Update meeting notes or collaborative documents. |

---

## 🕓 Event Listener Types

| Listener | Description | When to Use |
|-----------|--------------|--------------|
| **OnApplicationOpen** | Fires when a specific app (VS Code, Word, Chrome) is opened. | For tasks linked to specific tools or environments. |
| **OnObserverState** | Fires when the visible screen matches a short natural-language description (“GitHub PR page visible”). | For web or UI-based triggers. |
| **OnDocumentOpen** | Fires when a local document is opened. | For FileAgent operations. |
| **OnGoogleDocumentOpen** | Fires when a Google Drive document is opened. | For GoogleDriveAgent workflows. |
| **AtTime** | Fires at a specific wall-clock time. | For scheduled or delayed background work. |
| **AtCalendarEvent** | Fires before or after a meeting. | For preparing or summarizing notes. |

---

## 🧠 Trigger Guidance

- **Start Triggers:**  
  Wait until the user is idle or the right context appears (e.g., file opened, meeting approaching).  

- **Report Triggers:**  
  Use natural checkpoints when the user would want to review progress (e.g., reopens doc, views PR, meeting start).  

- **Always assist, never interfere.**  
  The orchestrator’s role is supportive automation that improves flow, not automation that overrides intent.

---

## 🧾 Output Structure

The output should include the following key elements:

- **rationale** → Short explanations for why each should be chosen  
- **agent** → Which agent to use  
- **start_listener** → When to safely begin background work (type + trigger condition)  
- **report_listener** → When to report or present results (type + trigger condition)  


Represent these conceptually — not as rigid JSON — just structured and readable for downstream processing.

---

## 🪄 Examples

### Example 1 — Resume Polishing
**Context:** Editing resume in Google Docs  
**Objective:** Improve language and formatting  

**Output Example:**
- Rationale →  
  - The document is on Drive, so use GoogleDriveAgent.  
  - Start when the resume is opened and idle.  
  - Report when the user revisits the same doc for review.  
- Agent → GoogleDriveAgent  
- Start Listener → OnGoogleDocumentOpen (“User opens resume in Google Docs”)  
- Report Listener → OnGoogleDocumentOpen (“User reopens resume to review changes”)  

---

### Example 2 — Code Cleanup
**Context:** Working on a GitHub repository  
**Objective:** Refactor README and clean structure  

**Output Example:**
- Rationale →  
  - The task involves repo edits, so CodeAgent is appropriate.  
  - Begin when repo context is open and stable.  
  - Report when the user is in review mode (PR page).  
- Agent → CodeAgent  
- Start Listener → OnObserverState (“GitHub repo page visible”)  
- Report Listener → OnObserverState (“Pull Request page visible”)  

---

### Example 3 — Meeting Notes Update
**Context:** Weekly sync meeting scheduled  
**Objective:** Update last week’s notes with action items  

**Output Example:**
- Rationale →  
  - Notes live in Drive.  
  - Update just before the meeting.  
  - Present results when the meeting begins (15 minutes before for review).  
- Agent → GoogleDriveAgent  
- Start Listener → AtCalendarEvent (“30 minutes before Weekly Sync”)  
- Report Listener → AtCalendarEvent (15 minutes before Weekly Sync)  


---

Keep outputs concise, structured, and readable.  
Your goal is to **choose the best background agent** and **natural trigger points** for safe, helpful, non-intrusive support.
"""

    # Inputs
    user_context: str = dspy.InputField(desc="Short description of what the user is currently doing/focused on.")
    screenshot_context: dspy.Image = dspy.InputField(desc="The current screenshot image (not OCR).")
    objectives: list[str] = dspy.InputField(desc="Concise list of background-side objectives (e.g., reformat doc, update notes).")

    # Outputs
    rationale: dict[str, str] = dspy.OutputField(desc='Keys: "why_agent", "why_start_trigger", "why_report_trigger".')
    agent: str = dspy.OutputField(desc='One of: "FileAgent", "CodeAgent", or "GoogleDriveAgent".')
    start_listener: EventListener = dspy.OutputField(desc="When to safely START background work (type + match).")
    report_listener: EventListener = dspy.OutputField(desc="When to REPORT results for review (type + match).")
    


class ObjectiveInducer():

    def __init__(self, gum: gum, calendar: Calendar):
        self.gum = gum
        self.calendar = calendar

        self._induce_with_screenshot = dspy.ChainOfThought(InduceObjectivesWithScreenshot.with_instructions(GOAL_INDUCTION__PROMPT))
        self._induce_without_screenshot = dspy.ChainOfThought(InduceObjectives.with_instructions(GOAL_INDUCTION__PROMPT))

        self._select_agent_and_triggers = dspy.ChainOfThought(SelectAgentAndTriggers)

    async def _get_context(self, context: str) -> str:
        user_name = self.gum.user_name

        user_details = await self.gum.query(str(context)) # TODO: Might be better to make this more clever
        calendar_events = self.calendar.query_str(start_delta=timedelta(days=0), end_delta=timedelta(days=1))

        context_str = f"User: {user_name}\nUser Details: {user_details}\nCalendar Events: {calendar_events}\nCurrent Context Update: {context}"

        return context_str

    async def induce_with_screenshot(self, context: str, limit: int = 3) -> List[Goal]:
        print("!!!Inducing with screenshot")
        screenshot = dspy.Image.from_PIL(grab_screen_at_mouse())
        context = await self._get_context(context)
        res = self._induce_with_screenshot(context=context, screenshot=screenshot, limit=limit)
        return res.goals, res.reasoning

    async def induce_without_screenshot(self, context: str, limit: int) -> List[Goal]:
        print("!!!Inducing without screenshot")
        context = await self._get_context(context)
        res = self._induce_without_screenshot(context=context, limit=limit)
        return res.goals, res.reasoning

    async def select_agent_and_triggers(self, context: str, goal_limit: int = 3) -> dict:
        print("!!!Selecting agent and triggers")
        screenshot = dspy.Image.from_PIL(grab_screen_at_mouse())
        context = await self._get_context(context)
        res = self._induce_with_screenshot(context=context, screenshot=screenshot, limit=goal_limit)

        stringified_goals = [goal.model_dump_json() for goal in res.goals]

        agent_res = self._select_agent_and_triggers(user_context=context, screenshot_context=screenshot, objectives=stringified_goals)
        return agent_res.rationale, agent_res.agent, agent_res.start_listener, agent_res.report_listener, res.reasoning, res.goals