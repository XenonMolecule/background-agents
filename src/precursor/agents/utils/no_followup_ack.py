"""
Random "no follow-up question" acknowledgements for the interviewer agent.

Goal
----
When the interviewer agent decides it has no clarifying question to ask (returns None),
we still want the user to receive a short acknowledgement so the interaction doesn't
feel silent/stalled.

Implementation notes
--------------------
- We keep this module dependency-light so it can be unit tested without DSPy/MCP.
- Selection is intentionally random; occasional repetition is acceptable.
"""

from __future__ import annotations

import random


# Keep these short (2–3 sentences max) and NEVER include questions.
NO_FOLLOWUP_ACKS: list[str] = [
    "Thanks — I think I have enough context for now. I’ll ask if anything becomes unclear later.",
    "Got it. I don’t have any follow-up questions right now, but I’ll flag anything that needs clarification later.",
    "Understood. I’m good on context for the moment and will reach out if something doesn’t add up later.",
    "This makes sense to me so far. I’ll follow up only if I hit an ambiguity later.",
    "Thanks for the update — I think I understand the project state right now. I’ll ask if I need to resolve any uncertainty later.",
    "I’m aligned on the current state. If anything seems underspecified later, I’ll ask a targeted question then.",
    "Noted. I’m not missing anything obvious at the moment; I’ll check back if a key detail is ambiguous later.",
    "All clear on my end for now. I’ll ask later only if a specific decision point requires it.",
    "Great, I think I’ve got the context I need. I’ll follow up if a concrete ambiguity comes up later.",
    "Understood — I’m satisfied with the current context. I’ll reach out if I’m unsure about something later.",
    "Thanks, I’m oriented. I’ll only ask if a future update creates a real fork in interpretation.",
    "This is enough for now. I’ll request clarification later if it would materially change how objectives should be structured.",
    "Makes sense. I’ll stay quiet unless I encounter a high-impact ambiguity later.",
    "I think I’m good for now. I’ll ask again if something would change prioritization or classification of the objectives.",
    "I’m tracking the current intent and context. I’ll check in if I need one crisp clarification later.",
    "Thanks — I’ve got a coherent picture right now. I’ll follow up if something is missing that would change the framing.",
    "Understood. No questions at the moment; I’ll raise one later if it’s truly necessary.",
    "I’m good on context for now. If something becomes ambiguous later, I’ll ask a single focused question then.",
    "Thanks, this is clear enough to proceed. I’ll ask later if a key assumption needs to be confirmed.",
    "All set for now. I’ll reach out if a specific missing detail would change how the shared context should be updated.",
]


