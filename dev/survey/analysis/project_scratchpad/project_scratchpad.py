import dspy
import argparse
import csv
import json
import difflib
import re
import os
import shutil
from typing import Literal, Optional, List
import pydantic

from dotenv import load_dotenv
from PIL import Image as PILImage
from tqdm import tqdm

TRUE_PROJECTS = [
    "Personalization Dataset Collection",
    "AutoMetrics Release",
    "NLP Retreat Planning",
    "Background Agents",
    "Logistics",
    "Health",
    "HW3 Assignment Creation",
    "Misc",
]

SECTIONS = [
    "Ongoing Objectives",
    "Completed Objectives",
    "Suggestions",
    "Notes",
    "Files, Repos, Folders, Collaborators, and Other Relevant Resources",
    "Next Steps",
]

class ProjectScratchpad():
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.sections = {section: [] for section in SECTIONS}

    def __repr__(self):
        return f"""# {self.project_name}

## Ongoing Objectives
{'\n'.join([f"[{idx}] {objective[0]} (confidence: {objective[1]})" for idx, objective in enumerate(self.sections["Ongoing Objectives"])]) or "None"}

## Completed Objectives
{'\n'.join([f"[{idx}] {objective[0]} (confidence: {objective[1]})" for idx, objective in enumerate(self.sections["Completed Objectives"])]) or "None"}

## Suggestions
{'\n'.join([f"[{idx}] {suggestion[0]} (confidence: {suggestion[1]})" for idx, suggestion in enumerate(self.sections["Suggestions"])]) or "None"}

## Notes
{'\n'.join([f"[{idx}] {note[0]} (confidence: {note[1]})" for idx, note in enumerate(self.sections["Notes"])]) or "None"}

## Files, Repos, Folders, Collaborators, and Other Relevant Resources
{'\n'.join([f"[{idx}] {resource[0]} (confidence: {resource[1]})" for idx, resource in enumerate(self.sections["Files, Repos, Folders, Collaborators, and Other Relevant Resources"])]) or "None"}

## Next Steps
{'\n'.join([f"[{idx}] {step[0]} (confidence: {step[1]})" for idx, step in enumerate(self.sections["Next Steps"])]) or "None"}"""

    def __str__(self):
        return self.__repr__()

    def append_to_scratchpad(self, section: str, proposition_text: str, confidence: int = 0):
        self.sections[section].append((proposition_text, confidence))
        return f"Added new proposition to the project scratchpad: {proposition_text}\n\n== UPDATED SCRATCHPAD ==\n{self.__repr__()}"

    def remove_from_scratchpad(self, section: str, index: int):
        old_note, _ = self.sections[section][index]
        self.sections[section].pop(index)
        return f"Removed proposition from the project scratchpad: {old_note}\n\n== UPDATED SCRATCHPAD ==\n{self.__repr__()}"

    def edit_in_scratchpad(self, section: str, index: int, new_proposition_text: str, new_confidence: int = 0):
        old_proposition_text, _ = self.sections[section][index]
        self.sections[section][index] = (new_proposition_text, new_confidence)
        return f"Edited proposition in the project scratchpad: {old_proposition_text} -> {new_proposition_text}\n\n== UPDATED SCRATCHPAD ==\n{self.__repr__()}"

    def get_scratchpad(self):
        return self.__repr__()

scratchpads = {project_name: ProjectScratchpad(project_name) for project_name in TRUE_PROJECTS}

# Debug instrumentation for tool calls
TOOL_CALL_LOG: list = []

def _tool_log_reset():
    global TOOL_CALL_LOG
    TOOL_CALL_LOG = []

def append_to_scratchpad(project_name: str, section: str, proposal_text: str, confidence: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 0):
    """Add a brand new note/observation to the project scratchpad.  The project name should be the exact match for the project name.  The section should be one of `Ongoing Objectives`, `Completed Objectives`, `Suggestions`, `Notes`, `Files, Repos, Folders, Collaborators, and Other Relevant Resources`, or `Next Steps`.
    
    The confidence should be a number between 0 and 10, where 0 is the lowest confidence and 10 is the highest confidence in the proposition."""
    # print(f"Appending to scratchpad: {project_name}, {section}, {note}")

    # If the section is one of the subsets of "Files, Repos, Folders, Collaborators, and Other Relevant Resources" then we should add it to the "Files, Repos, Folders, Collaborators, and Other Relevant Resources" section
    if "file" in section.lower() or "repo" in section.lower() or "folder" in section.lower() or "collaborator" in section.lower() or "relevant resource" in section.lower():
        section = "Files, Repos, Folders, Collaborators, and Other Relevant Resources"

    def _clean_proposition_text(text: str) -> str:
        t = str(text or "").strip()
        # Remove any trailing "(confidence: N)" segments, possibly repeated
        while True:
            new_t = re.sub(r"\s*\(confidence\s*:\s*\d+\)\s*$", "", t, flags=re.IGNORECASE)
            if new_t == t:
                break
            t = new_t.strip()
        # Remove leading enumerations like "1. ", "2) ", "- "
        t = re.sub(r"^\s*(?:\d+[\.)]\s+|[-•]\s+)", "", t)
        return t.strip()

    proposal_text = _clean_proposition_text(proposal_text)

    # If the model provided a numbered list, split into multiple propositions (robust regex)
    # Split anywhere a whitespace + number + '.' or ')' + whitespace appears (e.g., " 2. ", " 3) ")
    parts = re.split(r"\s+\d+[\.)]\s+", proposal_text)
    parts = [p.strip() for p in parts if p and p.strip()]

    # If no numbered items found, try splitting by newline-based resource titles like "Title: ..."
    if len(parts) <= 1:
        lines = [ln.strip() for ln in proposal_text.splitlines() if ln.strip()]
        # Identify lines that look like Title: description (starts with Capital and contains colon)
        title_colon = re.compile(r"^[A-Z][A-Za-z0-9 .,'/()&_-]*:\s+")
        if sum(1 for ln in lines if title_colon.match(ln)) >= 2:
            parts = lines
    # If still single, split bullet lines that start with '-' or '•'
    if len(parts) <= 1:
        lines = [ln.strip() for ln in proposal_text.splitlines() if ln.strip()]
        bullet_re = re.compile(r"^[\-•]\s+")
        if any(bullet_re.match(ln) for ln in lines):
            parts = []
            if lines and not bullet_re.match(lines[0]):
                parts.append(lines[0])
            parts.extend([bullet_re.sub("", ln, count=1).strip() for ln in lines if bullet_re.match(ln)])
    # If still single, split on semicolons that separate multiple items
    if len(parts) <= 1 and " ; " in proposal_text:
        parts = [p.strip() for p in proposal_text.split(" ; ") if p and p.strip()]

    # De-dup exact matches within section
    existing = set(x for (x, _) in scratchpads[project_name].sections.get(section, []))

    added = 0
    if len(parts) > 1:
        for it in parts:
            it_clean = _clean_proposition_text(it)
            if not it_clean or it_clean in existing:
                continue
            scratchpads[project_name].append_to_scratchpad(section, it_clean, confidence)
            existing.add(it_clean)
            added += 1
        return f"Added {added} propositions to the project scratchpad" if added else "No new propositions added (all duplicates)" + f"\n\n== UPDATED SCRATCHPAD ==\n{scratchpads[project_name].get_scratchpad()}"
    else:
        cleaned = _clean_proposition_text(parts[0] if parts else proposal_text)
        if not cleaned or cleaned in existing:
            return "No new propositions added (duplicate or empty)" + f"\n\n== UPDATED SCRATCHPAD ==\n{scratchpads[project_name].get_scratchpad()}"
        return scratchpads[project_name].append_to_scratchpad(section, cleaned, confidence)

def remove_from_scratchpad(project_name: str, section: str, index: int):
    """Remove a note/observation from the project scratchpad.  The project name should be the exact match for the project name.  The section should be one of `Ongoing Objectives`, `Completed Objectives`, `Suggestions`, `Notes`, `Files, Repos, Folders, Collaborators, and Other Relevant Resources`, or `Next Steps`."""
    # print(f"Removing from scratchpad: {project_name}, {section}, {index}")

    # If the section is one of the subsets of "Files, Repos, Folders, Collaborators, and Other Relevant Resources" then we should add it to the "Files, Repos, Folders, Collaborators, and Other Relevant Resources" section
    if "file" in section.lower() or "repo" in section.lower() or "folder" in section.lower() or "collaborator" in section.lower() or "relevant resource" in section.lower():
        section = "Files, Repos, Folders, Collaborators, and Other Relevant Resources"

    return scratchpads[project_name].remove_from_scratchpad(section, index)

def edit_in_scratchpad(project_name: str, section: str, index: int, new_proposition_text: str, new_confidence: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 0):
    """Edit a note/observation in the project scratchpad.  The project name should be the exact match for the project name.  The section should be one of `Ongoing Objectives`, `Completed Objectives`, `Suggestions`, `Notes`, `Files, Repos, Folders, Collaborators, and Other Relevant Resources`, or `Next Steps`.
    
    The confidence should be a number between 0 and 10, where 0 is the lowest confidence and 10 is the highest confidence in the proposition."""
    # print(f"Editing in scratchpad: {project_name}, {section}, {index}, {new_note}")

    # If the section is one of the subsets of "Files, Repos, Folders, Collaborators, and Other Relevant Resources" then we should add it to the "Files, Repos, Folders, Collaborators, and Other Relevant Resources" section
    if "file" in section.lower() or "repo" in section.lower() or "folder" in section.lower() or "collaborator" in section.lower() or "relevant resource" in section.lower():
        section = "Files, Repos, Folders, Collaborators, and Other Relevant Resources"

    def _clean_proposition_text(text: str) -> str:
        t = str(text or "").strip()
        while True:
            new_t = re.sub(r"\s*\(confidence\s*:\s*\d+\)\s*$", "", t, flags=re.IGNORECASE)
            if new_t == t:
                break
            t = new_t.strip()
        t = re.sub(r"^\s*(?:\d+[\.)]\s+|[-•]\s+)", "", t)
        return t.strip()

    cleaned = _clean_proposition_text(new_proposition_text)
    sect = scratchpads[project_name].sections.get(section, [])
    existing_other = set(txt for i, (txt, _) in enumerate(sect) if i != index)

    # Try multi-item splitting (numbered, title: lines, bullets)
    parts = re.split(r"\s+\d+[\.)]\s+", cleaned)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        title_colon = re.compile(r"^[A-Z][A-Za-z0-9 .,'/()&_-]*:\s+")
        if sum(1 for ln in lines if title_colon.match(ln)) >= 2:
            parts = lines
    if len(parts) <= 1:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        bullet_re = re.compile(r"^[\-•]\s+")
        if any(bullet_re.match(ln) for ln in lines):
            parts = []
            if lines and not bullet_re.match(lines[0]):
                parts.append(lines[0])
            parts.extend([bullet_re.sub("", ln, count=1).strip() for ln in lines if bullet_re.match(ln)])
    # If still single, split on semicolons that separate multiple items
    if len(parts) <= 1 and " ; " in cleaned:
        parts = [p.strip() for p in cleaned.split(" ; ") if p and p.strip()]

    if len(parts) <= 1:
        # Single item edit with dedup guard
        if cleaned in existing_other:
            return "No edit performed (duplicate of another entry)" + f"\n\n== UPDATED SCRATCHPAD ==\n{scratchpads[project_name].get_scratchpad()}"
        return scratchpads[project_name].edit_in_scratchpad(section, index, cleaned, new_confidence)

    # Multi-item: update current index with first unique, append the rest unique
    updated = False
    added = 0
    chosen = None
    for it in parts:
        it_clean = _clean_proposition_text(it)
        if not it_clean:
            continue
        if not updated and it_clean not in existing_other:
            scratchpads[project_name].edit_in_scratchpad(section, index, it_clean, new_confidence)
            existing_other.add(it_clean)
            updated = True
            chosen = it_clean
        elif it_clean not in existing_other:
            scratchpads[project_name].append_to_scratchpad(section, it_clean, new_confidence)
            existing_other.add(it_clean)
            added += 1

    if not updated:
        return "No edit performed (all items duplicate of existing entries)" + f"\n\n== UPDATED SCRATCHPAD ==\n{scratchpads[project_name].get_scratchpad()}"

    return f"Edited current item to '{chosen}' and added {added} more" + f"\n\n== UPDATED SCRATCHPAD ==\n{scratchpads[project_name].get_scratchpad()}"
    
def get_refreshed_scratchpad(project_name: str):
    """Get the refreshed project scratchpad.  The project name should be the exact match for the project name."""
    return scratchpads[project_name].get_scratchpad()

class ProjectResource(pydantic.BaseModel):
    name: str
    description: Optional[str] = None
    uri: Optional[str] = None

class ExtractProjectResourcesSignature(dspy.Signature):
    """Extract the **project resources** visible in the user’s current context and screenshot.

Your goal is to identify **files, folders, repositories, documents, or collaborators** that are *actually present* in the view. Each resource should include a short description and a **resource identifier** (e.g., a URL, file path, folder path, repository path, or collaborator email).

Do **not** invent or speculate about unseen resources. If no relevant resources are visible, return `NULL` or indicate that no resources were found.

Focus on **the most important resources** — high-level entities like repository names, shared document links, or project folders take priority over low-level details like individual code lines or small temporary files.

Think hierarchically about context:
- In a code editor, the repository or project folder is usually the most important resource.  
- Individual files belong to that repository; include the repository name as part of the identifier.  
- In collaborative tools, collaborator names or emails are key resources.  

When deciding what to extract, ask:
- What would another person need to locate or reference this project later?
- What entities define *where* and *with whom* the work is happening?

If the user appears to be in a **code editor**, the first step should always be to deduce the **repository or top-level project folder name**, and ensure it has been recorded before extracting lower-level resources.  This is often shown at the top of the code editor and is CRITICAL!!!"""
    current_project_name: Literal[tuple(TRUE_PROJECTS)] = dspy.InputField(description="The name of the project that the user is most likely currently working on (may be inaccurate)")
    current_project_scratchpad: str = dspy.InputField(description="The current project scratchpad")
    user_context: str = dspy.InputField(description="A short description of the user's current context, including what they are doing, what they are focused on, what they are thinking about, what they are working on, etc.")
    current_screenshot: dspy.Image = dspy.InputField(description="The screenshot of the user's current workspace")
    project_resources: list[ProjectResource] = dspy.OutputField(description="A list of up to 10 project resources that you observe in the context or screenshot that MAY be relevant to the project which aren't already in the project scratchpad")

class ExtractProjectResources(dspy.Module):
    def __init__(self):
        self.extract_project_resources = dspy.ChainOfThought(
            ExtractProjectResourcesSignature,
        )
    
    def forward(self, current_project_name: str, current_project_scratchpad: str, user_context: str, current_screenshot: dspy.Image) -> list[ProjectResource]:
        
        res = self.extract_project_resources(
            current_project_name=current_project_name,
            current_project_scratchpad=current_project_scratchpad,
            user_context=user_context,
            current_screenshot=current_screenshot,
        )

        # Convert to string
        return "\n".join([f"- {resource.name}: {resource.description} (uri: {resource.uri})" for resource in res.project_resources])

class EditProjectScratchpadSignature(dspy.Signature):
    """Based on the new information, edit the project scratchpad.  The scratchpad should be an ongoing log of the user's work on the project.

    Guidance for how to update sections:
    - If you believe an objective is currently being worked on, add it to `Ongoing Objectives`.
    - If you have some evidence that an objective is finished, move it to `Completed Objectives`.
    - Add `Suggestions` when there are helpful ideas, tools, or process improvements to recommend.
    - Use `Notes` for context, scope clarifications, constraints, or assumptions that aid understanding.
    - Use `Files, Repos, Folders, Collaborators, and Other Relevant Resources` for files, repos, folders, collaborators, or other resources that are relevant to the project. This section is SUPER IMPORTANT!  Please try to add as much detail here as possible (links if you are confident, file paths as best you can figure out, etc.)
    - Derive 1–3 actionable `Next Steps` from the current/former objectives whenever feasible (avoid trivial steps).

    Aim for diversity across sections when adding content (do not only update one section). Avoid duplicates and keep entries concise and useful. If there is no clear new information, it is reasonable to leave the scratchpad and maybe just make a note for the notes section.

    It is important to add notes and suggestions and next steps!  ALSO it is a good idea to remove objectives that are no longer ongoing and move them to a completed objective.  If a "next step" grows out of date or is completed, you should remove it or move it to a completed objective.

    Please try to add a confidence score to each objective, suggestion, note, and next step.  The confidence score should be a number between 0 and 10, where 0 is the lowest confidence and 10 is the highest confidence.  The confidence score should be based on the evidence you have for the proposition.

    You may consider improving the confidence score of a previously added proposition if you get more evidence for it!
    A good strategy is to start with a lower confidence score (2-3) and then increase it as you get more evidence (5-7) until you are very confident (9-10).

    PLEASE ONLY ADD ONE PROPOSITION AT A TIME!!!  If you have multiple propositions to add/edit/remove, you should call the tools multiple times, once per proposition instead of trying to do a numbered list.

    When adding project resources, try to add a uri if you have one!  This can be a link to the resource, a file path, a folder path, a collaborator's email, etc.  Do not make up a uri if you do not have one and note that the provided uri is a guess and should be treated as such. 
    Project repo names/folders are often THE MOST IMPORTANT resources to add to the scratchpad, so make sure to ALWAYS include a repo name/folder name if you can!  Tracking this is CRITICAL!
    """

    current_project_name: Literal[tuple(TRUE_PROJECTS)] = dspy.InputField(description="The name of the project that the user is most likely currently working on (may be inaccurate)")
    current_project_scratchpad: str = dspy.InputField(description="The current project scratchpad")
    speculated_current_objectives: list[str] = dspy.InputField(description="A list of objectives that we think the user might be working on (may be inaccurate)")
    speculated_former_objectives: list[str] = dspy.InputField(description="A list of objectives that we think the user might have been working on (may be inaccurate)")
    calendar_events: list[str] = dspy.InputField(description="A list of upcoming calendar events that the user has scheduled (may or may not be relevant to this project, you should consider that they may be more about other projects)")
    full_project_list: list[str] = dspy.InputField(description="A list of all the projects that the user has provided that they are working on at any point in time")
    user_context: str = dspy.InputField(description="A short description of the user's current context, including what they are doing, what they are focused on, what they are thinking about, what they are working on, etc.")
    potential_resources: str = dspy.InputField(description="A list of potential project resources that we think the user might be working on (may be inaccurate or related to other projects besides this one)")
    current_screenshot: dspy.Image = dspy.InputField(description="The screenshot of the user's current workspace")
    summary_of_edits: str = dspy.OutputField(description="A summary of the edits you made to the project scratchpad")

class EditProjectScratchpad(dspy.Module):
    def __init__(self):
        self.edit_project_scratchpad = dspy.ReAct(
            EditProjectScratchpadSignature,
            tools=[append_to_scratchpad, remove_from_scratchpad, edit_in_scratchpad, get_refreshed_scratchpad],
            max_iters=20,
        )
    
    def forward(self, current_project_name: str, current_project_scratchpad: str, speculated_current_objectives: list[str], speculated_former_objectives: list[str], calendar_events: list[str], full_project_list: list[str], user_context: str, potential_resources: str, current_screenshot: dspy.Image):
        res = self.edit_project_scratchpad(
            current_project_name=current_project_name,
            current_project_scratchpad=current_project_scratchpad,
            speculated_current_objectives=speculated_current_objectives,
            speculated_former_objectives=speculated_former_objectives,
            calendar_events=calendar_events,
            full_project_list=full_project_list,
            user_context=user_context,
            potential_resources=potential_resources,
            current_screenshot=current_screenshot,
        )
        return res.summary_of_edits, scratchpads[current_project_name].get_scratchpad()

def _safe_json_parse(raw: str, fallback: Optional[str] = "") -> str:
    if not raw:
        return fallback or ""
    s = str(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, dict)):
            return json.dumps(obj, ensure_ascii=False)
        return str(obj)
    except Exception:
        return s


def _extract_goals(row: dict, top_k: int = 3) -> List[str]:
    raw = row.get("goals", "")
    try:
        goals = json.loads(raw) if raw else []
    except Exception:
        goals = []
    results: List[str] = []
    if isinstance(goals, list):
        for g in goals[: max(0, top_k)]:
            if isinstance(g, dict):
                title = str(g.get("title") or g.get("name") or "Goal")
                desc = str(g.get("description") or g.get("desc") or "")
                results.append(f"{title}: {desc}" if desc else title)
            else:
                results.append(str(g))
    else:
        s = _safe_json_parse(raw)
        if s:
            results.append(s)
    return results


# Dropped propositions for simplicity; focus on objectives only


def _extract_calendar_list(row: dict) -> List[str]:
    raw = row.get("calendar_events", "")
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            return [json.dumps(obj, ensure_ascii=False)]
        return [str(obj)]
    except Exception:
        return [str(raw)] if str(raw) else []


def _extract_screenshot(row: dict) -> str:
    return str(row.get("screenshot_path", ""))


def _load_dspy_image(path: str) -> dspy.Image:
    try:
        if path and os.path.exists(path):
            pil = PILImage.open(path)
            pil = pil.convert("RGB")
            pil.load()
            max_side = 1600
            pil.thumbnail((max_side, max_side))
            return dspy.Image.from_PIL(pil)
    except Exception:
        pass
    return dspy.Image.from_PIL(PILImage.new("RGB", (1, 1), "white"))


def _parse_epoch(ts: str) -> int:
    if not ts:
        return 0
    s = str(ts)
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    if len(s) == 15 and s[8] == "_":
        try:
            from datetime import datetime

            y = int(s[0:4]); mo = int(s[4:6]); da = int(s[6:8])
            hh = int(s[9:11]); mm = int(s[11:13]); ss = int(s[13:15])
            dt = datetime(y, mo, da, hh, mm, ss)
            return int(dt.timestamp() * 1000)
        except Exception:
            return 0
    return 0


def run_sequential(
    context_log_csv: str,
    predictions_csv: str,
    output_csv: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    current_goals_top_k: int = 3,
    former_goals_window: int = 10,
    former_top_per_row: int = 1,
    former_max_total: int = 10,
    limit: Optional[int] = None,
    debug_history: bool = False,
) -> None:
    # Load .env
    try:
        load_dotenv()
    except Exception:
        pass

    # Configure LM
    model_str = model or os.environ.get("DSPY_MODEL")
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DSPY_API_KEY")
    dspy.configure(lm=dspy.LM(model_str, api_key=key, temperature=1.0, max_tokens=16000))
    if temperature is not None:
        try:
            lm = dspy.settings.lm
            if hasattr(lm, "kwargs"):
                lm.kwargs["temperature"] = temperature
        except Exception:
            pass

    module = EditProjectScratchpad()
    extract_project_resources = ExtractProjectResources()

    # Ensure output dir
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except Exception:
        pass

    # Load predictions → timestamp -> project
    preds_map = {}
    try:
        with open(predictions_csv, "r", newline="", encoding="utf-8") as f_pred:
            reader = csv.DictReader(f_pred)
            for r in reader:
                ts = str(r.get("timestamp", ""))
                proj = str(r.get("predicted_project", ""))
                if ts:
                    preds_map[ts] = proj
    except Exception:
        preds_map = {}

    # Read context rows and sort sequentially
    with open(context_log_csv, "r", newline="", encoding="utf-8") as f_in, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        rows.sort(key=lambda r: _parse_epoch(str(r.get("timestamp", ""))))
        if limit is not None and limit >= 0:
            rows = rows[:limit]
        writer = csv.DictWriter(
            f_out,
            fieldnames=["timestamp", "project", "scratchpad", "summary", "model"],
        )
        writer.writeheader()

        # For each row, sequentially update the scratchpad for the predicted project
        processed = 0
        for i, row in enumerate(tqdm(rows, desc="Updating scratchpads", unit="row")):
            ts = str(row.get("timestamp", ""))
            if not ts:
                continue
            pred = preds_map.get(ts) or "Misc"
            project = pred if pred in TRUE_PROJECTS else "Misc"

            # Build inputs
            current_pad = scratchpads[project].get_scratchpad()
            # Current objectives from THIS row
            speculated_current_objectives: List[str] = _extract_goals(row, top_k=current_goals_top_k)
            # Former objectives: gather from past rows (newest→oldest), up to former_max_total
            speculated_former_objectives: List[str] = []
            if former_goals_window > 0 and former_max_total > 0:
                start = max(0, i - former_goals_window)
                past_rows = rows[start:i][::-1]
                for r in past_rows:
                    if len(speculated_former_objectives) >= former_max_total:
                        break
                    tops = _extract_goals(r, top_k=former_top_per_row)
                    for t in tops:
                        if len(speculated_former_objectives) >= former_max_total:
                            break
                        speculated_former_objectives.append(t)
            cal_list = _extract_calendar_list(row)
            screenshot_img = _load_dspy_image(_extract_screenshot(row))

            # Run edit step
            _tool_log_reset()
            potential_resources = extract_project_resources(
                current_project_name=project,
                current_project_scratchpad=current_pad,
                user_context=row.get("context", ""),
                current_screenshot=screenshot_img,
            )

            # print(f"[scratchpad] potential_resources: {potential_resources}")

            summary, updated_pad = module(
                current_project_name=project,
                current_project_scratchpad=current_pad,
                speculated_current_objectives=speculated_current_objectives,
                speculated_former_objectives=speculated_former_objectives,
                calendar_events=cal_list,
                full_project_list=TRUE_PROJECTS,
                user_context=row.get("context", ""),
                potential_resources=potential_resources,
                current_screenshot=screenshot_img,
            )

            # Optional: inspect model history and tool calls for debugging
            if debug_history:
                print("\n[scratchpad][history] ----------------------------------------")
                print(f"timestamp={ts} project={project}")
                try:
                    # Print model inputs (truncated)
                    dbg_inputs = {
                        "current_project_name": project,
                        "current_project_scratchpad_len": len(current_pad or ""),
                        "speculated_current_objectives": speculated_current_objectives,
                        "speculated_former_objectives_top3": speculated_former_objectives[:3],
                        "calendar_events_top3": cal_list[:3],
                    }
                    print("[scratchpad][inputs]", json.dumps(dbg_inputs, ensure_ascii=False))
                except Exception:
                    pass
                try:
                    dspy.settings.lm.inspect_history(n=10)
                except Exception:
                    pass
                try:
                    if TOOL_CALL_LOG:
                        print("[scratchpad][tools]", json.dumps(TOOL_CALL_LOG, ensure_ascii=False))
                    else:
                        print("[scratchpad][tools] (no tool calls recorded)")
                except Exception:
                    pass
                try:
                    # Diff of scratchpad (first 60 lines)
                    a = (current_pad or "").splitlines()
                    b = (updated_pad or "").splitlines()
                    diff = list(difflib.unified_diff(a, b, lineterm=""))
                    preview = "\n".join(diff[:60])
                    print("[scratchpad][diff-preview]\n" + (preview or "(no diff)"))
                except Exception:
                    pass
                try:
                    print("[scratchpad][summary]", (summary or "").strip())
                except Exception:
                    pass

            # Fallback: if no changes and we have current objectives, auto-add them as Possible Ongoing Objectives
            if updated_pad == current_pad and speculated_current_objectives:
                try:
                    for obj in speculated_current_objectives:
                        scratchpads[project].append_to_scratchpad("Possible Ongoing Objectives", obj)
                    updated_pad = scratchpads[project].get_scratchpad()
                    auto_note = f"Auto-added {len(speculated_current_objectives)} current objectives (no agent edits)."
                    summary = (summary + "\n" + auto_note).strip() if summary else auto_note
                except Exception:
                    pass

            # Light debug: warn when no objectives found
            if not speculated_current_objectives and not speculated_former_objectives:
                print(f"[scratchpad] Warning: no objectives found for ts={ts} proj={project}")

            # Write snapshot
            lm_obj = getattr(dspy.settings, "lm", None)
            resolved_model = model or os.environ.get("DSPY_MODEL")
            if not resolved_model:
                resolved_model = getattr(lm_obj, "name", None) or (lm_obj if isinstance(lm_obj, str) else None)
            model_name = resolved_model or "unspecified-model"
            writer.writerow(
                {
                    "timestamp": ts,
                    "project": project,
                    "scratchpad": updated_pad,
                    "summary": summary,
                    "model": model_name,
                }
            )
            processed += 1

    # Also write/update a stable alias for the viewer
    try:
        out_dir = os.path.dirname(output_csv)
        latest_path = os.path.join(out_dir, "project_scratchpad_latest.csv")
        shutil.copyfile(output_csv, latest_path)
        alias_note = f" and alias {latest_path}"
    except Exception:
        latest_path = None
        alias_note = ""

    print(f"[scratchpad] Done. Wrote {processed} rows to {output_csv}{alias_note}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run scratchpad editor sequentially over context_log.csv and project predictions")
    parser.add_argument("--context-log", type=str, default=os.path.join("dev", "survey", "context_log.csv"))
    parser.add_argument(
        "--predicted-project-csv",
        type=str,
        default=os.path.join("dev", "survey", "analysis", "project_classification", "results", "context_project_predictions_openai_gpt-5_with_history.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV. If omitted, a default file including the model name will be used.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of context rows to process (sequential)")
    parser.add_argument("--model", default=os.environ.get("DSPY_MODEL"), help="LM name for dspy.configure (e.g., openai/gpt-4o, openai/gpt-5-mini)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key for the LM (env .env OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--current-goals-top-k", type=int, default=3, help="Top-k objectives from the current row")
    parser.add_argument("--former-goals-window", type=int, default=10, help="Number of previous rows to consider for former goals")
    parser.add_argument("--former-top-per-row", type=int, default=1, help="Top-k objectives to extract per past row")
    parser.add_argument("--former-max-total", type=int, default=10, help="Maximum number of former objectives to include")
    parser.add_argument("--debug-history", action="store_true", help="Print DSPy ReAct history for each row")
    args = parser.parse_args()

    # Derive default output path with model name if not provided
    if not args.output_csv:
        model_name = args.model or os.environ.get("DSPY_MODEL", "model")
        safe = model_name.replace("/", "_").replace(":", "_")
        base = f"project_scratchpad_{safe}.csv"
        args.output_csv = os.path.join("dev", "survey", "analysis", "project_scratchpad", "results", base)
    return args

def main():
    args = parse_args()
    run_sequential(
        context_log_csv=args.context_log,
        predictions_csv=args.predicted_project_csv,
        output_csv=args.output_csv,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        current_goals_top_k=args.current_goals_top_k,
        former_goals_window=args.former_goals_window,
        former_top_per_row=args.former_top_per_row,
        former_max_total=args.former_max_total,
        limit=args.limit,
        debug_history=args.debug_history,
    )

if __name__ == "__main__":
    main()