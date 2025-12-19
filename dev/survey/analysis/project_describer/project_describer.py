import dspy
import argparse
import csv
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from PIL import Image as PILImage
import re
from datetime import datetime
try:
    from dotenv import load_dotenv  # optional; OK if not installed
except Exception:
    def load_dotenv() -> None:
        return

class ProjectDescriber(dspy.Signature):
    """
Given some details about a project, the current work of the user on that project, and our existing notes about this project update the project description to list only of the HIGH LEVEL goals and objectives for the project.

The project description should NOT list all of the small details and tasks that the user is working on, but rather the important long term goals and objectives for the project.

Useful examples of high level goals and objectives are:
- "Advance the project toward a publishable scholarly contribution."
- "Embed interdisciplinary perspectives to enrich the project's approach."
- "Advance the project’s visibility within relevant communities."
- "Optimize the project’s workflow, processes, and decision-making systems."

The project descriptions should contain ALL high level goals for a project without getting too specific.
    """
    project_name: str = dspy.InputField(description="The name of the project that the user is most likely currently working on")
    context_update: str = dspy.InputField(description="A short description of the user's current context from a screenshot of their current workspace")
    screenshot: dspy.Image = dspy.InputField(description="A screenshot of the user's current workspace")
    project_scratchpad: str = dspy.InputField(description="The current project scratchpad with all the notes we have taken about the project")
    project_description: str = dspy.InputField(description="The existing description of the project")
    updated_project_description: str = dspy.OutputField(description="A description of the project focusing on the high level goals and objectives for the project")

class ProjectDescriberNoImage(dspy.Signature):
    """
Given some details about a project, the current work of the user on that project, and our existing notes about this project update the project description to list only of the HIGH LEVEL goals and objectives for the project.

The project description should NOT list all of the small details and tasks that the user is working on, but rather the important long term goals and objectives for the project.

Useful examples of high level goals and objectives are:
- "Advance the project toward a publishable scholarly contribution."
- "Embed interdisciplinary perspectives to enrich the project's approach."
- "Advance the project’s visibility within relevant communities."
- "Optimize the project’s workflow, processes, and decision-making systems."

The project descriptions should contain ALL high level goals for a project without getting too specific.
    """
    project_name: str = dspy.InputField(description="The name of the project that the user is most likely currently working on")
    context_update: str = dspy.InputField(description="A short description of the user's current context from a screenshot transcript or other text")
    project_scratchpad: str = dspy.InputField(description="The current project scratchpad with all the notes we have taken about the project")
    project_description: str = dspy.InputField(description="The existing description of the project")
    updated_project_description: str = dspy.OutputField(description="A description of the project focusing on the high level goals and objectives for the project")

class ProjectDescriberModule(dspy.Module):

    def __init__(self):
        self.project_describer = dspy.ChainOfThought(ProjectDescriber)
        self.project_describer_noimg = dspy.ChainOfThought(ProjectDescriberNoImage)

    def forward(self, project_name: str, context_update: str, screenshot: dspy.Image, project_scratchpad: str, project_description: str) -> str:
        res = self.project_describer(project_name=project_name, context_update=context_update, screenshot=screenshot, project_scratchpad=project_scratchpad, project_description=project_description)
        return getattr(res, "updated_project_description", "N/A")

    def describe_with_image(self, project_name: str, context_update: str, screenshot: dspy.Image, project_scratchpad: str, project_description: str) -> str:
        res = self.project_describer(project_name=project_name, context_update=context_update, screenshot=screenshot, project_scratchpad=project_scratchpad, project_description=project_description)
        return getattr(res, "updated_project_description", "N/A")

    def describe_text_only(self, project_name: str, context_update: str, project_scratchpad: str, project_description: str) -> str:
        res = self.project_describer_noimg(project_name=project_name, context_update=context_update, project_scratchpad=project_scratchpad, project_description=project_description)
        return getattr(res, "updated_project_description", "N/A")

# Note that DSPy has a batch functionality
"""
Here is an example of how to use the batch functionality:

import dspy

examples = [
    dspy.Example(project_name="Project 1", context_update="Context update 1", screenshot=dspy.Image.from_url("https://example.com/screenshot1.png"), project_scratchpad="Project scratchpad 1", project_description="Project description 1"),
    dspy.Example(project_name="Project 2", context_update="Context update 2", screenshot=dspy.Image.from_url("https://example.com/screenshot2.png"), project_scratchpad="Project scratchpad 2", project_description="Project description 2"),
]

outputs = self.project_describer.batch(examples)
for out in outputs:
    print(getattr(out, "updated_project_description", "N/A"))
"""

def _load_dspy_image(path: str) -> dspy.Image:
    """
    Load an image path into a dspy.Image via PIL; fall back to a tiny white image.
    Mirrors helpers used elsewhere in this repo for consistency.
    """

    if path and os.path.exists(path):
        pil = PILImage.open(path)
        pil = pil.convert("RGB")
        pil.load()
        max_side = 1600
        pil.thumbnail((max_side, max_side))
        return dspy.Image.from_PIL(pil)

    raise ValueError(f"Image path {path} does not exist")

    # Do NOT fabricate a fallback image; return None-equivalent by raising to callers.
    # Callers should decide how to handle missing screenshots explicitly.
    return None  # type: ignore[return-value]


def _read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def _build_screenshot_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    """
    Build map: (project, normalized_ts_variant) -> screenshot_path
    We index multiple timestamp variants to handle tz/millis formatting differences.
    """
    idx: Dict[Tuple[str, str], str] = {}
    for r in rows:
        ts = str(r.get("timestamp", "")).strip()
        proj = str(r.get("project", "")).strip()
        shot = str(r.get("screenshot_path", "")).strip()
        if proj and ts and shot:
            for v in _normalize_ts_variants(ts):
                idx[(proj, v)] = shot
    return idx


def _normalize_ts_variants(ts: str) -> List[str]:
    """
    Produce multiple comparable timestamp keys from a timestamp string.
    - Original
    - ISO without millis
    - ISO without tz and without millis
    - Compact YYYYMMDD_HHMMSS (UTC if tz present, else naive)
    """
    if not ts:
        return []
    s = str(ts).strip()
    out: List[str] = []
    out.append(s)

    # Normalize Z to +00:00 so fromisoformat can parse
    s_for_iso = s.replace("Z", "+00:00")
    # Strip millis: remove .ssssss
    s_no_ms = re.sub(r"\.\d+", "", s_for_iso)
    out.append(s_no_ms)

    # Strip timezone offset
    s_no_tz = re.sub(r"([+-]\d{2}:\d{2})$", "", s_no_ms)
    out.append(s_no_tz)

    # Compact variant YYYYMMDD_HHMMSS
    compact = ""
    try:
        dt = None
        # try ISO parse
        try:
            dt = datetime.fromisoformat(s_for_iso)
        except Exception:
            dt = None
        if dt is not None:
            # as given (respecting tz if present)
            compact = dt.strftime("%Y%m%d_%H%M%S")
            # and local-time compact if tz-aware
            try:
                if dt.tzinfo is not None:
                    dt_local = dt.astimezone()  # system local tz
                    local_compact = dt_local.strftime("%Y%m%d_%H%M%S")
                    if local_compact and local_compact not in out:
                        out.append(local_compact)
            except Exception:
                pass
        else:
            # maybe already in compact
            m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$", s)
            if m:
                compact = s
        if compact:
            out.append(compact)
    except Exception:
        pass

    # De-dup
    dedup: List[str] = []
    for v in out:
        if v and v not in dedup:
            dedup.append(v)
    return dedup


def _resolve_screenshot_path(
    project: str,
    timestamp: str,
    log_row_path: str,
    pipeline_idx: Dict[Tuple[str, str], str],
    repo_root: Optional[Path],
) -> str:
    # 1) Prefer pipeline_run.csv mapping with normalized variants
    path = ""
    for v in _normalize_ts_variants(timestamp):
        if (project, v) in pipeline_idx:
            path = pipeline_idx[(project, v)]
            break
    # 2) Fallback to the log.csv row's path if needed
    if not path:
        path = log_row_path or ""
    # 3) If still not absolute/exists try to resolve relative to repo (dev/survey/screenshots/<basename>)
    if path and not os.path.exists(path) and repo_root is not None:
        candidate = repo_root / "dev" / "survey" / "screenshots" / os.path.basename(path)
        if candidate.exists():
            return str(candidate)
    # 4) Derive from naming convention: YYYYMMDD_HHMMSS_Project_With_Underscores.png
    if (not path) and repo_root is not None:
        inferred = _derive_screenshot_convention_path(project, timestamp, repo_root)
        if inferred:
            return inferred
    return path


def _derive_screenshot_convention_path(project: str, timestamp: str, repo_root: Path) -> str:
    """
    Attempt to construct a screenshot path using the known convention:
    dev/survey/screenshots/{YYYYMMDD_HHMMSS}_{Project_With_Underscores}.png
    """
    compacts = _normalize_ts_variants(timestamp)
    compact_keys = [c for c in compacts if re.match(r"^\d{8}_\d{6}$", c)]
    if not compact_keys:
        return ""
    underscored = re.sub(r"\s+", "_", project.strip())
    shots_dir = repo_root / "dev" / "survey" / "screenshots"
    for c in compact_keys:
        candidate = shots_dir / f"{c}_{underscored}.png"
        if candidate.exists():
            return str(candidate)
        # try jpg/jpeg as well
        candidate_jpg = shots_dir / f"{c}_{underscored}.jpg"
        if candidate_jpg.exists():
            return str(candidate_jpg)
        candidate_jpeg = shots_dir / f"{c}_{underscored}.jpeg"
        if candidate_jpeg.exists():
            return str(candidate_jpeg)
    # Last resort: scan for nearest by date prefix and project suffix
    try:
        all_png = list((shots_dir).glob(f"*_{underscored}.png"))
        all_jpg = list((shots_dir).glob(f"*_{underscored}.jpg"))
        all_jpeg = list((shots_dir).glob(f"*_{underscored}.jpeg"))
        candidates = all_png + all_jpg + all_jpeg
        if not candidates:
            return ""
        # If we have a compact key, prefer same-date matches
        if compact_keys:
            want_date = compact_keys[0][:8]  # YYYYMMDD
            for p in candidates:
                name = p.name
                m = re.match(r"^(\d{8})_\d{6}_", name)
                if m and m.group(1) == want_date:
                    return str(p)
        # Otherwise return the lexicographically closest (sorted)
        candidates.sort()
        return str(candidates[0])
    except Exception:
        return ""


def _default_output_dir() -> str:
    # Prefer the workspace-relative path if available
    # dev/survey/data_collection/12_3_experiments
    return str((Path(__file__).resolve().parents[4] / "dev" / "survey" / "data_collection" / "12_11_experiments").resolve())


def run_cli(
    log_csv: str,
    pipeline_run_csv: str,
    output_dir: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    projects_filter: Optional[List[str]] = None,
    limit_per_project: Optional[int] = None,
) -> None:
    # Load .env so OPENAI_API_KEY is available
    try:
        load_dotenv()
    except Exception:
        pass

    model_id = model or "openai/gpt-5-mini"
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DSPY_API_KEY")
    lm = dspy.LM(model_id, api_key=key, temperature=1.0, max_tokens=24000)
    if temperature is not None:
        try:
            if hasattr(lm, "kwargs"):
                lm.kwargs["temperature"] = temperature
        except Exception:
            pass
    dspy.configure(lm=lm)

    # Ensure output directory exists
    out_dir_path = Path(output_dir).expanduser().resolve()
    os.makedirs(out_dir_path, exist_ok=True)
    out_csv = out_dir_path / "project_descriptions.csv"

    # Read inputs
    pipeline_rows = _read_csv(pipeline_run_csv)
    log_rows = _read_csv(log_csv)
    screenshot_idx = _build_screenshot_index(pipeline_rows)

    # Group by project
    by_project: Dict[str, List[Dict[str, str]]] = {}
    for r in log_rows:
        proj = str(r.get("project", "")).strip()
        if projects_filter and proj not in projects_filter:
            continue
        by_project.setdefault(proj, []).append(r)

    # Sort each project's rows by timestamp ascending (string compare OK for ISO format)
    for proj, rows in by_project.items():
        rows.sort(key=lambda rr: str(rr.get("timestamp", "")))

    # Process sequentially per project
    describer = ProjectDescriberModule()
    all_outputs: List[Dict[str, str]] = []
    repo_root = Path(__file__).resolve().parents[4]

    for proj, rows in by_project.items():
        if not proj:
            continue
        if limit_per_project is not None:
            rows = rows[: max(0, int(limit_per_project))]

        current_description = ""
        print(f"\n=== Project: {proj} ===")
        for idx, r in enumerate(rows, start=1):
            ts = str(r.get("timestamp", ""))
            context_update = str(r.get("context_update", "") or "")
            scratchpad_text = str(r.get("scratchpad_text", "") or "")
            log_shot = str(r.get("screenshot_path", "") or "")
            shot_path = _resolve_screenshot_path(proj, ts, log_shot, screenshot_idx, repo_root)
            has_image = bool(shot_path and os.path.exists(shot_path))
            if has_image:
                img = _load_dspy_image(shot_path)
                updated = describer.describe_with_image(
                    project_name=proj,
                    context_update=context_update,
                    screenshot=img,
                    project_scratchpad=scratchpad_text,
                    project_description=current_description,
                )
            else:
                print(f"[warn] missing screenshot → using text-only | project={proj} ts={ts} path={shot_path}")
                updated = describer.describe_text_only(
                    project_name=proj,
                    context_update=context_update,
                    project_scratchpad=scratchpad_text,
                    project_description=current_description,
                )
            current_description = updated or ""

            all_outputs.append(
                {
                    "project": proj,
                    "timestamp": ts,
                    "step_index": str(idx),
                    "screenshot_path": shot_path,
                    "updated_project_description": current_description,
                }
            )

            # Print intermediate result in order
            print(f"\n[{idx}] {ts}")
            print(current_description if current_description else "N/A")

        # Final summary line per project
        if rows:
            print(f"\n--- Final description for {proj} ---")
            print(current_description if current_description else "N/A")

    # Write combined CSV
    if all_outputs:
        fieldnames = ["project", "timestamp", "step_index", "screenshot_path", "updated_project_description"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_outputs:
                writer.writerow(row)
        print(f"\nSaved results to: {out_csv}")
    else:
        print("No outputs generated (no matching rows).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ProjectDescriber over a context log, joining screenshots from pipeline_run.csv")
    ap.add_argument("--log-csv", required=True, help="Path to the primary log.csv (e.g., dev/survey/data_collection/11_20_experiments/no_user/log.csv)")
    ap.add_argument("--pipeline-run-csv", required=True, help="Path to pipeline_run.csv with correct screenshot paths")
    ap.add_argument("--output-dir", default=_default_output_dir(), help="Output directory (default: dev/survey/data_collection/12_11_experiments)")
    ap.add_argument("--projects", default="", help="Comma-separated list of projects to include (default: all)")
    ap.add_argument("--model", default="openai/gpt-5-mini", help="DSPy model id (default: openai/gpt-5-mini)")
    ap.add_argument("--api-key", default="", help="API key for the LM (default: env OPENAI_API_KEY/DSPY_API_KEY)")
    ap.add_argument("--temperature", type=float, default=None, help="Optional temperature override")
    ap.add_argument("--limit-per-project", type=int, default=None, help="Process at most N rows per project (for quick tests)")
    args = ap.parse_args()

    projects_filter = [p.strip() for p in args.projects.split(",") if p.strip()] if args.projects else None
    run_cli(
        log_csv=args.log_csv,
        pipeline_run_csv=args.pipeline_run_csv,
        output_dir=args.output_dir,
        model=args.model,
        api_key=(args.api_key or None),
        temperature=args.temperature,
        projects_filter=projects_filter,
        limit_per_project=args.limit_per_project,
    )


if __name__ == "__main__":
    main()
