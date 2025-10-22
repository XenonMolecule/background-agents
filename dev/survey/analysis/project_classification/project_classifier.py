import argparse
import csv
import json
import os
from typing import Literal, Optional, List

from PIL import Image as PILImage

import dspy
from dotenv import load_dotenv
import shutil
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


class ProjectClassifier(dspy.Signature):
    """
    Classify the user's current work into one of the known projects using recent objectives, propositions/observations, calendar context, and a screenshot path.
    """

    recent_objectives: str = dspy.InputField(description="Recent objectives or goals that the user has been working on")
    recent_propositions: str = dspy.InputField(description="Recent propositions that we have inferred about the user (may be inaccurate)")
    calendar_events: str = dspy.InputField(description="Upcoming calendar events that the user has scheduled")
    screenshot: dspy.Image = dspy.InputField(description="The user's current screen")
    true_projects: list[str] = dspy.InputField(description="Set of ground truth projects that the user has provided that they are working on at any point in time")
    project: Literal[tuple(TRUE_PROJECTS)] = dspy.OutputField(description="Predicted project label that the user is currently working on")

class ProjectClassifierWithHistory(dspy.Signature):
    """
    Classify the user's current work into one of the known projects using recent objectives, propositions/observations, calendar context, and a screenshot path.
    """

    recent_objectives: str = dspy.InputField(description="Recent objectives or goals that the user has been working on")
    recent_propositions: str = dspy.InputField(description="Recent propositions that we have inferred about the user (may be inaccurate)")
    calendar_events: str = dspy.InputField(description="Upcoming calendar events that the user has scheduled")
    screenshot: dspy.Image = dspy.InputField(description="The user's current screen")
    recent_project_predictions: list[str] = dspy.InputField(description="Most recent project predictions for the user (may be inaccurate)")
    true_projects: list[str] = dspy.InputField(description="Set of ground truth projects that the user has provided that they are working on at any point in time")
    project: Literal[tuple(TRUE_PROJECTS)] = dspy.OutputField(description="Predicted project label that the user is currently working on")


class ClassifyProject(dspy.Module):
    def __init__(self, with_history: bool = False):
        self.with_history = with_history
        sig = ProjectClassifierWithHistory if with_history else ProjectClassifier
        self.project_classifier = dspy.ChainOfThought(sig)

    def forward(
        self,
        recent_objectives: str,
        recent_propositions: str,
        calendar_events: str,
        screenshot: dspy.Image,
        true_projects: list[str],
        recent_project_predictions: Optional[List[str]] = None,
    ):
        """Call the underlying classifier; return the output object with `.project`.

        When history is enabled, pass `recent_project_predictions` through to the signature.
        """
        kwargs = {
            "recent_objectives": recent_objectives,
            "recent_propositions": recent_propositions,
            "calendar_events": calendar_events,
            "screenshot": screenshot,
            "true_projects": true_projects,
        }
        if self.with_history:
            kwargs["recent_project_predictions"] = recent_project_predictions or []

        res = self.project_classifier(**kwargs)
        return res


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


def _extract_top_goal(row: dict, top_k: int = 1) -> List[str]:
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


def _extract_top_props(row: dict, top_k: int = 3) -> List[str]:
    raw = row.get("user_details", "")
    try:
        items = json.loads(raw) if raw else []
    except Exception:
        items = []
    results: List[str] = []
    if isinstance(items, list) and items:
        for p in items[: max(0, top_k)]:
            if isinstance(p, dict) and "text" in p:
                results.append(str(p["text"]))
            else:
                results.append(str(p))
        return results
    # fallback: try recent_observations if it is JSON list
    recent_obs = row.get("recent_observations", "")
    try:
        ro = json.loads(recent_obs)
        if isinstance(ro, list):
            return [str(x) for x in ro[: max(0, top_k)]]
    except Exception:
        pass
    s = _safe_json_parse(raw or recent_obs)
    return [s] if s else []


def _extract_calendar(row: dict) -> str:
    return _safe_json_parse(row.get("calendar_events", ""))


def _extract_screenshot(row: dict) -> str:
    return str(row.get("screenshot_path", ""))


def _load_dspy_image(path: str) -> dspy.Image:
    """Load an image path into a dspy.Image via PIL; fall back to a tiny white image."""
    try:
        if path and os.path.exists(path):
            pil = PILImage.open(path)
            pil = pil.convert("RGB")
            pil.load()
            # Lightly downscale to keep tokens reasonable
            max_side = 1600
            pil.thumbnail((max_side, max_side))
            return dspy.Image.from_PIL(pil)
    except Exception:
        pass
    # Fallback: 1x1 white pixel
    return dspy.Image.from_PIL(PILImage.new("RGB", (1, 1), "white"))


def _parse_epoch(ts: str) -> int:
    if not ts:
        return 0
    s = str(ts)
    # Try ISO-8601 / RFC strings
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    # Try 20251016_174439
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


def run_batch(
    input_csv: str,
    output_csv: str,
    *,
    limit: Optional[int] = None,
    temperature: Optional[float] = None,
    goals_window: int = 10,
    top_goals_per_row: int = 1,
    props_window: int = 5,
    top_props_per_row: int = 3,
    num_threads: Optional[int] = None,
    max_errors: Optional[int] = None,
    disable_progress_bar: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_history: bool = False,
    history_window_minutes: int = 10,
    history_top_k: int = 3,
) -> None:
    # Load .env (so OPENAI_API_KEY from project root is available)
    try:
        load_dotenv()
    except Exception:
        pass

    # Configure LM with OPENAI_API_KEY (or provided api_key)
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

    module = ClassifyProject(with_history=use_history)

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except Exception:
        pass

    with open(input_csv, "r", newline="", encoding="utf-8") as f_in, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        # Sort rows by parsed epoch ascending for stable windows
        rows.sort(key=lambda r: _parse_epoch(str(r.get("timestamp", ""))))
        writer = csv.DictWriter(f_out, fieldnames=["timestamp", "predicted_project", "model"])
        writer.writeheader()

        # Accumulators
        examples: List[dspy.Example] = []
        index_to_ts: List[str] = []
        # For history mode: keep prior predictions (epoch, label)
        prior_preds: List[tuple[int, str]] = []

        for i, row in enumerate(rows):
            ts = str(row.get("timestamp", ""))
            if not ts:
                continue
            # Recent goals: top1 from last N, newest→oldest, annotated with [offset]
            g_start = max(0, i - (goals_window - 1))
            g_rows = rows[g_start : i + 1][::-1]
            goal_lines: List[str] = []
            for offset, r in enumerate(g_rows):
                tops = _extract_top_goal(r, top_k=top_goals_per_row)
                if not tops:
                    continue
                goal_lines.append(f"[{offset}] {tops[0]}")
            # Prefix to make ordering visually explicit
            if goal_lines:
                recent_objectives = "Recent goals (Newest→Oldest):\n" + "\n".join(goal_lines)
            else:
                recent_objectives = "Recent goals (Newest→Oldest): none"

            # Recent propositions: topK from each of last M, newest→oldest, dedup order-preserving
            p_start = max(0, i - (props_window - 1))
            p_rows = rows[p_start : i + 1][::-1]
            seen = set()
            prop_lines: List[str] = []
            for r in p_rows:
                for txt in _extract_top_props(r, top_k=top_props_per_row):
                    t = txt.strip()
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    prop_lines.append(t)
            recent_propositions = "\n".join(prop_lines)

            calendar_events = _extract_calendar(row)
            screenshot_img = _load_dspy_image(_extract_screenshot(row))

            if use_history:
                # Gather recent predictions within window
                now_epoch = _parse_epoch(ts)
                window_ms = history_window_minutes * 60 * 1000
                recent_hist = [label for (e, label) in prior_preds if now_epoch - e <= window_ms]
                # Most recent first; limit to top-k
                recent_hist = recent_hist[::-1][: max(0, history_top_k)] if recent_hist else []
                ex = dspy.Example(
                    recent_objectives=recent_objectives,
                    recent_propositions=recent_propositions,
                    calendar_events=calendar_events,
                    screenshot=screenshot_img,
                    recent_project_predictions=recent_hist,
                    true_projects=TRUE_PROJECTS,
                ).with_inputs(
                    "recent_objectives", "recent_propositions", "calendar_events", "screenshot", "recent_project_predictions", "true_projects"
                )
            else:
                ex = dspy.Example(
                    recent_objectives=recent_objectives,
                    recent_propositions=recent_propositions,
                    calendar_events=calendar_events,
                    screenshot=screenshot_img,
                    true_projects=TRUE_PROJECTS,
                ).with_inputs(
                    "recent_objectives", "recent_propositions", "calendar_events", "screenshot", "true_projects"
                )
            examples.append(ex)
            index_to_ts.append(ts)
            if limit and len(examples) >= limit:
                break

        # Run inference
        predictions: List[str] = []
        if use_history:
            # Sequential to allow feeding predictions as history
            n = len(examples)
            for idx in tqdm(range(n), disable=disable_progress_bar, desc="History inference"):
                ex = examples[idx]
                ts = index_to_ts[idx]
                try:
                    res = module(
                        recent_objectives=ex.recent_objectives,
                        recent_propositions=ex.recent_propositions,
                        calendar_events=ex.calendar_events,
                        screenshot=ex.screenshot,
                        true_projects=getattr(ex, "true_projects", TRUE_PROJECTS),
                        recent_project_predictions=getattr(ex, "recent_project_predictions", []),
                    )
                    pred = getattr(res, "project", None) or "Misc"
                except Exception:
                    pred = "Misc"
                predictions.append(pred)
                # Record for future rows
                prior_preds.append((_parse_epoch(ts), pred))
        else:
            try:
                outputs = module.project_classifier.batch(
                    examples,
                    num_threads=num_threads,
                    max_errors=max_errors,
                    disable_progress_bar=disable_progress_bar,
                )
                for out in outputs:
                    predictions.append(getattr(out, "project", None) or "Misc")
            except Exception:
                for ex in tqdm(examples, disable=disable_progress_bar, desc="Sequential inference"):
                    try:
                        res = module(
                            recent_objectives=ex.recent_objectives,
                            recent_propositions=ex.recent_propositions,
                            calendar_events=ex.calendar_events,
                            screenshot=ex.screenshot,
                            true_projects=ex.true_projects,
                        )
                        predictions.append(getattr(res, "project", None) or "Misc")
                    except Exception:
                        predictions.append("Misc")

        # Resolve model name robustly and avoid 'unknown'
        lm_obj = getattr(dspy.settings, "lm", None)
        resolved_model = model or os.environ.get("DSPY_MODEL")
        if not resolved_model:
            resolved_model = getattr(lm_obj, "name", None) or (lm_obj if isinstance(lm_obj, str) else None)
        model_name = resolved_model or "unspecified-model"
        for ts, pred in zip(index_to_ts, predictions):
            writer.writerow({"timestamp": ts, "predicted_project": pred, "model": model_name})

    # Also write/update a stable alias for the viewer to consume by default
    try:
        out_dir = os.path.dirname(output_csv)
        latest_path = os.path.join(out_dir, "context_project_predictions_latest.csv")
        shutil.copyfile(output_csv, latest_path)
    except Exception:
        pass


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run project classifier over context_log.csv")
    parser.add_argument(
        "--input",
        default=os.path.join("dev", "survey", "context_log.csv"),
        help="Path to input context_log.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output predictions CSV (join on timestamp). If omitted, a default file including the model name will be used.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for test runs")
    parser.add_argument(
        "--model",
        default=os.environ.get("DSPY_MODEL"),
        help="LM name for dspy.configure (e.g., openai/gpt-4o, openai/gpt-5)",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key for the LM (env .env OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--goals-window", type=int, default=10, help="Window size (logs) for goals aggregation")
    parser.add_argument("--top-goals-per-row", type=int, default=1, help="Top-k goals per row to include")
    parser.add_argument("--props-window", type=int, default=5, help="Window size (logs) for propositions aggregation")
    parser.add_argument("--top-props-per-row", type=int, default=3, help="Top-k propositions per row to include")
    parser.add_argument("--use-history", action="store_true", help="Include recent predictions as input context")
    parser.add_argument("--history-window-minutes", type=int, default=10, help="Lookback window for prior predictions (minutes)")
    parser.add_argument("--history-top-k", type=int, default=3, help="Max prior predictions to include (most recent first)")
    args = parser.parse_args()

    # Derive default output path with model name if not provided
    if not args.output:
        model_name = args.model or os.environ.get("DSPY_MODEL", "model")
        safe = model_name.replace("/", "_").replace(":", "_")
        suffix = "with_history" if args.use_history else None
        base = f"context_project_predictions_{safe}{('_' + suffix) if suffix else ''}.csv"
        args.output = os.path.join("dev", "survey", "analysis", "project_classification", "results", base)

    run_batch(
        input_csv=args.input,
        output_csv=args.output,
        limit=args.limit,
        temperature=args.temperature,
        goals_window=args.goals_window,
        top_goals_per_row=args.top_goals_per_row,
        props_window=args.props_window,
        top_props_per_row=args.top_props_per_row,
        model=args.model,
        api_key=args.api_key,
        use_history=args.use_history,
        history_window_minutes=args.history_window_minutes,
        history_top_k=args.history_top_k,
    )


if __name__ == "__main__":
    cli()