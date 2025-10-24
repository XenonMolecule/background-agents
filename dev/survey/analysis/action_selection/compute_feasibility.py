import argparse
import csv
import json
import os
from typing import List, Dict, Tuple

import dspy
from dotenv import load_dotenv
from tqdm import tqdm

from action_selection import FeasibilityEstimation


def _parse_epoch(ts: str) -> int:
  if not ts:
    return 0
  s = str(ts)
  # ISO-like
  try:
    from datetime import datetime

    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)
  except Exception:
    pass
  # 20251016_174439
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


def _parse_actions_with_confidence(scratchpad: str) -> List[Tuple[str, int, str]]:
  """Return list of tuples (action_text, confidence_int, source) where source in {suggestion,next_step}."""
  actions: List[Tuple[str, int, str]] = []
  text = scratchpad or ""
  for header, source in (("## Suggestions", "suggestion"), ("## Next Steps", "next_step")):
    if header not in text:
      continue
    after = text.split(header, 1)[1]
    block = after.split("\n## ", 1)[0].strip()
    if not block:
      continue
    for ln in (ln.strip() for ln in block.split("\n") if ln.strip()):
      # Expected: "[idx] text (confidence: N)"
      txt = ln.split("] ", 1)[1] if "] " in ln else ln
      conf_val = 0
      if "(confidence:" in txt:
        parts = txt.rsplit("(confidence:", 1)
        conf_str = parts[1].split(')', 1)[0].strip()
        conf_val = int("".join(ch for ch in conf_str if ch.isdigit())) if conf_str else 0
        txt = parts[0].strip()
      act = txt.strip()
      if not act or act.lower() == "none":
        continue
      actions.append((act, conf_val, source))
  return actions


def write_feasibility_csv(
  transitions_csv: str,
  scratchpad_csv: str,
  output_csv: str,
  *,
  model: str | None = None,
  api_key: str | None = None,
  temperature: float | None = None,
  batch_size: int = 10,
  limit: int | None = None,
  threads: int | None = None,
  max_errors: int | None = None,
  disable_progress_bar: bool = False,
  examples_per_batch: int | None = None,
) -> None:
  # Load env and configure DSPy
  load_dotenv()
  model_name = model or os.environ.get("DSPY_MODEL") or "openai/gpt-5"
  key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DSPY_API_KEY")
  dspy.configure(lm=dspy.LM(model_name, api_key=key, temperature=1.0, max_tokens=16000))
  if temperature is not None:
    lm = dspy.settings.lm
    if hasattr(lm, "kwargs"):
      lm.kwargs["temperature"] = temperature

  estimator = FeasibilityEstimation(batch_size=batch_size)

  # Load transitions and scratchpads
  with open(transitions_csv, "r", newline="", encoding="utf-8") as f:
    tr_rows = list(csv.DictReader(f))
  with open(scratchpad_csv, "r", newline="", encoding="utf-8") as f:
    sp_rows = list(csv.DictReader(f))

  # Build map timestamp -> scratchpad string
  ts_to_scratch: Dict[str, Dict[str, str]] = {}
  for r in sp_rows:
    ts = str(r.get("timestamp", ""))
    if not ts:
      continue
    ts_to_scratch[ts] = {
      "project": r.get("project", ""),
      "scratchpad": r.get("scratchpad", ""),
      "summary": r.get("summary", ""),
    }

  # Ensure output dir exists
  try:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  except Exception:
    pass

  fieldnames = [
    "timestamp",
    "is_transition",
    "from_project",
    "to_project",
    "smoothed_project",
    "feasibility_list",
  ]
  # Sort transitions by epoch for stability
  tr_rows.sort(key=lambda r: _parse_epoch(str(r.get("timestamp", ""))))

  # Determine which timestamps to process: transition rows and the row right before each transition
  ts_interest: List[str] = []
  seen: set[str] = set()
  processed_tr: int = 0
  for i, r in enumerate(tr_rows):
    is_tr = str(r.get("is_transition", "")).strip().upper() == "TRUE"
    if not is_tr:
      continue
    if limit is not None and processed_tr >= limit:
      continue
    cur_ts = str(r.get("timestamp", ""))
    if cur_ts and cur_ts not in seen:
      ts_interest.append(cur_ts); seen.add(cur_ts)
    # also add previous row timestamp if available
    if i > 0:
      prev_ts = str(tr_rows[i - 1].get("timestamp", ""))
      if prev_ts and prev_ts not in seen:
        ts_interest.append(prev_ts); seen.add(prev_ts)
    processed_tr += 1

  # Build a global batch with one example per selected timestamp
  dataset: List[dspy.Example] = []
  dataset_owner: List[str] = []  # map each example to its timestamp
  ts_to_parsed: Dict[str, List[Tuple[str, int, str]]] = {}

  for ts in ts_interest:
    sp = ts_to_scratch.get(ts)
    if not sp or not sp.get("scratchpad"):
      continue
    scratchpad_text = sp["scratchpad"]
    parsed_actions = _parse_actions_with_confidence(scratchpad_text)
    ts_to_parsed[ts] = parsed_actions
    if not parsed_actions:
      continue
    actions_only = [txt for (txt, _conf, _src) in parsed_actions]
    ex = dspy.Example(project_scratchpad=scratchpad_text, next_steps=actions_only).with_inputs("project_scratchpad", "next_steps")
    dataset.append(ex)
    dataset_owner.append(ts)

  # Run batched inference (optionally in smaller chunks to avoid thread hangs)
  outputs = []
  if dataset:
    total = len(dataset)
    chunk_size = max(1, int(examples_per_batch) if examples_per_batch else total)
    bar = tqdm(total=total, desc="Processed examples")
    for i in range(0, total, chunk_size):
      chunk = dataset[i:i + chunk_size]
      chunk_out = estimator.estimator.batch(
        chunk,
        num_threads=threads,
        max_errors=max_errors,
        disable_progress_bar=disable_progress_bar,
      )
      # materialize and extend
      chunk_list = list(chunk_out)
      outputs.extend(chunk_list)
      bar.update(len(chunk))
    bar.close()
    assert len(outputs) == len(dataset), f"Expected {len(dataset)} outputs, got {len(outputs)}"

  # Aggregate feasibility items per timestamp
  ts_to_feas: Dict[str, List[object]] = {}
  for owner_ts, out in zip(dataset_owner, outputs):
    ts_to_feas[owner_ts] = list(out.feasibility or [])

  # Write results
  with open(output_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in tqdm(tr_rows, desc="Writing results"):
      ts = str(r.get("timestamp", ""))
      is_tr = str(r.get("is_transition", "")).strip().upper() == "TRUE"
      out_row: Dict[str, str] = {
        "timestamp": ts,
        "is_transition": "TRUE" if is_tr else "FALSE",
        "from_project": r.get("from_project", ""),
        "to_project": r.get("to_project", ""),
        "smoothed_project": r.get("smoothed_project", ""),
        "feasibility_list": "",
      }

      # We write feasibility if this ts is in our interest set (transition or pre-transition)
      if ts not in seen:
        w.writerow(out_row)
        continue

      parsed_actions = ts_to_parsed.get(ts, [])
      feas_list = ts_to_feas.get(ts, [])

      def norm(s: str) -> str:
        return (s or "").strip().lower()

      action_meta: Dict[str, Dict[str, object]] = {}
      for item in feas_list:
        if isinstance(item, dict):
          action_txt = item["action"]
          feas_val = item["feasibility"]
          missing_ctx = item.get("missing_context")
        else:
          action_txt = item.action
          feas_val = item.feasibility
          missing_ctx = getattr(item, "missing_context", None)
        action_meta[norm(str(action_txt))] = {
          "feasibility": int(feas_val),
          "missing_context": (missing_ctx or "")
        }

      combined: Dict[str, Dict[str, object]] = {}
      for txt, conf, source in parsed_actions:
        key = norm(txt)
        if not key or key == "none":
          continue
        meta = action_meta.get(key, {})
        cur = combined.get(key) or {
          "action": txt,
          "confidence": 0,
          "feasibility": int(meta.get("feasibility", 0) if isinstance(meta, dict) else 0),
          "missing_context": (meta.get("missing_context") if isinstance(meta, dict) else ""),
          "source": source
        }
        cur["confidence"] = max(int(cur.get("confidence", 0)), int(conf))
        cur["feasibility"] = int(meta.get("feasibility", cur.get("feasibility", 0)) if isinstance(meta, dict) else cur.get("feasibility", 0))
        # keep missing_context from meta if available
        if isinstance(meta, dict) and meta.get("missing_context"):
          cur["missing_context"] = meta.get("missing_context")
        combined[key] = cur

      ranked = sorted(
        combined.values(),
        key=lambda o: (-int(o.get("feasibility", 0)), -int(o.get("confidence", 0)), str(o.get("action", "")))
      )

      out_row["feasibility_list"] = json.dumps(ranked, ensure_ascii=False)
      w.writerow(out_row)

  # write/update stable alias
  out_dir = os.path.dirname(output_csv)
  latest_path = os.path.join(out_dir, "feasibility_latest.csv")
  from shutil import copyfile
  copyfile(output_csv, latest_path)


def cli() -> None:
  parser = argparse.ArgumentParser(description="Compute feasibility rankings at transition points")
  parser.add_argument(
    "--transitions",
    default=os.path.join(
      "dev", "survey", "analysis", "transition_classifier", "results", "transitions_latest.csv"
    ),
    help="Path to transitions CSV",
  )
  parser.add_argument(
    "--scratchpads",
    default=os.path.join(
      "dev", "survey", "analysis", "project_scratchpad", "results", "project_scratchpad_latest.csv"
    ),
    help="Path to project_scratchpad CSV",
  )
  parser.add_argument(
    "--output",
    default=os.path.join(
      "dev", "survey", "analysis", "action_selection", "results", "feasibility_openai_gpt-5.csv"
    ),
    help="Path to output feasibility CSV",
  )
  parser.add_argument("--model", default=os.environ.get("DSPY_MODEL", "openai/gpt-5"))
  parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
  parser.add_argument("--temperature", type=float, default=None)
  parser.add_argument("--batch-size", type=int, default=10)
  parser.add_argument("--limit", type=int, default=None, help="Limit number of transition points processed")
  parser.add_argument("--threads", type=int, default=None, help="Number of parallel threads for DSPy batch")
  parser.add_argument("--max-errors", type=int, default=None, help="Max errors tolerated in DSPy batch")
  parser.add_argument("--no-batch-progress", action="store_true", help="Disable DSPy batch progress bar")
  parser.add_argument("--examples-per-batch", type=int, default=None, help="Chunk size for DSPy batch calls (prevents thread hangs)")
  args = parser.parse_args()

  write_feasibility_csv(
    transitions_csv=args.transitions,
    scratchpad_csv=args.scratchpads,
    output_csv=args.output,
    model=args.model,
    api_key=args.api_key,
    temperature=args.temperature,
    batch_size=args.batch_size,
    limit=args.limit,
    threads=args.threads,
    max_errors=args.max_errors,
    disable_progress_bar=bool(args.no_batch_progress),
    examples_per_batch=args.examples_per_batch,
  )


if __name__ == "__main__":
  cli()


# How to run this script:
#
# 1. Prerequisites:
#    - Ensure you have transitions_latest.csv (from transition_classifier)
#    - Ensure you have project_scratchpad_latest.csv (from project_scratchpad)
#    - Set OPENAI_API_KEY environment variable or create .env file
#
# 2. Basic usage:
#    python dev/survey/analysis/action_selection/compute_feasibility.py
#
# 3. Custom paths:
#    python dev/survey/analysis/action_selection/compute_feasibility.py \
#      --transitions path/to/transitions.csv \
#      --scratchpads path/to/scratchpads.csv \
#      --output path/to/output.csv
#
# 4. Model options:
#    python dev/survey/analysis/action_selection/compute_feasibility.py \
#      --model openai/gpt-4o \
#      --temperature 0.7 \
#      --batch-size 5
#
# 5. Output:
#    - Creates feasibility_openai_gpt-5.csv (or custom output path)
#    - Also creates feasibility_latest.csv for slides to consume
#    - Only transition points (is_transition=TRUE) get feasibility lists
#    - Actions are ranked by feasibility (desc) then confidence (desc)
#
# 6. View results:
#    - Open dev/survey/slides/index.html in browser
#    - Transition slides will show "Feasible Next Actions" section
#    - Override feasibility CSV with ?feas=path/to/your.csv


