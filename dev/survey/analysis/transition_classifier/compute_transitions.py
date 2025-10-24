import argparse
import csv
import os
from typing import List, Dict, Tuple


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


def smooth_labels(
    rows: List[Dict[str, str]],
    window_minutes: int = 15,
    label_field: str = "predicted_project",
) -> List[str]:
    epochs = [_parse_epoch(r.get("timestamp", "")) for r in rows]
    labels = [str(r.get(label_field, "")) for r in rows]
    n = len(rows)
    out: List[str] = [""] * n
    window_ms = window_minutes * 60 * 1000

    # Backward-looking window: for each i, consider indices [start, i]
    start = 0
    for i in range(n):
        current_t = epochs[i]
        # advance start so that epochs[i] - epochs[start] <= window
        while start < i and (current_t - epochs[start]) > window_ms:
            start += 1
        # window indices are [start, i]
        counts: Dict[str, int] = {}
        last_seen: Dict[str, Tuple[int, int]] = {}  # label -> (last_epoch, last_index)
        for k in range(start, i + 1):
            lab = labels[k] or ""
            counts[lab] = counts.get(lab, 0) + 1
            last_seen[lab] = (epochs[k], k)
        if not counts:
            out[i] = labels[i] or ""
            continue
        # majority within backward window
        max_count = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == max_count]
        if len(tied) == 1:
            out[i] = tied[0]
        elif len(tied) == 2:
            # pick the label seen later in time within the window
            later = max(tied, key=lambda lab: last_seen.get(lab, (0, -1)))
            out[i] = later
        else:
            # >2-way tie → Misc
            out[i] = "Misc"

    return out


def write_transitions_csv(
    input_csv: str,
    output_csv: str,
    window_minutes: int = 15,
) -> None:
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sort rows by timestamp asc for stable windows and transition detection
    rows.sort(key=lambda r: _parse_epoch(r.get("timestamp", "")))
    smoothed = smooth_labels(rows, window_minutes=window_minutes, label_field="predicted_project")

    # Compute transitions on smoothed labels
    fieldnames = [
        "timestamp",
        "predicted_project",
        "smoothed_project",
        "is_transition",
        "from_project",
        "to_project",
    ]
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except Exception:
        pass
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        prev: str = "IDLE"
        for i, row in enumerate(rows):
            cur = smoothed[i]
            # Determine previous state for transition comparison
            if i == 0:
                prev = "IDLE"
            else:
                # If gap exceeds window, treat previous state as IDLE
                gap_ms = _parse_epoch(rows[i]["timestamp"]) - _parse_epoch(rows[i - 1]["timestamp"])
                prev = "IDLE" if gap_ms > (window_minutes * 60 * 1000) else smoothed[i - 1]

            changed = (cur != prev)
            # Forward-looking idle detection (allowed for offline analysis):
            # If nothing occurs for > window after the current row, treat as transition to IDLE
            cur_epoch = _parse_epoch(rows[i]["timestamp"])
            if i == len(rows) - 1:
                next_gap_ms = float("inf")
            else:
                next_epoch = _parse_epoch(rows[i + 1]["timestamp"])
                next_gap_ms = next_epoch - cur_epoch
            forward_idle = next_gap_ms > (window_minutes * 60 * 1000)

            if forward_idle and not changed and cur:
                is_tr = "TRUE"
                from_p = cur
                to_p = "IDLE"
            else:
                is_tr = "TRUE" if changed else "FALSE"
                from_p = prev if changed else ""
                to_p = cur if changed else ""
            w.writerow({
                "timestamp": row.get("timestamp", ""),
                "predicted_project": row.get("predicted_project", ""),
                "smoothed_project": cur,
                "is_transition": is_tr,
                "from_project": from_p,
                "to_project": to_p,
            })
            prev = cur


def cli() -> None:
    parser = argparse.ArgumentParser(description="Compute smoothed project labels and transition points")
    parser.add_argument(
        "--input",
        default=os.path.join(
            "dev",
            "survey",
            "analysis",
            "project_classification",
            "results",
            "context_project_predictions_latest.csv",
        ),
        help="Path to predictions CSV (timestamp,predicted_project,model)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            "dev",
            "survey",
            "analysis",
            "transition_classifier",
            "results",
            "transitions_latest.csv",
        ),
        help="Path to output transitions CSV",
    )
    parser.add_argument("--window-minutes", type=int, default=15, help="Smoothing window in minutes")
    args = parser.parse_args()

    write_transitions_csv(args.input, args.output, window_minutes=args.window_minutes)


if __name__ == "__main__":
    cli()


