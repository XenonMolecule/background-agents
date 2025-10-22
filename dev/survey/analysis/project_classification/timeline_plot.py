import argparse
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import tz as dateutil_tz
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Interval:
    project: str
    start: pd.Timestamp  # timezone-aware
    end: pd.Timestamp  # timezone-aware, end > start

    def as_tuple_num(self) -> Tuple[float, float]:
        """Return (start_num, duration_days) for broken_barh."""
        start_num = mdates.date2num(self.start.to_pydatetime())
        end_num = mdates.date2num(self.end.to_pydatetime())
        return start_num, max(end_num - start_num, 1e-9)


# -----------------------------
# Parsing and interval building
# -----------------------------

def normalize_label(label: str) -> str:
    """Normalize a project label minimally (trim and collapse internal spaces).

    We preserve original case for display but use this function for equality.
    """
    if not isinstance(label, str):
        return ""
    collapsed = " ".join(label.strip().split())
    return collapsed


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


def canonicalize_project(label: str) -> str:
    """Map various free-form labels into canonical TRUE_PROJECTS buckets.

    Heuristics are intentionally simple and readable.
    """
    raw = normalize_label(label)
    s = raw.lower()
    if not s:
        return ""

    # Direct contains/keyword mappings
    if "autometrics" in s:
        return "AutoMetrics Release"
    if "nlp retreat" in s:
        return "NLP Retreat Planning"
    if "personalization" in s:
        return "Personalization Dataset Collection"
    if "background agents" in s:
        return "Background Agents"
    if "logistics" in s:
        return "Logistics"
    if "health" in s:
        return "Health"
    if "hw3" in s:
        return "HW3 Assignment Creation"
    if "in between" in s or "misc" in s:
        return "Misc"

    # Exact match against canonical set (case-insensitive)
    lowered_set = {p.lower(): p for p in TRUE_PROJECTS}
    if s in lowered_set:
        return lowered_set[s]

    # Fallback to the minimally normalized label to avoid data loss
    return raw


def build_intervals_from_observations(
    df: pd.DataFrame,
    project_col: str,
    horizon_minutes: int = 45,
) -> List[Interval]:
    """Given a dataframe with columns ['timestamp', project_col], build timeline intervals.

    Rules:
    - Each observation at time t implies working on that project for horizon minutes.
    - If a new observation occurs before that horizon elapses, the previous segment ends at the new observation.
    - Gaps (no observations) render as blank; we do not fill beyond horizon.
    """
    if df.empty:
        return []

    df_local = df[["timestamp", project_col]].copy()
    df_local = df_local.dropna(subset=["timestamp", project_col])
    df_local[project_col] = df_local[project_col].map(normalize_label)
    df_local = df_local.sort_values("timestamp").reset_index(drop=True)

    intervals: List[Interval] = []
    horizon = pd.Timedelta(minutes=horizon_minutes)

    for i, row in df_local.iterrows():
        start_time: pd.Timestamp = row["timestamp"]
        project: str = canonicalize_project(row[project_col])
        # Default end is start + horizon
        end_time = start_time + horizon
        # Truncate if there is a next observation earlier than end
        if i + 1 < len(df_local):
            next_time = df_local.loc[i + 1, "timestamp"]
            if next_time < end_time:
                end_time = next_time
        if end_time > start_time:
            intervals.append(Interval(project=project, start=start_time, end=end_time))

    return intervals


def merge_adjacent_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Merge adjacent or overlapping intervals with the same project.

    Intervals must be provided in chronological order.
    """
    intervals_sorted = sorted(intervals, key=lambda x: (x.start, x.end))
    merged: List[Interval] = []
    for seg in intervals_sorted:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg.project == last.project and seg.start <= last.end:
            # Extend last interval
            merged[-1] = Interval(project=last.project, start=last.start, end=max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged


# -----------------------------
# Plotting
# -----------------------------

def generate_color_map(labels: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Stable color map from labels using a qualitative palette."""
    unique = list(dict.fromkeys(labels))  # preserve order of first appearance
    cmap = plt.get_cmap("tab20")
    colors: Dict[str, Tuple[float, float, float, float]] = {}
    for idx, label in enumerate(unique):
        colors[label] = cmap(idx % cmap.N)
    return colors


def _plot_single_timeline(
    ax: plt.Axes,
    intervals: List[Interval],
    title: str,
    color_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    show_legend: bool = False,
) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_yticks([])
    ax.set_ylabel("")

    if not intervals:
        # Draw an empty baseline
        ax.set_ylim(0, 20)
        ax.set_xlim(0, 1)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#6b7280")
        return

    # Build color map if not supplied
    projects = [seg.project for seg in intervals]
    if color_map is None:
        color_map = generate_color_map(projects)

    # Render as a single lane using broken_barh
    ax.set_ylim(0, 20)
    bar_y, bar_height = 5, 10
    for seg in intervals:
        start_num, dur_days = seg.as_tuple_num()
        ax.broken_barh([(start_num, dur_days)], (bar_y, bar_height), facecolors=color_map[seg.project], edgecolors="none")

    # Legend handled at figure-level


def add_day_breaks(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp, tz: str) -> None:
    """Add vertical lines at local midnight boundaries and annotate dates."""
    # Align to midnight in the given timezone
    start_local = start.tz_convert(tz).normalize()
    end_local = end.tz_convert(tz).normalize() + pd.Timedelta(days=1)

    cur = start_local
    while cur <= end_local:
        cur_utc = cur.tz_convert("UTC")
        x = mdates.date2num(cur_utc.to_pydatetime())
        ax.axvline(x=x, color="#e5e7eb", linestyle="--", linewidth=1)
        # Date label slightly above x-axis
        ax.text(x, -0.15, cur.strftime("%a %b %d"), transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=9, color="#6b7280")
        cur = cur + pd.Timedelta(days=1)


def configure_time_axis(ax: plt.Axes, tz: str) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=dateutil_tz.gettz(tz)))
    ax.tick_params(axis="x", labelrotation=30)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#f1f5f9", linestyle="-", linewidth=0.8)


# -----------------------------
# IO helpers
# -----------------------------

def load_csv_observations(
    path: str,
    tz: str,
    project_col: str,
    timestamp_format: Optional[str] = None,
    naive_origin: str = "utc",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")
    if project_col not in df.columns:
        raise ValueError(f"CSV must contain a '{project_col}' column")

    # Parse timestamps. If a format is provided, treat parsed datetimes as naive and localize.
    if timestamp_format:
        ts = pd.to_datetime(df["timestamp"], format=timestamp_format, errors="coerce")
        # Localize naive timestamps
        if naive_origin.lower() == "utc":
            ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
        else:
            ts = ts.dt.tz_localize(tz)
    else:
        # Assume strings contain timezone info or are UTC-like, then convert to tz
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)

    df = df.assign(timestamp=ts)
    df = df.dropna(subset=["timestamp"]).copy()
    return df


# -----------------------------
# Compact plotting (skip long idle gaps)
# -----------------------------

def _compute_blocks(human_intervals: List[Interval], min_gap: pd.Timedelta) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not human_intervals:
        return []
    segs = sorted(human_intervals, key=lambda s: (s.start, s.end))
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    block_start = segs[0].start
    last_end = segs[0].end
    for seg in segs[1:]:
        gap = seg.start - last_end
        if gap >= min_gap:
            blocks.append((block_start, last_end))
            block_start = seg.start
        if seg.end > last_end:
            last_end = seg.end
    blocks.append((block_start, last_end))
    return blocks


def _plot_compact(
    human_intervals: List[Interval],
    ai_intervals: List[Interval],
    ai_hist_intervals: Optional[List[Interval]],
    tz: str,
    min_gap_minutes: int,
    output_path: str,
    figure_size: Tuple[int, int],
    color_map: Dict[str, Tuple[float, float, float, float]],
    ai_title: Optional[str] = None,
    ai_hist_title: Optional[str] = None,
) -> None:
    if not human_intervals:
        # Nothing to plot; create an empty image with message
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(1, 1, figsize=figure_size)
        ax.axis("off")
        ax.text(0.5, 0.5, "No human intervals", transform=ax.transAxes, ha="center", va="center")
        # Ensure output directory exists
        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    min_gap = pd.Timedelta(minutes=min_gap_minutes)
    blocks = _compute_blocks(human_intervals, min_gap)
    if not blocks:
        blocks = [(human_intervals[0].start, human_intervals[-1].end)]

    # Build sequential x positions for each block
    block_gap_days = (pd.Timedelta(minutes=10) / pd.Timedelta(days=1))  # visual gap between blocks
    block_positions: List[Tuple[Tuple[pd.Timestamp, pd.Timestamp], float, float]] = []  # ((start,end), x_start, x_end)
    cur_x = 0.0
    for (b_start, b_end) in blocks:
        duration_days = (b_end - b_start) / pd.Timedelta(days=1)
        x_start = cur_x
        x_end = x_start + float(duration_days)
        block_positions.append(((b_start, b_end), x_start, x_end))
        cur_x = x_end + block_gap_days

    def to_compact_x(ts: pd.Timestamp) -> Optional[float]:
        for (b_start, b_end), x_start, x_end in block_positions:
            if b_start <= ts <= b_end:
                offset_days = (ts - b_start) / pd.Timedelta(days=1)
                return x_start + float(offset_days)
        return None

    def intervals_to_compact(segs: List[Interval]) -> List[Tuple[float, float, str]]:
        compact: List[Tuple[float, float, str]] = []
        for seg in segs:
            x0 = to_compact_x(seg.start)
            x1 = to_compact_x(seg.end)
            if x0 is None or x1 is None:
                # If an interval straddles a removed gap, split it at block boundaries
                for (b_start, b_end), x_start, x_end in block_positions:
                    s = max(seg.start, b_start)
                    e = min(seg.end, b_end)
                    if e > s:
                        xs = to_compact_x(s)
                        xe = to_compact_x(e)
                        if xs is not None and xe is not None and xe > xs:
                            compact.append((xs, xe - xs, seg.project))
                continue
            width = max(x1 - x0, 1e-9)
            compact.append((x0, width, seg.project))
        return compact

    human_compact = intervals_to_compact(human_intervals)
    ai_compact = intervals_to_compact(ai_intervals) if ai_intervals else []

    # Plot
    plt.style.use("seaborn-v0_8")
    num_rows = 2 + (1 if ai_hist_intervals else 0)
    height_ratios = [1] * num_rows
    fig, axes = plt.subplots(num_rows, 1, figsize=figure_size, sharex=True, height_ratios=height_ratios, constrained_layout=True)
    if num_rows == 1:
        axes = [axes]

    def plot_lane(ax: plt.Axes, data: List[Tuple[float, float, str]], title: str, show_legend: bool) -> None:
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_yticks([])
        ax.set_ylim(0, 20)
        bar_y, bar_height = 5, 10
        for x, w, proj in data:
            ax.broken_barh([(x, w)], (bar_y, bar_height), facecolors=color_map.get(proj, "#999999"), edgecolors="none")
        # Legend handled at figure-level

    plot_lane(axes[0], human_compact, "Human-reported timeline", False)
    plot_lane(axes[1], ai_compact, ai_title or "AI-predicted timeline", False)
    if ai_hist_intervals:
        ai_hist_compact = intervals_to_compact(ai_hist_intervals)
        plot_lane(axes[2], ai_hist_compact, ai_hist_title or "AI-predicted timeline (with history)", False)

    # X-axis formatter to show original timestamps
    def fmt_x(x: float, pos: int) -> str:
        for (b_start, _), x_start, x_end in block_positions:
            if x_start <= x <= x_end:
                delta_days = x - x_start
                ts = b_start + pd.to_timedelta(delta_days, unit="D")
                return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M")
        # In visual gap: return empty label
        return ""

    axes[-1].xaxis.set_major_formatter(FuncFormatter(fmt_x))
    axes[-1].tick_params(axis="x", labelrotation=30)
    for ax in axes:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", color="#f1f5f9", linestyle="-", linewidth=0.8)

    # Vertical separators between blocks
    for (_, _), x_start, x_end in block_positions:
        for ax in axes:
            ax.axvline(x_start, color="#e5e7eb", linestyle=":", linewidth=1)
            ax.axvline(x_end, color="#e5e7eb", linestyle=":", linewidth=1)

    # Limits
    total_x_end = block_positions[-1][2]
    for ax in axes:
        ax.set_xlim(0, total_x_end)

    fig.suptitle(f"Project timelines (compact view; {tz})", fontsize=14)

    # Figure-level legend at bottom
    legend_handles = []
    legend_labels = []
    seen = set()
    for proj, color in color_map.items():
        if proj in seen:
            continue
        seen.add(proj)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=8))
        legend_labels.append(proj)
    if legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, max(1, len(legend_labels))),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.14),
        )
        fig.subplots_adjust(bottom=0.32)
    # Ensure output directory exists
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main entry
# -----------------------------

def plot_timelines(
    human_csv: str,
    output_path: str,
    tz: str = "US/Pacific",
    horizon_minutes: int = 45,
    ai_csv: Optional[str] = None,
    ai_horizon_minutes: int = 4,
    ai_project_col: str = "predicted_project",
    ai_timestamp_format: Optional[str] = "%Y%m%d_%H%M%S",
    ai_naive_origin: str = "local",
    compact_output_path: Optional[str] = None,
    compact_min_gap_minutes: int = 60,
    ai_history_csv: Optional[str] = None,
    ai_history_project_col: str = "predicted_project",
    figure_size: Tuple[int, int] = (14, 5),
) -> None:
    # Load human observations
    human_df = load_csv_observations(human_csv, tz=tz, project_col="project_now")
    human_intervals = build_intervals_from_observations(human_df, project_col="project_now", horizon_minutes=horizon_minutes)
    human_intervals = merge_adjacent_intervals(human_intervals)

    # Optional AI observations (baseline)
    ai_intervals: List[Interval] = []
    ai_loaded = False
    ai_df_for_accuracy: Optional[pd.DataFrame] = None
    if ai_csv:
        try:
            ai_df = load_csv_observations(
                ai_csv,
                tz=tz,
                project_col=ai_project_col,
                timestamp_format=ai_timestamp_format,
                naive_origin=ai_naive_origin,
            )
            ai_df_for_accuracy = ai_df.copy()
            ai_intervals = build_intervals_from_observations(ai_df, project_col=ai_project_col, horizon_minutes=ai_horizon_minutes)
            ai_intervals = merge_adjacent_intervals(ai_intervals)
            ai_loaded = True
        except Exception:
            ai_loaded = False

    # Optional AI observations (with history)
    ai_hist_intervals: List[Interval] = []
    ai_hist_loaded = False
    ai_hist_df_for_accuracy: Optional[pd.DataFrame] = None
    if ai_history_csv:
        try:
            ai_hist_df = load_csv_observations(
                ai_history_csv,
                tz=tz,
                project_col=ai_history_project_col,
                timestamp_format=ai_timestamp_format,
                naive_origin=ai_naive_origin,
            )
            ai_hist_df_for_accuracy = ai_hist_df.copy()
            ai_hist_intervals = build_intervals_from_observations(ai_hist_df, project_col=ai_history_project_col, horizon_minutes=ai_horizon_minutes)
            ai_hist_intervals = merge_adjacent_intervals(ai_hist_intervals)
            ai_hist_loaded = True
        except Exception:
            ai_hist_loaded = False

    # Determine overall time bounds from available intervals
    def bounds(itvs: List[Interval]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        if not itvs:
            return None
        return min(i.start for i in itvs), max(i.end for i in itvs)

    b1 = bounds(human_intervals)
    b2 = bounds(ai_intervals)
    b3 = bounds(ai_hist_intervals)
    if b1 and (b2 or b3):
        candidates = [b for b in [b1, b2, b3] if b]
        start = min(b[0] for b in candidates)
        end = max(b[1] for b in candidates)
    elif b1:
        start, end = b1
    elif b2:
        start, end = b2
    elif b3:
        start, end = b3
    else:
        # Fallback to now-range if nothing loaded
        now = pd.Timestamp.now(tz)
        start, end = now - pd.Timedelta(hours=1), now

    plt.style.use("seaborn-v0_8")
    num_rows = 1 + (1 if ai_loaded else 0) + (1 if ai_hist_loaded else 0)
    height_ratios = [1] * num_rows
    fig, axes = plt.subplots(num_rows, 1, figsize=figure_size, sharex=True, height_ratios=height_ratios, constrained_layout=True)
    if num_rows == 1:
        axes = [axes]

    # Use a shared color map derived from all timelines for consistency
    all_labels = [seg.project for seg in human_intervals]
    if ai_loaded:
        all_labels.extend(seg.project for seg in ai_intervals)
    if ai_hist_loaded:
        all_labels.extend(seg.project for seg in ai_hist_intervals)
    shared_colors = generate_color_map(all_labels)

    # Accuracy calculations (exact match within human-labeled intervals)
    def _label_at(ts: pd.Timestamp, intervals: List[Interval]) -> Optional[str]:
        for seg in intervals:
            if seg.start <= ts <= seg.end:
                return seg.project
        return None

    def _calc_acc(ai_df: Optional[pd.DataFrame], col: str) -> Optional[Tuple[int, int, float]]:
        if ai_df is None or ai_df.empty:
            return None
        total = 0
        correct = 0
        for _, r in ai_df.iterrows():
            ts = r["timestamp"]
            if not isinstance(ts, pd.Timestamp):
                continue
            gt = _label_at(ts, human_intervals)
            if gt is None:
                continue
            total += 1
            pred = canonicalize_project(str(r.get(col, "")))
            if pred == gt:
                correct += 1
        if total == 0:
            return (0, 0, 0.0)
        return (correct, total, 100.0 * correct / total)

    base_acc = _calc_acc(ai_df_for_accuracy, ai_project_col)
    hist_acc = _calc_acc(ai_hist_df_for_accuracy, ai_history_project_col)

    _plot_single_timeline(
        ax=axes[0],
        intervals=human_intervals,
        title="Human-reported timeline",
        color_map=shared_colors,
        show_legend=False,
    )

    row_idx = 1
    if ai_loaded:
        ai_title = "AI-predicted timeline"
        if base_acc is not None and base_acc[1] > 0:
            ai_title += f" — acc {base_acc[2]:.1f}% (n={base_acc[1]})"
        _plot_single_timeline(
            ax=axes[row_idx],
            intervals=ai_intervals,
            title=ai_title,
            color_map=shared_colors,
            show_legend=False,
        )
        row_idx += 1
    if ai_hist_loaded:
        ai_hist_title = "AI-predicted timeline (with history)"
        if hist_acc is not None and hist_acc[1] > 0:
            ai_hist_title += f" — acc {hist_acc[2]:.1f}% (n={hist_acc[1]})"
        _plot_single_timeline(
            ax=axes[row_idx],
            intervals=ai_hist_intervals,
            title=ai_hist_title,
            color_map=shared_colors,
            show_legend=False,
        )
        row_idx += 1
    if not ai_loaded and not ai_hist_loaded and num_rows > 1:
        axes[1].set_title("AI-predicted timeline (placeholder)", fontsize=12, pad=10)
        axes[1].set_yticks([])
        axes[1].text(0.5, 0.5, "Add --ai-csv to visualize predictions", transform=axes[1].transAxes, ha="center", va="center", fontsize=10, color="#6b7280")

    # Configure time axis and day breaks
    for ax in axes:
        configure_time_axis(ax, tz=tz)
        ax.set_xlim(mdates.date2num(start.to_pydatetime()), mdates.date2num(end.to_pydatetime()))
        add_day_breaks(ax, start=start.tz_convert("UTC"), end=end.tz_convert("UTC"), tz=tz)

    # Build title
    fig.suptitle(f"Project timelines ({tz})", fontsize=14)

    # Figure-level legend at the bottom (includes both human and AI labels)
    legend_handles = []
    legend_labels = []
    seen = set()
    for proj, color in shared_colors.items():
        if proj in seen:
            continue
        seen.add(proj)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=8))
        legend_labels.append(proj)
    if legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, max(1, len(legend_labels))),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.14),
        )
        # Leave more room for legend
        fig.subplots_adjust(bottom=0.32)

    # Ensure output directory exists
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optionally render compact plot that skips long idle gaps
    if compact_output_path:
        # Build per-row titles for compact plot as well
        compact_ai_title = None
        compact_ai_hist_title = None
        if base_acc is not None and base_acc[1] > 0:
            compact_ai_title = f"AI-predicted timeline — acc {base_acc[2]:.1f}% (n={base_acc[1]})"
        if hist_acc is not None and hist_acc[1] > 0:
            compact_ai_hist_title = f"AI-predicted timeline (with history) — acc {hist_acc[2]:.1f}% (n={hist_acc[1]})"

        _plot_compact(
            human_intervals=human_intervals,
            ai_intervals=ai_intervals if ai_loaded else [],
            ai_hist_intervals=ai_hist_intervals if ai_hist_loaded else None,
            tz=tz,
            min_gap_minutes=compact_min_gap_minutes,
            output_path=compact_output_path,
            figure_size=figure_size,
            color_map=shared_colors,
            ai_title=compact_ai_title,
            ai_hist_title=compact_ai_hist_title,
        )


def _default_human_csv() -> str:
    # Default to repo survey_responses.csv if present
    return "/Users/michaelryan/Documents/School/Stanford/Research/background-agents/dev/survey/survey_responses.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot human and AI project timelines from CSV logs.")
    parser.add_argument("--input", dest="human_csv", default=_default_human_csv(), help="Path to human survey CSV (default: repo survey_responses.csv)")
    default_out = "/Users/michaelryan/Documents/School/Stanford/Research/background-agents/dev/survey/analysis/project_classification/results/timeline.png"
    parser.add_argument("--output", dest="output_path", default=default_out, help="Path to save the output image (PNG)")
    parser.add_argument("--tz", dest="tz", default="US/Pacific", help="Timezone name for display (e.g., US/Pacific)")
    parser.add_argument("--horizon", dest="horizon", type=int, default=45, help="Continuity horizon in minutes for each observation")
    parser.add_argument("--ai-csv", dest="ai_csv", default=None, help="Optional path to AI predictions CSV")
    parser.add_argument("--ai-horizon", dest="ai_horizon", type=int, default=4, help="Continuity horizon in minutes for AI observations (default: 4)")
    parser.add_argument("--ai-col", dest="ai_col", default="predicted_project", help="AI CSV project column name (default: predicted_project)")
    parser.add_argument("--ai-ts-format", dest="ai_ts_format", default="%Y%m%d_%H%M%S", help="AI CSV timestamp format string (default: %Y%m%d_%H%M%S)")
    parser.add_argument(
        "--ai-naive-origin",
        dest="ai_naive_origin",
        choices=["local", "utc"],
        default="local",
        help="Origin for naive AI timestamps when using --ai-ts-format (default: local)",
    )
    parser.add_argument("--width", dest="width", type=int, default=1400, help="Figure width in pixels")
    parser.add_argument("--height", dest="height", type=int, default=600, help="Figure height in pixels")
    default_compact = "/Users/michaelryan/Documents/School/Stanford/Research/background-agents/dev/survey/analysis/project_classification/results/timeline_compact.png"
    parser.add_argument("--compact-output", dest="compact_output", default=default_compact, help="Optional path to save compact plot that skips idle gaps")
    parser.add_argument("--compact-gap", dest="compact_gap", type=int, default=60, help="Minimum idle gap minutes to remove in compact plot (default: 60)")
    # Optional AI with history
    default_hist = "/Users/michaelryan/Documents/School/Stanford/Research/background-agents/dev/survey/analysis/project_classification/results/context_project_predictions_openai_gpt-5_with_history.csv"
    parser.add_argument("--ai-history-csv", dest="ai_history_csv", default=None, help="Optional path to AI-with-history predictions CSV")
    parser.add_argument("--ai-history-col", dest="ai_history_col", default="predicted_project", help="AI-with-history CSV project column name (default: predicted_project)")

    args = parser.parse_args()
    # Matplotlib uses inches; convert px to inches assuming 100 dpi baseline, real dpi set in savefig
    fig_w_in = args.width / 100.0
    fig_h_in = args.height / 100.0

    plot_timelines(
        human_csv=args.human_csv,
        output_path=args.output_path,
        tz=args.tz,
        horizon_minutes=args.horizon,
        ai_csv=args.ai_csv,
        ai_horizon_minutes=args.ai_horizon,
        ai_project_col=args.ai_col,
        ai_timestamp_format=args.ai_ts_format,
        ai_naive_origin=args.ai_naive_origin,
        compact_output_path=args.compact_output,
        compact_min_gap_minutes=args.compact_gap,
        ai_history_csv=args.ai_history_csv,
        ai_history_project_col=args.ai_history_col,
        figure_size=(fig_w_in, fig_h_in),
    )


if __name__ == "__main__":
    main()


