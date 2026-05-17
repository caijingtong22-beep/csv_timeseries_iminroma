"""
Ginny-style CVC fNIRS visualisation

Input logic:
    ts_subj_x_sess_y.csv
    +
    CVC_000x_Annotations.csv

Output:
    For each subject/session/channel:
        1 figure with 4 panels:
            Low CWL Block
            High CWL Block
            Low CWL Continuous
            High CWL Continuous

Each panel:
    - x-axis reset to condition onset = 0
    - faint background blocks: A / Rest / B / Rest / C
    - green operative-stage onset markers
    - orange extraneous-load onset markers
    - fNIRS trace
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# =========================
# 1. Change these folders
# =========================

TIME_SERIES_DIR = Path("cvc_time_series_data")
ANNOTATION_DIR = Path("studyB_annotations")
OUTPUT_DIR = Path("studyB_ginny_style_plots")

# Example folder structure:
#
# studyB_time_series/
#   ts_subj_1_sess_1.csv
#
# studyB_annotations/
#   CVC_0001_Annotations.csv


# =========================
# 2. Visual settings
# =========================

CONDITION_MAP = {
    "LWBE": "Low CWL Block",
    "HWBE": "High CWL Block",
    "LWCE": "Low CWL Continuous",
    "HWCE": "High CWL Continuous",
}

CONDITION_TRACE_COLOUR = {
    "LWBE": "tab:blue",
    "LWCE": "tab:blue",
    "HWBE": "tab:red",
    "HWCE": "tab:red",
}

PANEL_ORDER = ["LWBE", "HWBE", "LWCE", "HWCE"]

PERIOD_COLOURS = {
    "A": "#fff3a8",
    "B": "#baf7d0",
    "C": "#ffd6c2",
    "Rest": "#cfe5ff",
    "Baseline": "#cfe5ff",
    "Recovery": "#cfe5ff",
    "Procedure": "#baf7d0",
}

OPERATIVE_COLOUR = "#198c3a"
EXTRANEOUS_COLOUR = "#e36c18"

# If you confirm signal coding later, change names here.
SIGNAL_NAMES = {
    1: "Signal 1",
    2: "Signal 2",
    # 1: "HbO2",
    # 2: "HHb",
}


# =========================
# 3. File utilities
# =========================

def parse_subject_session(ts_path: Path):
    """
    Parse filenames like:
        ts_subj_1_sess_1.csv
    """
    m = re.search(r"subj_(\d+)_sess_(\d+)", ts_path.name)
    if not m:
        raise ValueError(f"Cannot parse subject/session from filename: {ts_path.name}")
    return int(m.group(1)), int(m.group(2))


def find_annotation_file(subject: int):
    """
    Match subject 1 to CVC_0001_Annotations.csv.
    """
    candidates = [
        ANNOTATION_DIR / f"CVC_{subject:04d}_Annotations.csv",
        ANNOTATION_DIR / f"CVC_{subject:03d}_Annotations.csv",
        ANNOTATION_DIR / f"CVC_{subject}_Annotations.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find annotation file for subject "
        f"{subject}. Tried: " + ", ".join(str(c) for c in candidates)
    )


# =========================
# 4. Annotation cleaning
# =========================

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    fixes = {
        "Ultrasound  scanning": "Ultrasound scanning",
        "Postion check (ultrasound)": "Position check (ultrasound)",
        "Wire Advancement": "Wire advancement",
        "Wire Removal": "Wire removal",
        "Wire Removal (with catheter)": "Wire removal",
    }
    return fixes.get(text, text)


def get_condition_code(text):
    text = str(text)
    for code in PANEL_ORDER:
        if code in text:
            return code
    return None


def prepare_annotations(ann: pd.DataFrame):
    required = {"Tier", "Start_ms", "End_ms", "Text"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {missing}")

    a = ann.copy()
    a["Text_clean"] = a["Text"].apply(clean_text)
    a["start_s_abs"] = a["Start_ms"] / 1000.0
    a["end_s_abs"] = a["End_ms"] / 1000.0
    a["condition_code"] = a["Text_clean"].apply(get_condition_code)
    return a


# =========================
# 5. Categorisation rules
# =========================

def operative_marker(text):
    """
    Return marker symbol for operative stages.

    Ginny's sketch:
        ● US / position check
        * Venipuncture
        ◇ Wire
        △ Catheter
    """
    t = text.lower()

    if "ultrasound" in t or "position check" in t:
        return "o", "US / position check"

    if "venipuncture" in t:
        return "*", "Venipuncture"

    if "wire" in t:
        return "D", "Wire adv/removal"

    if "catheter" in t:
        return "^", "Catheterisation"

    return None, None


def extraneous_marker(text):
    """
    Return marker symbol for extraneous load.

    Ginny's sketch:
        ● Bleep
        ◇ Cognitive interruption
        * Other
    """
    t = text.lower()

    if "bleep" in t:
        return "o", "Bleep"

    # Cognitive interruption: questions / queries / ODP / med student.
    if (
        "query" in t
        or "question" in t
        or "medical student" in t
        or "odp" in t
        or "clinical query" in t
    ):
        return "D", "Cognitive interruption"

    # Clinically salient / miscellaneous interesting events.
    if (
        "unsure" in t
        or "anatomy" in t
        or "not happy" in t
        or "other" in t
        or "ectopics" in t
    ):
        return "*", "Other / clinically salient"

    return None, None


def block_label_from_text(text):
    """
    Convert block labels into A / B / C / Rest.

    Examples:
        LWBE-A-Baseline -> A
        LWBE-A-Procedure -> A
        LWBE-A-Recovery/ LWBE-B-Baseline -> Rest
        HWCE-Procedure -> B or Procedure fallback
    """
    text = str(text)

    if "Recovery" in text or "Baseline" in text:
        return "Rest"

    # Prefer A/B/C if present.
    m = re.search(r"-(A|B|C)-", text)
    if m:
        return m.group(1)

    # Continuous files may not have A/B/C in the same way.
    if "Procedure" in text:
        return "Procedure"

    return "Rest"


# =========================
# 6. Timeline extraction
# =========================

def get_condition_window(annp: pd.DataFrame, condition_code: str):
    cond_rows = annp[
        (annp["Tier"].str.lower() == "condition")
        & (annp["condition_code"] == condition_code)
    ]

    if cond_rows.empty:
        return None

    start = cond_rows["start_s_abs"].min()
    end = cond_rows["end_s_abs"].max()
    return start, end


def get_relative_rows(annp: pd.DataFrame, condition_code: str, tier: str):
    window = get_condition_window(annp, condition_code)
    if window is None:
        return pd.DataFrame()

    start, end = window

    rows = annp[
        (annp["Tier"].str.lower() == tier.lower())
        & (annp["start_s_abs"] >= start)
        & (annp["start_s_abs"] <= end)
    ].copy()

    rows["start_s"] = rows["start_s_abs"] - start
    rows["end_s"] = rows["end_s_abs"] - start
    return rows


def get_relative_signal(ts: pd.DataFrame, condition_start: float, condition_end: float, signal_id, channel_col):
    d = ts[
        (ts["signal"] == signal_id)
        & (ts["timestamp"] >= condition_start)
        & (ts["timestamp"] <= condition_end)
    ].copy()

    d["time_rel"] = d["timestamp"] - condition_start
    return d[["time_rel", channel_col]].dropna()


# =========================
# 7. Plotting
# =========================

def draw_background_blocks(ax, block_rows: pd.DataFrame, x_max: float):
    """
    Use Block tier to draw A / Rest / B / Rest / C.
    If block labels are incomplete, the plot still works.
    """
    if block_rows.empty:
        ax.add_patch(Rectangle((0, -1e9), x_max, 2e9, facecolor="#eeeeee", alpha=0.15))
        return

    for _, r in block_rows.iterrows():
        label = block_label_from_text(r["Text_clean"])
        x0 = max(0, r["start_s"])
        x1 = min(x_max, r["end_s"])
        if x1 <= x0:
            continue

        colour = PERIOD_COLOURS.get(label, "#eeeeee")
        ax.axvspan(x0, x1, color=colour, alpha=0.50, zorder=0)

        # label only if span is wide enough
        if x1 - x0 > 12:
            ax.text(
                (x0 + x1) / 2,
                0.94,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                alpha=0.75,
            )


def draw_annotation_timelines(ax, stage_rows: pd.DataFrame, event_rows: pd.DataFrame, y_stage, y_extra):
    ax.axhline(y_stage, color=OPERATIVE_COLOUR, linewidth=1.0, alpha=0.8)
    ax.axhline(y_extra, color=EXTRANEOUS_COLOUR, linewidth=1.0, alpha=0.8)

    ax.text(
        -0.015,
        y_stage,
        "Operative\nstage",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8,
        color=OPERATIVE_COLOUR,
    )

    ax.text(
        -0.015,
        y_extra,
        "Extraneous\nload",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8,
        color=EXTRANEOUS_COLOUR,
    )

    # Operative stage markers: onset only.
    for _, r in stage_rows.iterrows():
        marker, _ = operative_marker(r["Text_clean"])
        if marker is None:
            continue

        ax.scatter(
            r["start_s"],
            y_stage,
            marker=marker,
            s=70 if marker != "*" else 105,
            color=OPERATIVE_COLOUR,
            facecolors="none" if marker in ["D", "^"] else OPERATIVE_COLOUR,
            linewidths=1.4,
            zorder=5,
        )

    # Extraneous markers: onset only.
    for _, r in event_rows.iterrows():
        marker, _ = extraneous_marker(r["Text_clean"])
        if marker is None:
            continue

        ax.scatter(
            r["start_s"],
            y_extra,
            marker=marker,
            s=64 if marker != "*" else 105,
            color=EXTRANEOUS_COLOUR,
            facecolors="none" if marker == "D" else EXTRANEOUS_COLOUR,
            linewidths=1.4,
            zorder=5,
        )


def add_custom_legend(fig):
    period_handles = [
        Rectangle((0, 0), 1, 1, facecolor=PERIOD_COLOURS["A"], alpha=0.50, label="A"),
        Rectangle((0, 0), 1, 1, facecolor=PERIOD_COLOURS["B"], alpha=0.50, label="B"),
        Rectangle((0, 0), 1, 1, facecolor=PERIOD_COLOURS["C"], alpha=0.50, label="C"),
        Rectangle((0, 0), 1, 1, facecolor=PERIOD_COLOURS["Rest"], alpha=0.50, label="Rest"),
    ]

    operative_handles = [
        Line2D([0], [0], marker="o", color=OPERATIVE_COLOUR, markerfacecolor=OPERATIVE_COLOUR, lw=0, label="US / position check"),
        Line2D([0], [0], marker="*", color=OPERATIVE_COLOUR, markerfacecolor=OPERATIVE_COLOUR, lw=0, markersize=11, label="Venipuncture"),
        Line2D([0], [0], marker="D", color=OPERATIVE_COLOUR, markerfacecolor="none", lw=0, label="Wire adv/removal"),
        Line2D([0], [0], marker="^", color=OPERATIVE_COLOUR, markerfacecolor="none", lw=0, label="Catheterisation"),
    ]

    extraneous_handles = [
        Line2D([0], [0], marker="o", color=EXTRANEOUS_COLOUR, markerfacecolor=EXTRANEOUS_COLOUR, lw=0, label="Bleep"),
        Line2D([0], [0], marker="D", color=EXTRANEOUS_COLOUR, markerfacecolor="none", lw=0, label="Cognitive interruption"),
        Line2D([0], [0], marker="*", color=EXTRANEOUS_COLOUR, markerfacecolor=EXTRANEOUS_COLOUR, lw=0, markersize=11, label="Other / clinically salient"),
    ]

    fig.legend(
        handles=period_handles + operative_handles + extraneous_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
    )


def plot_channel_four_conditions(ts: pd.DataFrame, annp: pd.DataFrame, subject: int, session: int, channel_col: str, signal_id=1):
    """
    One output figure:
        subject + channel + signal
        4 panels = LWBE/HWBE/LWCE/HWCE
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=False)

    for ax, condition_code in zip(axes, PANEL_ORDER):
        window = get_condition_window(annp, condition_code)

        if window is None:
            ax.set_title(f"{CONDITION_MAP[condition_code]} - not found", loc="left")
            ax.axis("off")
            continue

        c_start, c_end = window
        x_max = c_end - c_start

        signal = get_relative_signal(ts, c_start, c_end, signal_id, channel_col)
        block_rows = get_relative_rows(annp, condition_code, "Block")
        stage_rows = get_relative_rows(annp, condition_code, "Stage")
        event_rows = get_relative_rows(annp, condition_code, "Event")

        draw_background_blocks(ax, block_rows, x_max)

        if not signal.empty:
            trace_colour = CONDITION_TRACE_COLOUR[condition_code]
            ax.plot(signal["time_rel"], signal[channel_col], color=trace_colour, linewidth=1.1)

        # Put timelines near the top of each panel.
        # Use y limits after plotting signal.
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        y_stage = y_max + 0.20 * span
        y_extra = y_max + 0.08 * span
        ax.set_ylim(y_min, y_max + 0.35 * span)

        draw_annotation_timelines(ax, stage_rows, event_rows, y_stage, y_extra)

        ax.set_xlim(0, x_max)
        ax.set_title(CONDITION_MAP[condition_code], loc="left", fontsize=11)
        ax.set_ylabel(f"{channel_col}\n{SIGNAL_NAMES.get(signal_id, f'Signal {signal_id}')}")
        ax.set_xlabel("time from condition onset / seconds")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Subject {subject} Session {session} - Channel {channel_col.replace('Var', '')} - {SIGNAL_NAMES.get(signal_id, f'Signal {signal_id}')}",
        fontsize=14,
        y=0.98,
    )

    add_custom_legend(fig)
    fig.tight_layout(rect=[0.04, 0.07, 1, 0.96])
    return fig


# =========================
# 8. Main batch process
# =========================

def process_one_file(ts_path: Path, signal_id=1):
    subject, session = parse_subject_session(ts_path)
    ann_path = find_annotation_file(subject)

    print(f"Processing subject {subject}, session {session}")
    print(f"  time series: {ts_path}")
    print(f"  annotation:  {ann_path}")

    ts = pd.read_csv(ts_path)
    ann = pd.read_csv(ann_path)
    annp = prepare_annotations(ann)

    channel_cols = sorted(
        [c for c in ts.columns if re.fullmatch(r"Var\d+", c)],
        key=lambda x: int(x.replace("Var", ""))
    )

    subject_out = OUTPUT_DIR / f"subj_{subject}_sess_{session}"
    subject_out.mkdir(parents=True, exist_ok=True)

    for channel_col in channel_cols:
        fig = plot_channel_four_conditions(
            ts=ts,
            annp=annp,
            subject=subject,
            session=session,
            channel_col=channel_col,
            signal_id=signal_id,
        )

        out_path = subject_out / (
            f"ginny_style_subj_{subject}_sess_{session}_"
            f"channel_{channel_col.replace('Var', '').zfill(2)}_"
            f"signal_{signal_id}.png"
        )

        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    print(f"  saved {len(channel_cols)} figures to {subject_out}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    ts_files = sorted(TIME_SERIES_DIR.glob("ts_subj_*_sess_*.csv"))
    if not ts_files:
        raise FileNotFoundError(
            f"No time-series files found in {TIME_SERIES_DIR.resolve()}"
        )

    # Default: plot signal 1 only.
    # If you also want signal 2, change to: for signal_id in [1, 2]
    for ts_path in ts_files:
        process_one_file(ts_path, signal_id=1)

    print(f"Done. All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
