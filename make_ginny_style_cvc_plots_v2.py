"""
Ginny-style CVC fNIRS visualisation - revised version

Fixes compared with the previous version:
1. The condition window is inferred primarily from the Block tier, not only the Condition tier.
   This avoids continuous sections, especially HWCE, being truncated.
2. If Block tier is incomplete, it falls back to Condition tier.
3. It prints diagnostics for each panel: window, number of rows, and y-range.
4. If a panel has no valid signal, it writes a message instead of silently producing a misleading flat plot.

Expected folder structure:

project_folder/
    make_ginny_style_cvc_plots_v2.py

    studyB_time_series/
        ts_subj_1_sess_1.csv

    studyB_annotations/
        CVC_0001_Annotations.csv

Output:
    studyB_ginny_style_plots_v2/
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


TIME_SERIES_DIR = Path("cvc_time_series_data")
ANNOTATION_DIR = Path("studyB_annotations")
OUTPUT_DIR = Path("studyB_ginny_style_plots_v2")

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
    "Procedure": "#baf7d0",
    "Unknown": "#eeeeee",
}

OPERATIVE_COLOUR = "#198c3a"
EXTRANEOUS_COLOUR = "#e36c18"

SIGNAL_NAMES = {
    1: "Signal 1",   # change to "HbO2" if confirmed
    2: "Signal 2",   # change to "HHb" if confirmed
}


def parse_subject_session(ts_path: Path):
    m = re.search(r"subj_(\d+)_sess_(\d+)", ts_path.name)
    if not m:
        raise ValueError(f"Cannot parse subject/session from filename: {ts_path.name}")
    return int(m.group(1)), int(m.group(2))


def find_annotation_file(subject: int):
    candidates = [
        ANNOTATION_DIR / f"CVC_{subject:04d}_Annotations.csv",
        ANNOTATION_DIR / f"CVC_{subject:03d}_Annotations.csv",
        ANNOTATION_DIR / f"CVC_{subject}_Annotations.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find annotation file for subject {subject}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    fixes = {
        "Ultrasound  scanning": "Ultrasound scanning",
        "Postion check (ultrasound)": "Position check (ultrasound)",
        "Wire Advancement": "Wire advancement",
        "Wire Removal": "Wire removal",
        "Wire Removal (with catheter)": "Wire removal",
        "Wire Removal  (with catheter)": "Wire removal",
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
    a["Tier_clean"] = a["Tier"].astype(str).str.strip()
    a["start_s_abs"] = a["Start_ms"] / 1000.0
    a["end_s_abs"] = a["End_ms"] / 1000.0
    a["condition_code"] = a["Text_clean"].apply(get_condition_code)
    return a


def operative_marker(text):
    t = str(text).lower()
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
    t = str(text).lower()
    if "bleep" in t:
        return "o", "Bleep"
    if (
        "query" in t
        or "question" in t
        or "medical student" in t
        or "odp" in t
        or "clinical query" in t
    ):
        return "D", "Cognitive interruption"
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
    text = str(text)
    if "Recovery" in text or "Baseline" in text:
        return "Rest"
    m = re.search(r"-(A|B|C)-", text)
    if m:
        return m.group(1)
    if "Procedure" in text:
        return "Procedure"
    return "Unknown"


def get_condition_window(annp: pd.DataFrame, condition_code: str):
    """
    Prefer Block tier rows containing the condition code.
    This captures the complete A/Rest/B/Rest/C or Procedure/Rest structure.
    Fall back to Condition tier only if no Block rows exist.
    """
    block_rows = annp[
        (annp["Tier_clean"].str.lower() == "block")
        & (annp["Text_clean"].astype(str).str.contains(condition_code, regex=False, na=False))
    ]
    if not block_rows.empty:
        return block_rows["start_s_abs"].min(), block_rows["end_s_abs"].max(), "Block"

    condition_rows = annp[
        (annp["Tier_clean"].str.lower() == "condition")
        & (annp["Text_clean"].astype(str).str.contains(condition_code, regex=False, na=False))
    ]
    if not condition_rows.empty:
        return condition_rows["start_s_abs"].min(), condition_rows["end_s_abs"].max(), "Condition"

    return None


def get_relative_rows(annp: pd.DataFrame, condition_code: str, tier: str):
    window = get_condition_window(annp, condition_code)
    if window is None:
        return pd.DataFrame()
    start, end, _source = window
    rows = annp[
        (annp["Tier_clean"].str.lower() == tier.lower())
        & (annp["start_s_abs"] >= start)
        & (annp["start_s_abs"] <= end)
    ].copy()
    rows["start_s"] = rows["start_s_abs"] - start
    rows["end_s"] = rows["end_s_abs"] - start
    return rows


def get_condition_blocks(annp: pd.DataFrame, condition_code: str):
    window = get_condition_window(annp, condition_code)
    if window is None:
        return pd.DataFrame()
    start, end, _source = window

    rows = annp[
        (annp["Tier_clean"].str.lower() == "block")
        & (annp["Text_clean"].astype(str).str.contains(condition_code, regex=False, na=False))
    ].copy()

    if rows.empty:
        rows = annp[
            (annp["Tier_clean"].str.lower() == "block")
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


def draw_background_blocks(ax, block_rows: pd.DataFrame, x_max: float):
    if block_rows.empty:
        ax.axvspan(0, x_max, color=PERIOD_COLOURS["Unknown"], alpha=0.18, zorder=0)
        return

    for _, r in block_rows.iterrows():
        label = block_label_from_text(r["Text_clean"])
        x0 = max(0, float(r["start_s"]))
        x1 = min(x_max, float(r["end_s"]))
        if x1 <= x0:
            continue
        colour = PERIOD_COLOURS.get(label, PERIOD_COLOURS["Unknown"])
        ax.axvspan(x0, x1, color=colour, alpha=0.50, zorder=0)
        if x1 - x0 > 10:
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

    ax.text(-0.015, y_stage, "Operative\nstage", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, color=OPERATIVE_COLOUR)
    ax.text(-0.015, y_extra, "Extraneous\nload", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, color=EXTRANEOUS_COLOUR)

    for _, r in stage_rows.iterrows():
        marker, _label = operative_marker(r["Text_clean"])
        if marker is None:
            continue
        ax.scatter(
            r["start_s"], y_stage,
            marker=marker,
            s=70 if marker != "*" else 105,
            color=OPERATIVE_COLOUR,
            facecolors="none" if marker in ["D", "^"] else OPERATIVE_COLOUR,
            linewidths=1.4,
            zorder=5,
        )

    for _, r in event_rows.iterrows():
        marker, _label = extraneous_marker(r["Text_clean"])
        if marker is None:
            continue
        ax.scatter(
            r["start_s"], y_extra,
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
        Rectangle((0, 0), 1, 1, facecolor=PERIOD_COLOURS["Procedure"], alpha=0.50, label="Procedure"),
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
    fig.legend(handles=period_handles + operative_handles + extraneous_handles,
               loc="lower center", ncol=6, frameon=False, fontsize=8)


def plot_channel_four_conditions(ts: pd.DataFrame, annp: pd.DataFrame, subject: int, session: int, channel_col: str, signal_id=1):
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=False)

    for ax, condition_code in zip(axes, PANEL_ORDER):
        window = get_condition_window(annp, condition_code)
        if window is None:
            ax.set_title(f"{CONDITION_MAP[condition_code]} - not found", loc="left")
            ax.text(0.5, 0.5, "Condition not found in annotation file", transform=ax.transAxes,
                    ha="center", va="center")
            ax.axis("off")
            continue

        c_start, c_end, source = window
        x_max = c_end - c_start
        signal = get_relative_signal(ts, c_start, c_end, signal_id, channel_col)
        block_rows = get_condition_blocks(annp, condition_code)
        stage_rows = get_relative_rows(annp, condition_code, "Stage")
        event_rows = get_relative_rows(annp, condition_code, "Event")

        if signal.empty:
            y_range_text = "empty"
        else:
            y_range_text = f"{signal[channel_col].min():.3f} to {signal[channel_col].max():.3f}"
        print(
            f"    {channel_col} | signal {signal_id} | {condition_code}: "
            f"window={c_start:.2f}-{c_end:.2f}s ({x_max:.2f}s, source={source}), "
            f"rows={len(signal)}, y_range={y_range_text}"
        )

        draw_background_blocks(ax, block_rows, x_max)

        if signal.empty:
            ax.text(0.5, 0.45, "No valid signal data in this condition window",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11, color="0.3")
            ax.set_ylim(0, 1)
        else:
            y = signal[channel_col]
            y_min, y_max = float(y.min()), float(y.max())
            if np.isclose(y_min, y_max):
                pad = max(abs(y_min) * 0.1, 1.0)
                ax.set_ylim(y_min - pad, y_max + pad)
            ax.plot(signal["time_rel"], signal[channel_col],
                    color=CONDITION_TRACE_COLOUR[condition_code], linewidth=1.1)

        y_min, y_max = ax.get_ylim()
        span = y_max - y_min if y_max != y_min else 1.0
        y_stage = y_max + 0.20 * span
        y_extra = y_max + 0.08 * span
        ax.set_ylim(y_min, y_max + 0.35 * span)

        draw_annotation_timelines(ax, stage_rows, event_rows, y_stage, y_extra)

        ax.set_xlim(0, max(x_max, 1))
        ax.set_title(f"{CONDITION_MAP[condition_code]}  [window from {source} tier]", loc="left", fontsize=11)
        ax.set_ylabel(f"{channel_col}\n{SIGNAL_NAMES.get(signal_id, f'Signal {signal_id}')}")
        ax.set_xlabel("time from condition onset / seconds")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Subject {subject} Session {session} - Channel {channel_col.replace('Var', '')} - "
        f"{SIGNAL_NAMES.get(signal_id, f'Signal {signal_id}')}",
        fontsize=14,
        y=0.98,
    )
    add_custom_legend(fig)
    fig.tight_layout(rect=[0.04, 0.08, 1, 0.96])
    return fig


def process_one_file(ts_path: Path, signal_id=1):
    subject, session = parse_subject_session(ts_path)
    ann_path = find_annotation_file(subject)

    print(f"\nProcessing subject {subject}, session {session}")
    print(f"  time series: {ts_path}")
    print(f"  annotation:  {ann_path}")

    # ts = pd.read_csv(ts_path)
    # ann = pd.read_csv(ann_path)
    # annp = prepare_annotations(ann)
    
    ts = pd.read_csv(ts_path)
    ann = pd.read_csv(ann_path)
    annp = prepare_annotations(ann)

    # =========================
    # DEBUG: check recording range
    # =========================
    print("\n=== Time-series range check ===")

    ts_min = ts["timestamp"].min()
    ts_max = ts["timestamp"].max()

    print(f"timestamp min: {ts_min:.2f}s")
    print(f"timestamp max: {ts_max:.2f}s")

    # Compare against annotation condition windows
    for cond in PANEL_ORDER:
        window = get_condition_window(annp, cond)

        if window is None:
            print(f"{cond}: no annotation window found")
            continue

        c_start, c_end, source = window

        print(
            f"{cond}: annotation window = "
            f"{c_start:.2f}s -> {c_end:.2f}s "
            f"(source={source})"
        )

        # check overlap with recording
        if c_start > ts_max:
            print(
                f"  WARNING: {cond} starts AFTER recording ended"
            )

        elif c_end > ts_max:
            print(
                f"  WARNING: {cond} ends AFTER recording ended"
            )

    print("================================\n")
    channel_cols = sorted(
        [c for c in ts.columns if re.fullmatch(r"Var\d+", c)],
        key=lambda x: int(x.replace("Var", ""))
    )

    subject_out = OUTPUT_DIR / f"subj_{subject}_sess_{session}"
    subject_out.mkdir(parents=True, exist_ok=True)

    for channel_col in channel_cols:
        fig = plot_channel_four_conditions(ts, annp, subject, session, channel_col, signal_id=signal_id)
        out_path = subject_out / (
            f"ginny_style_v2_subj_{subject}_sess_{session}_"
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
        raise FileNotFoundError(f"No time-series files found in {TIME_SERIES_DIR.resolve()}")

    # Default: signal 1 only. To plot both signals, change [1] to [1, 2].
    for signal_id in [1]:
        for ts_path in ts_files:
            process_one_file(ts_path, signal_id=signal_id)

    print(f"\nDone. All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
