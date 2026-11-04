# -*- coding: utf-8 -*-
"""
kimi_efficiency_metrics_v2.py — Revised efficiency comparison figures.

Task division:
  - literature selection, metric definition, and manuscript use: (author)
  - metric table assembly, plotting code, and figure generation: Kimi

Three metrics, plotted as square bar charts in the same visual style as the
previous comparison figures:
  1. Element coverage
  2. Additional computational data generated for model training
  3. Reported time required to search/map the full candidate space

Literature labels contain author surname(s), year, and the final citation
number (mapped from reference-index.csv: Liu et al. [9], Torsti et al. [63],
Li et al. [64], Das & Pandey [65], Dai et al. [66]). Journal names and
explanatory notes are kept in efficiency_metrics_v2.csv for later use in the
manuscript or figure legends.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(BASE, "final_fig", "efficiency_metrics_v2.csv")
OUT_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "final_fig", "reference_figs"))
os.makedirs(OUT_DIR, exist_ok=True)

# final reference numbers (final_fig/reference-index.csv)
REF_NO = {"Li et al.": 64, "Liu et al.": 9, "Dai et al.": 66,
          "Das & Pandey": 65, "Torsti et al.": 63}


def resolve_refs(df):
    """Replace the [ref] placeholder in work_label with the final number."""
    out = df.copy()
    for author, no in REF_NO.items():
        out["work_label"] = out["work_label"].str.replace(
            f"{author} [ref]", f"{author} [{no}]", regex=False)
    return out

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_LIT = "#7fa8c9"
C_WORK = "#d43f3a"
EDGE = "black"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.grid(False)
    ax.set_axisbelow(True)


def x_label(row, label_notes=None):
    if row["work_label"] == "This work":
        label = "This work"
    else:
        label = f"{row['work_label']}\n{int(row['year'])}"
    if label_notes and row["work_label"] in label_notes:
        label += f"\n{label_notes[row['work_label']]}"
    return label


def order_bars(df):
    out = df.copy()
    out["__work_first"] = (out["work_label"] != "This work").astype(int)
    return out.sort_values(["__work_first", "value"], kind="stable").drop(columns="__work_first")


def plot_bar(df, ylabel, out_name, log_scale=False, symlog_zero=False,
             ylim=None, value_fmt="{:,}", time_labels=None, label_notes=None,
             xtick_fontsize=11.5):
    df = order_bars(df)
    labels = [x_label(row, label_notes) for _, row in df.iterrows()]
    values = df["value"].astype(float).tolist()
    colors = [C_WORK if x == "This work" else C_LIT for x in df["work_label"]]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    bars = ax.bar(range(len(values)), values, width=0.60,
                  color=colors, edgecolor=EDGE, linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=xtick_fontsize, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=14)
    if symlog_zero:
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(*ylim)
        ticks = [0, 10, 100, 1000, 10000, 100000, 1000000]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", "10", "100", "1,000", "10,000", "100,000", "1,000,000"])
    elif log_scale:
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
    elif ylim is not None:
        ax.set_ylim(*ylim)
    style_ax(ax)

    ymax = ax.get_ylim()[1]
    for i, (bar, value) in enumerate(zip(bars, values)):
        if time_labels is not None:
            label = time_labels[i]
        else:
            label = value_fmt.format(value)
        if symlog_zero:
            label_y = max(value * 1.20, 8)
        elif log_scale:
            label_y = value * 1.22
        else:
            label_y = value + ymax * 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, label,
                ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("saved:", out_path)


def main():
    data = resolve_refs(pd.read_csv(DATA_PATH, encoding="utf-8-sig"))

    coverage = data[data["metric"] == "element_coverage"]
    plot_bar(
        coverage,
        "Number of elements covered",
        "fig_v2_element_coverage.png",
        ylim=(0, 72),
        value_fmt="{:.0f}",
    )

    computed = data[data["metric"] == "additional_computed_training_data"]
    computed_notes = {
        "Li et al. [64]": "(from literature)",
        "This work": "(Transferred from\nuniversal databases)",
    }
    plot_bar(
        computed,
        "Newly computed training data entries",
        "fig_v2_additional_computed_data.png",
        symlog_zero=True,
        ylim=(0, 5e6),
        value_fmt="{:,.0f}",
        label_notes=computed_notes,
        xtick_fontsize=10.0,
    )

    # Keep the same order used by plot_bar when building the time labels.
    search = data[(data["metric"] == "full_space_search_time") & data["value"].notna()]
    search = order_bars(search)
    time_labels = []
    for _, row in search.iterrows():
        if row["unit"] == "CPU h":
            time_labels.append(f"{row['value']:.0f} CPU·h")
        else:
            time_labels.append(f"{row['value']:.0f} h")
    plot_bar(
        search,
        "Reported full-space search time [h]",
        "fig_v2_full_space_search_time.png",
        log_scale=True,
        ylim=(5, 500),
        time_labels=time_labels,
    )


if __name__ == "__main__":
    main()
