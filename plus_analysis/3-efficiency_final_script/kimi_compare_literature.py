# -*- coding: utf-8 -*-
"""
Comparison bar charts: This work (Mix-GNN) vs. 4 literature ML alloy-design models.

Three dimensions (each dimension only includes papers that actually report it):
  1. Element coverage (number of elements the model can handle / covers)
  2. Training data size (number of training entries)
  3. Computational cost (reported runtime + hardware)

Data extracted from the PDFs in the parent folder (see extracted_comparison_data.csv
for the exact source sentences / page clues).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "final_fig")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_LIT = "#7fa8c9"      # literature bars: light steel blue
C_WORK = "#d43f3a"     # this work: red
EDGE = "black"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def add_value_labels(ax, bars, fmt="{:,}", dy_frac=0.01, fontsize=10.5):
    ymax = ax.get_ylim()[1]
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * dy_frac,
                fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")


# ----------------------------------------------------------------------------
# Figure 1: element coverage
# ----------------------------------------------------------------------------
labels1 = [
    "Liu et al.\nnpj Comput. Mater.\n2020 (No. 8)",
    "Das & Pandey\nJ. Alloys Compd.\n2024 (No. 14)",
    "Li et al.\nJ. Alloys Compd.\n2024 (No. 13)",
    "Dai et al.\nAdv. Sci.\n2025 (No. 27)",
    "This work\n(Mix-GNN)",
]
values1 = [7, 20, 25, 42, 60]
colors1 = [C_LIT, C_LIT, C_LIT, C_LIT, C_WORK]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
bars = ax.bar(range(len(values1)), values1, width=0.62,
              color=colors1, edgecolor=EDGE, linewidth=0.8, zorder=3)
ax.set_xticks(range(len(values1)))
ax.set_xticklabels(labels1, fontsize=9)
ax.set_ylabel("Number of elements covered", fontsize=12)
ax.set_ylim(0, 72)
style_ax(ax)
add_value_labels(ax, bars, fmt="{:.0f}")
fig.text(0.5, 0.005,
         "Note: Liu et al. and Das & Pandey: optimization search-space element pool; Li et al.: elements present in the\n"
         "training data (their optimization search pool covered 9 elements); Dai et al.: element coverage of both training data and search space.",
         ha="center", va="bottom", fontsize=7.5, color="0.25")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "fig_element_coverage.png"), dpi=350)
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2: training data size (log scale)
# ----------------------------------------------------------------------------
labels2 = [
    "Li et al.\nJ. Alloys Compd.\n2024 (No. 13)",
    "Liu et al.\nnpj Comput. Mater.\n2020 (No. 8)",
    "This work\n(Mix-GNN)",
    "Dai et al.\nAdv. Sci.\n2025 (No. 27)",
]
values2 = [544, 1327, 18800, 460656]
colors2 = [C_LIT, C_LIT, C_WORK, C_LIT]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
bars = ax.bar(range(len(values2)), values2, width=0.55,
              color=colors2, edgecolor=EDGE, linewidth=0.8, zorder=3)
ax.set_yscale("log")
ax.set_ylim(100, 5e6)
ax.set_xticks(range(len(values2)))
ax.set_xticklabels(labels2, fontsize=9.5)
ax.set_ylabel("Training data entries (log scale)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
style_ax(ax)
for b, v in zip(bars, values2):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold")
fig.text(0.5, 0.005,
         "Li et al.: 544 experimental MPEA compositions; Liu et al.: 1,000 CALPHAD + 327 experimental entries;\n"
         "This work: 13,172 (Materials Project) + 5,652 (AFLOW); Dai et al.: 460,656 DFT-computed compositions "
         "(1,842,628 DFT calculations).\nDas & Pandey (No. 14) did not report the training set size and is not shown.",
         ha="center", va="bottom", fontsize=7.5, color="0.25")
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "fig_training_data_size.png"), dpi=350)
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 3: computational cost (reported runtime, log scale)
# ----------------------------------------------------------------------------
labels3 = [
    "Li et al.\nJ. Alloys Compd.\n2024 (No. 13)",
    "This work\n(Mix-GNN)",
]
values3 = [2, 120]
notes3 = ["~2 h (BO)\nhardware not reported", "~120 h\n1x NVIDIA RTX 4080"]
colors3 = [C_LIT, C_WORK]

fig, ax = plt.subplots(figsize=(5.2, 4.6))
bars = ax.bar(range(len(values3)), values3, width=0.42,
              color=colors3, edgecolor=EDGE, linewidth=0.8, zorder=3)
ax.set_yscale("log")
ax.set_ylim(1, 1000)
ax.set_xticks(range(len(values3)))
ax.set_xticklabels(labels3, fontsize=9.5)
ax.set_ylabel("Reported computation time (hours, log scale)", fontsize=11.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
style_ax(ax)
for b, v, n in zip(bars, values3, notes3):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.3, n,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
fig.text(0.5, 0.005,
         "Li et al.: Bayesian-optimization search ~2 h (exhaustive search ~14 h), hardware not reported.\n"
         "Dai et al. (No. 27) used the Lichtenberg HPC cluster but did not report runtime;\n"
         "Das & Pandey (No. 14) and Liu et al. (No. 8) did not report computational cost.",
         ha="center", va="bottom", fontsize=7.5, color="0.25")
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "fig_computational_cost.png"), dpi=350)
plt.close(fig)

print("Saved figures to:", OUT_DIR)
for f in ["fig_element_coverage.png", "fig_training_data_size.png", "fig_computational_cost.png"]:
    p = os.path.join(OUT_DIR, f)
    print(" ", f, os.path.getsize(p), "bytes")
