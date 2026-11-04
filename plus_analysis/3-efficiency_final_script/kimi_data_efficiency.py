# -*- coding: utf-8 -*-
"""
kimi_data_efficiency.py
Task 3 addendum: instead of raw training-data size, compare
**training data entries / element coverage** (data per element) — a data-
efficiency metric. Lower = more data-efficient (broader element space per
training entry).

Values from extracted_comparison_data.csv (see ../final_fig/):
  Liu et al. (npj Comput. Mater. 2020, No. 8):  1,327 entries / 7 elements
  Li et al. (J. Alloys Compd. 2024, No. 13):      544 entries / 25 elements
  Dai et al. (Adv. Sci. 2025, No. 27):        460,656 entries / 42 elements
  This work (Mix-GNN):                         18,800 entries / 60 elements
  Das & Pandey (No. 14): training size not reported -> excluded from this plot.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "final_fig"))

entries = [
    ("Liu et al.\nnpj Comput. Mater.\n2020 (No. 8)", 1327, 7),
    ("Li et al.\nJ. Alloys Compd.\n2024 (No. 13)", 544, 25),
    ("Dai et al.\nAdv. Sci.\n2025 (No. 27)", 460656, 42),
    ("This work\n(Mix-GNN)", 18800, 60),
]

plt.rcParams.update({"font.family": "Arial", "font.size": 15})

fig, ax = plt.subplots(figsize=(9.5, 6), dpi=400)
names = [e[0] for e in entries]
ratios = [e[1] / e[2] for e in entries]
colors = ["#6a9ac4"] * (len(entries) - 1) + ["#d9534f"]

bars = ax.bar(range(len(entries)), ratios, color=colors, edgecolor="#444444",
              linewidth=0.8, width=0.62, zorder=2)
for i, (r, e) in enumerate(zip(ratios, entries)):
    ax.text(i, r * 1.08, f"{r:,.0f}", ha="center", fontsize=15, fontweight="bold")
    ax.text(i, r * 0.55 if r > 60 else r * 1.55,
            f"{e[1]:,} entries /\n{e[2]} elements", ha="center", fontsize=10,
            color="white" if r > 60 else "#333333")

ax.set_yscale("log")
ax.set_ylim(10, 60000)
ax.set_xticks(range(len(entries)))
ax.set_xticklabels(names, fontsize=11.5)
ax.set_ylabel("Training data entries per element covered (log scale)")
ax.set_title("Data efficiency: training data size / element coverage", pad=14)
ax.grid(axis="y", ls=":", color="#dddddd", lw=0.8, zorder=1)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

note = ("Note: lower = broader element space covered per training entry. "
        "Das & Pandey (No. 14) did not report training-set size and is not shown.\n"
        "Li et al.: 544 experimental MPEA compositions / 25 elements in training data; "
        "Dai et al.: 460,656 DFT-computed entries / 42 elements.")
fig.text(0.5, -0.04, note, ha="center", fontsize=9.5, color="#555555")

out = os.path.join(FIG_DIR, "fig_data_efficiency.png")
fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.25)
print("saved:", out)
for n, e in zip(names, entries):
    print(n.replace("\n", " "), "->", round(e[1] / e[2], 1), "entries/element")
