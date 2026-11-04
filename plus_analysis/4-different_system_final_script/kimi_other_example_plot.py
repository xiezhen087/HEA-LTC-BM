# -*- coding: utf-8 -*-
"""
kimi_other_example_plot.py — figure for the additional refractory series
validation. Bars: model dB% (5-seed mean +/- std). Stars: literature changes
(+V pair is on the SAME property, bulk modulus, from Hu 2020 DFT).
Input: ../other_example/validation_results.csv
Output: ../other_example/fig_other_refractory_series.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "other_example"))
TASK1_SCRIPT_DIR = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "..", "1-reference_plus", "final_script"))

plt.rcParams.update({"font.family": "Arial", "font.size": 16})
MODEL_COLOR = "#7fa8c9"
MODEL_EDGE = "#3d6e9e"
LIT_COLOR = "#c0392b"

# literature % changes (see validation_results.csv 'literature' for sources)
# for +V the literature value is on the SAME property (bulk modulus, DFT, Hu 2020)
LIT = {
    "+V (NbMoTaW)": -4.5,      # bulk modulus, VCA, Hu 2020
    "+Ti (NbMoTaW)": 46.0,     # yield strength, Han 2018
    "+W (HfNbTaTiZr)": 275.0,  # YS @1200C, Wang 2019
    "+Mo+W (HfNbTaTiZr)": 664.0,  # YS @1200C, Wang 2019
    "+Ti (NbTaV)": None,       # no quantitative value retrieved
    "+W (NbTaTiV)": None,
}


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "validation_results.csv"))
    fig, ax = plt.subplots(figsize=(11.5, 7), dpi=400)
    xs = np.arange(len(df))
    ax.bar(xs, df["d_bm_pct"], width=0.52, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
           label="This work (model, B)", zorder=2)
    YLIM = (-70, 80)
    for i, (_, row) in enumerate(df.iterrows()):
        v = LIT.get(row["pair"])
        if v is not None:
            if abs(v) <= YLIM[1]:
                ax.scatter(i + 0.30, v, marker="*", s=420, color=LIT_COLOR,
                           edgecolors="#7b241c", linewidths=1.0, zorder=3)
            else:
                # off-scale literature: marker at top edge with value annotated
                ax.scatter(i + 0.30, YLIM[1] - 4, marker="*", s=420, color=LIT_COLOR,
                           edgecolors="#7b241c", linewidths=1.0, zorder=3)
                ax.annotate(f"{v:+.0f}%", (i + 0.30, YLIM[1] - 4),
                            textcoords="offset points", xytext=(10, 2),
                            fontsize=12, color=LIT_COLOR, fontweight="bold")
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["pair"], rotation=12, ha="right")
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Change with element added [%]")
    ax.set_title("Refractory series validation: model dB vs literature", pad=12)
    ax.scatter([], [], marker="*", s=420, color=LIT_COLOR, edgecolors="#7b241c",
               label="Literature (see notes)")
    ax.legend(fontsize=12, frameon=False, loc="lower left")
    ax.grid(axis="y", ls=":", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.04,
             "Literature: +V: bulk modulus (Hu 2020, DFT); +Ti: YS (Han 2018); "
             "+W / +Mo+W: YS@1200C (Wang 2019, off-scale values annotated). "
             "Only +V shares the same property (B).",
             ha="center", fontsize=11, color="#555555")
    out = os.path.join(OUT_DIR, "fig_other_refractory_series.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.3)
    print("saved:", out)


if __name__ == "__main__":
    main()
