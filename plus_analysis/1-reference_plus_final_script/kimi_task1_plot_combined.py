# -*- coding: utf-8 -*-
"""
kimi_task1_plot_combined.py — side-by-side figure:

Task division:
  - alloy modeling, model prediction and literature-data curation: (author)
  - combined plotting and CI-shading variant: Kimi

  left:  Experiment Reference (replicates reference_painting.py using the
         original reference.csv, original Type colors and styles; each
         composition label carries its final reference number, LIT_REFS)
  right: Model Prediction (final task-1 figure: random substitution,
         multi-seed mean +/- std, lighter hue-matched colors; no refs)

All drawing is done in matplotlib (no AI-generated imagery).
Output: ../../final_fig/reference_figs/fig_experiment_vs_prediction.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from kimi_task1_plot import (alloy_type, lighten, FACE_OVERRIDE, BASE_COLOR,
                             TYPE_LABEL, TYPE_ORDER, MARKERS, short_label,
                             draw_uncertainty_point)

REF_CSV = os.path.join(TASK1_DIR, "reference.csv")
PRED_CSV = os.path.join(TASK1_DIR, "final_data", "predicted_bulk_modulus_and_ltc_random.csv")
FIG_OUT = os.path.abspath(os.path.join(TASK1_DIR, "..", "..", "final_fig",
                                       "reference_figs", "fig_experiment_vs_prediction.png"))

# original palette of reference_painting.py, indexed by reference.csv "Type"
COLOR_ID = ['#FF5E5E', '#00A1CB', '#FABE28', '#2B3E51', '#8A2BE2', '#FF5A5F', '#76D7EA']

# final reference numbers for the literature (experiment) panel, verified
# against final_fig/参考文献.xlsx (DOIs) and final_fig/6family.txt (numbering);
# Chou 2009 [81] numbering follows final_fig/reference-index.csv
LIT_REFS = {
    "CoCrFeNi": "[81,107,108,111]",
    "FeCoNiMn": "[107]",
    "NiCoMnCr": "[107]",
    "CrMnFeCoNi(Cantor)": "[109-111]",
    "NiCoFeCrPd": "[107,111]",
    "Al0.5CoCrFeNi": "[81,112,113]",
    "AlCoCrFeNi": "[81,112,113]",
    "Al1.5CoCrFeNi": "[81,112,113]",
    "Al2CoCrFeNi": "[81,112,113]",
    "Cu5.3Cr36.8Fe36.8V21": "[114]",
    "Cu5Cr35Fe35V20Ti5": "[114]",
    "Cu5Cr35Fe35V20Ta5": "[114]",
    "Cu5Cr35Fe35V20W5": "[114]",
    "Cu5Cr35Fe35V20Mo5": "[114]",
    "Mg-2Zn-3.6Cu-0.5Ce": "[102]",
    "Mg-Al-Ca-Mn-Cu (AXMC)": "[103,104]",
    "Mg-Al-Ca-Mn (AXM)": "[103,106]",
    "Mg-Al-Zn (AZ31)": "[103,105]",
    "T22-TiC-RS14": "[115]",
    "T22-TiC-RS16": "[115]",
    "T22-TiC-RS18": "[115]",
    "TaTiNb": "[99]",
    "TaTiNbZr": "[99]",
    "TaTiNbZrMo": "[99]",
    "TaTiNbZrW": "[99]",
    "TaTiNb-HEBM": "[99]",
    "TaTiNbZr-HEBM": "[99]",
    "TaTiNbZrMo-HEBM": "[99]",
    "TaTiNbZrW-HEBM": "[99]",
    "TiZrHfNb": "[100,101]",
}

# the four Mg-alloy legend entries put their reference numbers on a second
# line, so the second legend column stays narrow enough at the original anchor
TWO_LINE_REFS = {"Mg-2Zn-3.6Cu-0.5Ce", "Mg-Al-Ca-Mn-Cu (AXMC)",
                 "Mg-Al-Ca-Mn (AXM)", "Mg-Al-Zn (AZ31)"}


def model_func(x, a, b):
    return a / x + b


def draw_experiment(ax, uncertainty_style="errorbar"):
    """Replicates reference_painting.py (original colors, markers, fit curve)."""
    df = pd.read_csv(REF_CSV)
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h', ',', '.', r'$\star$', r'$\triangle$']

    # this-work alloys (manuscript Table 7; RE-3 hardness 567 per Table 7)
    pt_1_hardness, pt_1_tc = 816, 8.583
    re_3_hardness, re_3_tc = 567, 9.519
    re_4_hardness, re_4_tc = 801, 18.002

    color = [COLOR_ID[int(df["Type"][i])] for i in range(len(df["Type"]))]
    x_data = df["Thermal Conductivity at 300 K"]
    y_data = df["Hardness"]
    x_err = df.iloc[:, -1]   # Error Bar 2 = thermal conductivity error
    y_err = df.iloc[:, -2]   # Error Bar 1 = hardness error

    j = 0
    last_type = df["Type"][0]
    for i in range(len(df["Composition"])):
        if df["Type"][i] != last_type:
            last_type = df["Type"][i]
            j = 0
        comp = df["Composition"][i]
        if comp in LIT_REFS:
            sep = "\n" if comp in TWO_LINE_REFS else " "
            label = f"{comp}{sep}{LIT_REFS[comp]}"
        else:
            label = comp
        draw_uncertainty_point(ax, x_data[i], y_data[i], x_err[i], y_err[i],
                               markers[j], color[i], color[i],
                               label,
                               uncertainty_style=uncertainty_style)
        j += 1

    ax.scatter(pt_1_tc, pt_1_hardness, label="Pt-1 (This work)", s=1200, color="darkblue", marker="*")
    ax.scatter(re_3_tc, re_3_hardness, label="RE-3 (This work)", s=1200, color="orange", marker="*")
    ax.scatter(re_4_tc, re_4_hardness, label="RE-4 (This work)", s=1200, color="red", marker="*")

    x = [8.7, 12.6, 14.1, 13.8, 17.74, 22.7, 85.88, 114.49, 117, 140.4]
    y = [1327, 867, 735, 694, 560, 469, 55, 57.7, 63.5, 62.75]
    params, _ = curve_fit(model_func, x, y, p0=[1, 1])
    a_fit, b_fit = params
    x_fit = np.linspace(min(x), max(x), 100)
    ax.plot(x_fit, model_func(x_fit, a_fit, b_fit), '--', linewidth=4)

    ax.set_ylabel('Hardness [HV]')
    ax.set_xlabel(r'Total Thermal Conductivity at 300 K [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]')
    ax.set_title('Experiment Reference', pad=20)
    # two-column legend, family-balanced break: the four Mg-alloy entries head
    # the second column with their reference numbers on a second line; tighter
    # handle/column spacing keeps the ref-tagged legend inside the panel at
    # the original anchor position
    ax.legend(bbox_to_anchor=(0.20, 0.62, 0.5, 0.38), loc='upper left', ncol=2,
              fontsize=21, handletextpad=0.5, columnspacing=1.0)


def draw_prediction(ax, uncertainty_style="errorbar"):
    """Same rendering as the finalized task-1 prediction figure."""
    df = pd.read_csv(PRED_CSV)
    ax.grid(False)  # plain white background, no gridlines
    ax.set_axisbelow(True)
    ax.set_xlim(0, 150)

    counters = {t: 0 for t in TYPE_ORDER}
    for _, row in df.iterrows():
        t = alloy_type(row["Index"])
        m = MARKERS[counters[t] % len(MARKERS)]
        counters[t] += 1
        face = FACE_OVERRIDE.get(t, lighten(BASE_COLOR[t], 0.45))
        edge = lighten(BASE_COLOR[t], 0.10)
        draw_uncertainty_point(ax, row["LTC_pred_WmK"], row["BM_pred_GPa"],
                               row.get("LTC_pred_std", 0), row.get("BM_pred_std", 0),
                               m, face, edge, short_label(row["Composition"]),
                               uncertainty_style=uncertainty_style)

    present_types = [t for t in TYPE_ORDER if (df["Index"].map(alloy_type) == t).any()]
    type_handles = [plt.Line2D([], [], marker='o', ls='', markersize=14,
                               markerfacecolor=FACE_OVERRIDE.get(t, lighten(BASE_COLOR[t], 0.45)),
                               markeredgecolor=lighten(BASE_COLOR[t], 0.10),
                               markeredgewidth=1.4, label=TYPE_LABEL[t])
                    for t in present_types]
    leg1 = ax.legend(handles=type_handles, loc="lower right", fontsize=22,
                     frameon=True, framealpha=0.9, edgecolor="#cccccc",
                     title="Alloy family", title_fontsize=22)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=22,
              frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.0)

    ax.set_xlabel(r'Predicted Lattice Thermal Conductivity $\kappa_l^{pred}$ '
                  r'[$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]', labelpad=8)
    ax.set_ylabel(r'Predicted Bulk Modulus $B^{pred}$ [GPa]', labelpad=8)
    ax.set_title("Model Prediction", pad=20)


def render(fig_out, uncertainty_style="errorbar"):
    plt.rcParams.update({'font.size': 36})
    plt.rcParams['font.family'] = 'Arial'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15), dpi=600)
    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_color("#999999")
    draw_experiment(ax1, uncertainty_style=uncertainty_style)
    draw_prediction(ax2, uncertainty_style=uncertainty_style)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.11, wspace=0.22)
    fig.savefig(fig_out, dpi=600)
    plt.close(fig)
    print("saved:", fig_out)


def render_single(draw_fn, fig_out):
    """One panel per file, same 15x15 canvas as half of the combined figure."""
    plt.rcParams.update({'font.size': 36})
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(15, 15), dpi=600)
    for s in ax.spines.values():
        s.set_color("#999999")
    draw_fn(ax)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.93, bottom=0.11)
    fig.savefig(fig_out, dpi=600)
    plt.close(fig)
    print("saved:", fig_out)


def main():
    # final submission version: classic error bars only (CI-shading variant
    # was dropped), reference numbers annotated on the experiment panel
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    render(FIG_OUT)
    # the two panels also saved as separate figures
    render_single(draw_experiment,
                  FIG_OUT.replace(".png", "_experiment_only.png"))
    render_single(draw_prediction,
                  FIG_OUT.replace(".png", "_prediction_only.png"))


if __name__ == "__main__":
    main()
