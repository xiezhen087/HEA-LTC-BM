# -*- coding: utf-8 -*-
"""
kimi_task1_plot.py — beautified rendering of the task-1 prediction figure.

Task division:
  - alloy modeling and model prediction: (author)
  - plotting, including the CI-shading variant: Kimi

Reads ../final_data/predicted_bulk_modulus_and_ltc_random.csv (produced by
kimi_task1_build_predict.py) and draws the predicted bulk modulus vs predicted
lattice thermal conductivity scatter, styled after reference_painting.py
(same type hues, lighter shades; error bars from multi-seed statistics).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV_IN = os.path.join(TASK1_DIR, "final_data", "predicted_bulk_modulus_and_ltc_random.csv")
FIG_OUT = os.path.join(TASK1_DIR, "final_fig", "fig_predicted_B_vs_ltc_random.png")

# dilute Mg alloys (~98 at.% Mg) are out of the model's validated domain
# (pure metals excluded from training; HCP subspace unvalidated) -> excluded
# from the main figure, kept in the CSV with an OOD note
EXCLUDE_FROM_FIG = ["mg-1", "mg-2", "mg-3", "mg-4"]

# type hues follow the experimental reference figure (Fig. 8 of the manuscript)
BASE_COLOR = {"3d": "#2B3E51", "al_3d": "#FABE28", "cu_3d": "#1A1A1A",
              "mg": "#8A2BE2", "tic": "#FF5A5F", "refractory": "#76D7EA"}
# explicit face colors where the lightened version reads too pale/gray
FACE_OVERRIDE = {"3d": "#66CDAA"}  # medium aquamarine
TYPE_LABEL = {"3d": "3d transition-metal HEAs", "al_3d": "Al-containing 3d HEAs",
              "cu_3d": "Cu-containing 3d HEAs", "mg": "Multi-principal Mg alloys",
              "tic": "Refractory-TiC composites", "refractory": "Refractory HEAs"}
TYPE_ORDER = ["3d", "al_3d", "cu_3d", "mg", "tic", "refractory"]
MARKERS = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h']


def alloy_type(index):
    if index.startswith("re-tic"):
        return "tic"
    if index.startswith("re-"):
        return "refractory"
    if index.startswith("3d"):
        return "3d"
    if index.startswith("al_3d"):
        return "al_3d"
    if index.startswith("cu-3d"):
        return "cu_3d"
    if index.startswith("mg"):
        return "mg"
    return "other"


def lighten(hex_color, blend=0.45):
    """Lighten in HLS space: keep the hue (and roughly the saturation) so that
    e.g. dark navy #2B3E51 becomes a clear steel blue instead of gray."""
    import colorsys
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    ll = ll + (1.0 - ll) * blend
    ss = min(1.0, ss * (1.0 + 0.4 * blend))  # slight saturation boost keeps hue vivid
    r, g, b = colorsys.hls_to_rgb(hh, ll, ss)
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in (r, g, b))


def short_label(comp):
    m = re.search(r"\((T22[^)]*)\)", comp)
    if m:
        return m.group(1)
    return comp


def draw_uncertainty_point(ax, x, y, xerr, yerr, marker, face, edge, label,
                           uncertainty_style="errorbar"):
    """Draw one point with either classic error bars or a 2D CI shading."""
    if uncertainty_style == "ci_shading":
        if xerr > 0 and yerr > 0:
            rect = Rectangle((x - xerr, y - yerr), 2 * xerr, 2 * yerr,
                             facecolor=face, edgecolor=edge, linewidth=1.4,
                             alpha=0.18, zorder=2)
            ax.add_patch(rect)
        ax.scatter([x], [y], marker=marker, s=220, facecolor=face,
                   edgecolor=edge, linewidth=1.4, alpha=0.9, zorder=3,
                   label=label)
    else:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt=marker, markersize=14, markerfacecolor=face,
                    markeredgecolor=edge, markeredgewidth=1.4, ecolor=edge,
                    elinewidth=2.5, capsize=6, capthick=2.5, alpha=0.7,
                    zorder=3, label=label)


def render(df, fig_out, uncertainty_style="errorbar"):
    plt.rcParams.update({'font.size': 36})
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(15, 15), dpi=600)
    for s in ax.spines.values():
        s.set_color("#999999")
    ax.grid(ls=":", color="#e4e4e4", lw=0.9, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 150)  # requested: thermal-conductivity axis 0-150

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

    # type color legend (inside, lower right)
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

    # composition legend (inside, upper right)
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=22,
              frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.0)

    ax.set_xlabel(r'Predicted Lattice Thermal Conductivity $\kappa_l^{pred}$ '
                  r'[$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]', labelpad=8)
    ax.set_ylabel(r'Predicted Bulk Modulus $B^{pred}$ [GPa]', labelpad=8)
    ax.set_title("Model Prediction", pad=20)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.93, bottom=0.11)
    fig.savefig(fig_out, dpi=600)  # keep the exact 15x15 canvas like the original
    plt.close(fig)
    print("saved:", fig_out)


def main():
    df = pd.read_csv(CSV_IN)
    # main figure: all alloys (rolled-back version)
    render(df, FIG_OUT)
    render(df, FIG_OUT.replace(".png", "_ci_shading.png"),
           uncertainty_style="ci_shading")
    # archive version without the out-of-domain dilute Mg alloys
    render(df[~df["Index"].isin(EXCLUDE_FROM_FIG)],
           FIG_OUT.replace(".png", "_without_mg.png"))
    render(df[~df["Index"].isin(EXCLUDE_FROM_FIG)],
           FIG_OUT.replace(".png", "_without_mg_ci_shading.png"),
           uncertainty_style="ci_shading")


if __name__ == "__main__":
    main()
