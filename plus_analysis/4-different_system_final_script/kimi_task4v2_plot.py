# -*- coding: utf-8 -*-
"""
kimi_task4v2_plot.py — Task 4 (v2) figures.
Four separate figures + one 2x2 combined figure (no panel-letter prefixes in
titles; lettering is added at manuscript assembly).

  fig_refractory_mechanical.png: model dB% (bars + error bars, 10 seeds)
      for Mo/Ta/Zr vs literature dYS% (stars, Tseng 2019 Table 3).
  fig_alxcocrfeni_ltc.png: lattice thermal conductivity vs Al content x in
      AlxCoCrFeNi; model (dashed, phase-fraction weighted, 5 seeds, error
      bars) vs literature (solid + range error bars, Chou 2009 Table 2).
  fig_light_be_specific_stiffness.png: specific stiffness vs Be fraction
      for AlBeMgTiLi/Si/Cu; literature E/rho (solid, digitized by the user
      from Sorkin 2023) vs model B/rho (dashed, grid means). Units MN*m/kg.
  fig_lowact_taw.png: model B vs x in VCrFeTaxWx vs Zhang 2018 hardness
      fit (both normalized at x = 0.1).
  combined: fig_task4_combined.png (2x2)

Units note (c): B[GPa] / rho[g/cm3] = MN*m/kg (numerically identical).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(TASK4_DIR, "final_data")
FIG_DIR = os.path.abspath(os.path.join(TASK4_DIR, "..", "..", "final_fig", "reference_figs"))

# final reference numbers (final_fig/reference-index.csv):
#   Tseng 2019 -> [80], Chou 2009 -> [81], Sorkin 2023 -> [82], Zhang 2018 -> [83]

plt.rcParams.update({"font.family": "Arial", "font.size": 16})
MODEL_COLOR = "#7fa8c9"
MODEL_EDGE = "#3d6e9e"
LIT_COLOR = "#c0392b"
FAM_COLOR = {"AlBeMgTiLi": "#2ca02c", "AlBeMgTiSi": "#1f77b4", "AlBeMgTiCu": "#cc660b"}


def pct_err(mu_w, sd_w, n_w, mu_wo, sd_wo, n_wo):
    sem_w = sd_w / np.sqrt(max(n_w, 1))
    sem_wo = sd_wo / np.sqrt(max(n_wo, 1))
    r = mu_w / mu_wo
    return 100.0 * r * np.sqrt((sem_w / mu_w)**2 + (sem_wo / mu_wo)**2)


def style_ax(ax):
    ax.grid(False)  # plain white background, no gridlines
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_a(ax):
    df = pd.read_csv(os.path.join(DATA_DIR, "task4v2_refractory.csv"))
    g = df.groupby("case")["bm_GPa"].agg(["mean", "std", "size"])
    base_mu, base_sd, base_n = g.loc["HfMoNbTaTiZr", ["mean", "std", "size"]]
    variants = {"Mo": "HfNbTaTiZr(-Mo)", "Ta": "HfMoNbTiZr(-Ta)", "Zr": "HfMoNbTaTi(-Zr)"}
    lit = {"Mo": 62.8, "Ta": 11.9, "Zr": 10.5}
    vals, errs = [], []
    for e, case in variants.items():
        mu, sd, n = g.loc[case, ["mean", "std", "size"]]
        vals.append(100.0 * (base_mu - mu) / mu)
        errs.append(pct_err(base_mu, base_sd, base_n, mu, sd, n))
    xs = np.arange(3)
    ax.bar(xs, vals, yerr=errs, width=0.5, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
           capsize=6, error_kw=dict(elinewidth=1.8, capthick=1.8),
           label="This work (model, B)", zorder=2)
    for i, e in enumerate(variants):
        ax.scatter(i + 0.30, lit[e], marker="*", s=420, color=LIT_COLOR,
                   edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax.scatter([], [], marker="*", s=420, color=LIT_COLOR, edgecolors="#7b241c",
               label="Literature (YS, Tseng 2019 [80])")
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(variants.keys()))
    ax.set_ylabel("Change with element present [%]")
    ax.set_title("Refractory: mechanical property change", pad=10)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    style_ax(ax)


def panel_b(ax):
    df = pd.read_csv(os.path.join(DATA_DIR, "task4v2_alxcocrfeni.csv"))
    ax.errorbar(df["x"], df["lit_kph_mid"], yerr=df["lit_kph_err"], fmt="o-",
                color=LIT_COLOR, lw=2.4, ms=8, capsize=4, zorder=3,
                label="Literature (Chou 2009 [81], Table 2)")
    ax.errorbar(df["x"], df["ltc_weighted"], yerr=df["ltc_std"], fmt="s--",
                color="#1f5fa8", lw=2.4, ms=8, capsize=4, zorder=2,
                label="This work (model, phase-weighted)")
    ax.set_xlabel("x in Al$_x$CoCrFeNi")
    ax.set_ylabel(r"Lattice thermal conductivity [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]")
    ax.set_title(r"Al$_x$CoCrFeNi: lattice thermal conductivity", pad=10)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    style_ax(ax)


def panel_c(ax):
    ref = pd.read_csv(os.path.join(TASK4_DIR, "reference_data.csv"), encoding="gbk")
    ref = ref.iloc[1:]  # first data row holds the family names
    for c in ref.columns:  # digitized values read as text -> force numeric
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
    lit_map = {"AlBeMgTiLi": ("Unnamed: 0", "E/ρ MNm/kg"),
               "AlBeMgTiSi": ("Unnamed: 3", "E/ρ MNm/kg.1"),
               "AlBeMgTiCu": ("Unnamed: 6", "E/ρ MNm/kg.2")}
    for fam, color in FAM_COLOR.items():
        xc, yc = lit_map[fam]
        sub = ref[[xc, yc]].dropna().sort_values(xc)
        ax.plot(sub[xc], sub[yc], "-", color=color, lw=2.6, zorder=3,
                label=f"{fam} literature (Sorkin 2023 [82], E/$\\rho$)")
        grid = pd.read_csv(os.path.join(DATA_DIR, f"task4v2_light_grid_{fam}.csv"))
        grid["B_over_rho"] = grid["bm_GPa"] / grid["density"]
        gm = grid.groupby(grid["case"].str.split("-").str[1].astype(int) * 5)["B_over_rho"].mean()
        ax.plot(gm.index, gm.values, "--", color=color, lw=2.6, zorder=2,
                label=f"{fam} this work (model, B/$\\rho$)")
    ax.set_xlabel("Be molar fraction [%]")
    ax.set_ylabel("Specific stiffness [MN·m/kg]")
    ax.set_title("Light alloys: specific stiffness vs Be content", pad=10)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    style_ax(ax)


def panel_d(ax):
    df = pd.read_csv(os.path.join(DATA_DIR, "task4v2_lowact.csv"))
    g = df.groupby("case")["bm_GPa"].agg(["mean", "std"])
    xs_d = [0.0, 0.1, 0.2, 0.3, 0.4]
    bm_vals = [g.loc[f"VCrFeTaW_x{x:.1f}", "mean"] for x in xs_d]
    bm_errs = [g.loc[f"VCrFeTaW_x{x:.1f}", "std"] for x in xs_d]
    bm_rel = np.array(bm_vals) / bm_vals[1]
    bm_rel_err = np.array(bm_errs) / bm_vals[1]
    hv_fit = lambda x: 553.1 + 609.2 * x
    hv_rel = np.array([hv_fit(x) for x in xs_d]) / hv_fit(0.1)
    ax.errorbar(xs_d, bm_rel, yerr=bm_rel_err, fmt="o--", color="#1f5fa8",
                lw=2.4, ms=8, capsize=4, label="This work (model, B)")
    ax.plot(xs_d, hv_rel, "s-", color=LIT_COLOR, lw=2.4, ms=7,
            label="Literature hardness fit (Zhang 2018 [83])")
    ax.scatter([1.0], [(553.1 + 609.2) / hv_fit(0.1)], marker="*", s=420,
               color=LIT_COLOR, edgecolors="#7b241c", zorder=3,
               label="Literature x = 1.0 (1135 HV)")
    ax.set_xlabel("x in VCrFeTa$_x$W$_x$")
    ax.set_ylabel("Relative to x = 0.1")
    ax.set_title("Low-activation alloys: Ta + W additions", pad=10)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    style_ax(ax)


PANELS = [panel_a, panel_b, panel_c, panel_d]
FIG_NAMES = ["fig_refractory_mechanical.png", "fig_alxcocrfeni_ltc.png",
             "fig_light_be_specific_stiffness.png", "fig_lowact_taw.png"]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    # four separate figures
    for fn, name in zip(PANELS, FIG_NAMES):
        fig, ax = plt.subplots(figsize=(8.5, 7), dpi=500)
        fn(ax)
        out = os.path.join(FIG_DIR, name)
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print("saved:", out)
    # combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.5), dpi=500)
    for fn, ax in zip(PANELS, axes.flat):
        fn(ax)
    fig.suptitle("Element add/remove effects: model vs literature", fontsize=21)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "fig_task4_combined.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.3)
    print("saved:", out)


if __name__ == "__main__":
    main()
