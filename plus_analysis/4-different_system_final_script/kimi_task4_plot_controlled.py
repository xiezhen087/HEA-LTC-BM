# -*- coding: utf-8 -*-
"""
kimi_task4_plot_controlled.py — Task 4 (redesigned) comparison figure, 2x3 layout.

  (a) refractory, mechanical: model dB% (Tseng-2019 mirror series; W from the
      Shkodich/task-1 series) vs literature d(YS/HV)% (Tseng 2019 Table 3;
      Shkodich 2022).
  (b) refractory, thermal: model d(kappa_l)% vs literature d(TC)% for Mo/W
      (Shkodich 2022 series).
  (c) low_act: model B vs x in VCrFeTaxWx vs Zhang 2018 hardness fit
      (both normalized at x = 0.1).
  (d) light, density vs Be fraction: model grid cloud + mean line + literature
      star points READ FROM Sorkin 2023 tables (best-composition values).
  (e) light, modulus vs Be fraction: model grid cloud + mean line + Sorkin
      Young's modulus star points (same source).
  (f) notes: literature sources and interpretation remarks.

Model = bars/lines/clouds + error bars; literature = scatter stars with values
taken verbatim from the cited papers (see final_data/kimi_task4_literature_refs.csv).
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
FIG_DIR = os.path.join(TASK4_DIR, "final_fig")
TASK1_CSV = os.path.join(TASK4_DIR, "..", "1-reference_plus", "final_data",
                         "predicted_bulk_modulus_and_ltc_random.csv")

plt.rcParams.update({"font.family": "Arial", "font.size": 15})
MODEL_COLOR = "#7fa8c9"
MODEL_EDGE = "#3d6e9e"
LIT_COLOR = "#c0392b"


def pct(a, b):
    return 100.0 * (a - b) / b


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summ = pd.read_csv(os.path.join(DATA_DIR, "task4_controlled_summary.csv"))
    task1 = pd.read_csv(TASK1_CSV)
    grid = pd.read_csv(os.path.join(DATA_DIR, "task4_light_grid_predictions.csv"))
    grid["B_over_rho"] = grid["bm_GPa"] / grid["density"]

    # ---------- model values: refractory Tseng mirror ----------
    ref = summ[summ["system"] == "refractory"].set_index("case")
    base_bm, base_ltc = ref.loc["HfMoNbTaTiZr", "bm"], ref.loc["HfMoNbTaTiZr", "ltc"]
    model_bm_tseng = {
        "Mo": pct(base_bm, ref.loc["HfNbTaTiZr(-Mo)", "bm"]),
        "Nb": pct(base_bm, ref.loc["HfMoTaTiZr(-Nb)", "bm"]),
        "Ta": pct(base_bm, ref.loc["HfMoNbTiZr(-Ta)", "bm"]),
        "Ti": pct(base_bm, ref.loc["HfMoNbTaZr(-Ti)", "bm"]),
        "Zr": pct(base_bm, ref.loc["HfMoNbTaTi(-Zr)", "bm"]),
    }
    t1 = task1.set_index("Index")
    model_bm_shk = {
        "Mo": pct(t1.loc["re-3", "BM_pred_GPa"], t1.loc["re-2", "BM_pred_GPa"]),
        "W": pct(t1.loc["re-4", "BM_pred_GPa"], t1.loc["re-2", "BM_pred_GPa"]),
    }
    model_ltc_shk = {
        "Mo": pct(t1.loc["re-3", "LTC_pred_WmK"], t1.loc["re-2", "LTC_pred_WmK"]),
        "W": pct(t1.loc["re-4", "LTC_pred_WmK"], t1.loc["re-2", "LTC_pred_WmK"]),
    }
    lit_bm = {"Mo": 62.8, "Ta": 11.9, "Nb": -5.5, "Ti": -0.8, "Zr": 10.5,
              "W": 103.2}   # Tseng 2019 Table 3 (YS); W: Shkodich 2022 (HV)
    lit_ltc = {"Mo": 29.9, "W": -10.3}  # Shkodich 2022 (total TC)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=400)
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    # ---------------- (a) refractory mechanical ----------------
    elements = ["Mo", "W", "Ta", "Nb", "Ti", "Zr"]
    model_bm = [model_bm_tseng["Mo"], model_bm_shk["W"], model_bm_tseng["Ta"],
                model_bm_tseng["Nb"], model_bm_tseng["Ti"], model_bm_tseng["Zr"]]
    xs = np.arange(len(elements))
    ax_a.bar(xs, model_bm, width=0.5, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
             label="This work (model, B)", zorder=2)
    for i, e in enumerate(elements):
        ax_a.scatter(i + 0.30, lit_bm[e], marker="*", s=380, color=LIT_COLOR,
                     edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax_a.scatter([], [], marker="*", s=380, color=LIT_COLOR, edgecolors="#7b241c",
                 label="Literature (yield strength / HV)")
    ax_a.axhline(0, color="#666666", lw=1.0)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(elements)
    ax_a.set_ylabel("Change with element present [%]")
    ax_a.set_title("(a) Refractory: mechanical property change", pad=10)
    ax_a.legend(fontsize=11, frameon=False, loc="upper right")
    ax_a.annotate("*Mo/Ta/Nb/Ti/Zr: Tseng mirror (as-cast, YS);\n"
                  "W: TaTiNbZr+W series (HEBM+SPS, HV)",
                  xy=(0.02, 0.02), xycoords="axes fraction", fontsize=10,
                  va="bottom", color="#555555")

    # ---------------- (b) refractory thermal ----------------
    xs2 = np.arange(2)
    vals = [model_ltc_shk["Mo"], model_ltc_shk["W"]]
    lit_vals = [lit_ltc["Mo"], lit_ltc["W"]]
    ax_b.bar(xs2, vals, width=0.42, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
             label=r"This work (model, $\kappa_l$)", zorder=2)
    for i, v in enumerate(lit_vals):
        ax_b.scatter(i + 0.26, v, marker="*", s=380, color=LIT_COLOR,
                     edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax_b.scatter([], [], marker="*", s=380, color=LIT_COLOR, edgecolors="#7b241c",
                 label="Literature (total TC, Shkodich 2022)")
    ax_b.axhline(0, color="#666666", lw=1.0)
    ax_b.set_xticks(xs2)
    ax_b.set_xticklabels(["+Mo\n(TaTiNbZr base)", "+W\n(TaTiNbZr base)"])
    ax_b.set_ylabel("Change with element present [%]")
    ax_b.set_title("(b) Refractory: thermal conductivity change", pad=10)
    ax_b.legend(fontsize=11, frameon=False, loc="upper left")

    # ---------------- (c) low_act: VCrFeTaxWx ----------------
    low = summ[summ["system"] == "low_act"].set_index("case")
    xs_d = [0.0, 0.1, 0.2, 0.3, 0.4]
    bm_vals = [low.loc[f"VCrFeTaW_x{x:.1f}", "bm"] for x in xs_d]
    bm_errs = [low.loc[f"VCrFeTaW_x{x:.1f}", "bm_std"] for x in xs_d]
    bm_rel = np.array(bm_vals) / bm_vals[1]  # normalize at x=0.1
    bm_rel_err = np.array(bm_errs) / bm_vals[1]
    hv_fit = lambda x: 553.1 + 609.2 * x  # Zhang 2018 linear fit (R=0.942)
    hv_rel = np.array([hv_fit(x) for x in xs_d]) / hv_fit(0.1)
    ax_c.errorbar(xs_d, bm_rel, yerr=bm_rel_err, fmt="o-", color="#1f77b4",
                  lw=2.4, ms=8, capsize=4, label="This work (model, B)")
    ax_c.plot(xs_d, hv_rel, "s--", color=LIT_COLOR, lw=2.2, ms=7,
              label="Literature hardness fit (Zhang 2018)")
    ax_c.scatter([1.0], [(553.1 + 609.2 * 1.0) / hv_fit(0.1)], marker="*", s=420,
                 color=LIT_COLOR, edgecolors="#7b241c", zorder=3,
                 label="Literature x = 1.0 (1135 HV)")
    ax_c.set_xlabel("x in VCrFeTa$_x$W$_x$")
    ax_c.set_ylabel("Relative to x = 0.1")
    ax_c.set_title("(c) Low-activation alloys: Ta + W additions", pad=10)
    ax_c.legend(fontsize=11, frameon=False, loc="upper left")

    # ---------------- (d) light: density vs Be ----------------
    be_x = grid["Be"] * 5  # stored in 5% units
    ax_d.scatter(be_x, grid["density"], s=10, color="#a9d5a2", alpha=0.35,
               zorder=1, label="Model grid points")
    g = grid.groupby(grid["Be"] * 5).agg(density=("density", "mean"),
                                         density_sd=("density", "std"),
                                         n=("Be", "size"))
    ax_d.errorbar(g.index, g["density"], yerr=g["density_sd"] / np.sqrt(g["n"]),
                  fmt="o-", color="#1e7e1e", lw=2.6, ms=8, capsize=4, zorder=2,
                  label="Model (grid mean)")
    # Sorkin 2023 AlBeMgTiSi best-composition values (verbatim from tables)
    sorkin_rho = [(15, 2.43, "most stable"), (35, 2.43, "lightest")]
    for be, rho, tag in sorkin_rho:
        ax_d.scatter(be, rho, marker="*", s=460, color=LIT_COLOR,
                     edgecolors="#7b241c", linewidths=1.2, zorder=3)
    ax_d.scatter([], [], marker="*", s=460, color=LIT_COLOR, edgecolors="#7b241c",
                 label="Sorkin 2023 (DFT, best comps.)")
    ax_d.set_xlabel("Be molar fraction [%]")
    ax_d.set_ylabel("Density [g/cm$^3$]")
    ax_d.set_title("(d) Light alloys: density vs Be content", pad=10)
    ax_d.legend(fontsize=11, frameon=False, loc="upper right")

    # ---------------- (e) light: modulus vs Be ----------------
    g2 = grid.groupby(grid["Be"] * 5).agg(bm=("bm_GPa", "mean"),
                                          bm_sd=("bm_GPa", "std"),
                                          n=("Be", "size"))
    ax_e.scatter(be_x, grid["bm_GPa"], s=10, color="#a9c7de", alpha=0.35,
                 zorder=1, label="Model grid points")
    ax_e.errorbar(g2.index, g2["bm"], yerr=g2["bm_sd"] / np.sqrt(g2["n"]),
                  fmt="s-", color="#1f5fa8", lw=2.6, ms=8, capsize=4, zorder=2,
                  label="Model (grid mean, B)")
    sorkin_E = [(15, 135.77), (15, 222.85), (35, 135.77)]  # Young's modulus
    for be, E in sorkin_E:
        ax_e.scatter(be, E, marker="*", s=460, color=LIT_COLOR,
                     edgecolors="#7b241c", linewidths=1.2, zorder=3)
    ax_e.scatter([], [], marker="*", s=460, color=LIT_COLOR, edgecolors="#7b241c",
                 label="Sorkin 2023 (DFT, Young's mod.)")
    ax_e.set_xlabel("Be molar fraction [%]")
    ax_e.set_ylabel("Modulus [GPa]")
    ax_e.set_title("(e) Light alloys: modulus vs Be content", pad=10)
    ax_e.legend(fontsize=11, frameon=False, loc="upper right")

    # ---------------- (f) notes ----------------
    ax_f.axis("off")
    notes = (
        "Literature sources\n"
        "  Tseng et al., Entropy 21 (2019) 15, doi:10.3390/e21010015 (Table 3)\n"
        "  Shkodich et al., JAC 893 (2022) 162030, doi:10.1016/j.jallcom.2021.162030\n"
        "  Zhang et al., Entropy 20 (2018) 951, doi:10.3390/e20120951\n"
        "  Sorkin et al., Sci. Rep. 13 (2023) 22549, doi:10.1038/s41598-023-49258-z\n"
        "     (stars in d/e are their best-composition table values,\n"
        "      i.e. extreme optima, not grid means)\n\n"
        "Interpretation notes\n"
        "  - Model outputs: intrinsic elastic/thermal properties of ideal\n"
        "    single-phase solid solutions (B, kappa_l).\n"
        "  - Literature values: mostly plastic properties (YS, hardness) of\n"
        "    processed alloys; percent changes of B are intrinsically smaller.\n"
        "  - Mo/Ta/Zr act via the bonding-strength channel -> model captures.\n"
        "  - Ti/Nb literature effects concern ductility/toughness (plasticity\n"
        "    channel) -> not visible to B/kappa_l models.\n"
        "  - W and TC literature points are HEBM+SPS nanocrystalline\n"
        "    (processing-dominated), not directly comparable."
    )
    ax_f.text(0.0, 1.0, notes, fontsize=12.5, va="top", ha="left",
              color="#333333", linespacing=1.5)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.grid(axis="y", ls=":", color="#dddddd", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Element add/remove effects: controlled comparison vs literature",
                 fontsize=21, y=0.99)
    fig_out = os.path.join(FIG_DIR, "fig_controlled_element_effects.png")
    fig.savefig(fig_out, dpi=600, bbox_inches="tight", pad_inches=0.3)
    print("saved:", fig_out)


if __name__ == "__main__":
    main()
