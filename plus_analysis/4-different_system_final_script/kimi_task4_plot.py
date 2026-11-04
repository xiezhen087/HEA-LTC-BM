# -*- coding: utf-8 -*-
"""
kimi_task4_plot.py — Task 4 comparison figures.
Model side: bar + error bar (percent change of the property when the element
is present vs absent; error from group statistics).
Literature side: scatter + error bar (percent change computed from reported
values; sparse points allowed).

Panels:
  (a) light:      effect of Be on density and bulk modulus
  (b) low_act:    effect of micro elements (Ta/Ti/Zr/Si ...) on hardness
                  (model bulk modulus as hardness proxy)
  (c) refractory: effect of Mo / W / Ti presence on bulk modulus
                  (literature: RT compressive yield strength, Tseng 2019)

Inputs (../final_data/):
  task4_predictions.csv              (from kimi_task4_predict.py)
  task4_refractory_element_effect.csv (from kimi_task4_refractory_analysis.py)
  kimi_task4_literature.csv          (manually curated literature values)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(TASK4_DIR, "final_data")
FIG_DIR = os.path.join(TASK4_DIR, "final_fig")

plt.rcParams.update({"font.family": "Arial", "font.size": 16})
MODEL_COLOR = "#7fa8c9"
MODEL_EDGE = "#3d6e9e"
LIT_COLOR = "#c0392b"


def pct_change(with_v, without_v):
    return 100.0 * (with_v - without_v) / without_v


def pct_err(mu_w, sd_w, n_w, mu_wo, sd_wo, n_wo):
    """SEM of (mu_w - mu_wo)/mu_wo in percent (independent groups)."""
    sem_w = sd_w / np.sqrt(max(n_w, 1))
    sem_wo = sd_wo / np.sqrt(max(n_wo, 1))
    r = mu_w / mu_wo
    var = r**2 * ((sem_w / mu_w)**2 + (sem_wo / mu_wo)**2)
    return 100.0 * np.sqrt(var)


def model_pair(df_sub, flag_col, value_col):
    gw = df_sub[df_sub[flag_col]]
    go = df_sub[~df_sub[flag_col]]
    pct = pct_change(gw[value_col].mean(), go[value_col].mean())
    err = pct_err(gw[value_col].mean(), gw[value_col].std(), len(gw),
                  go[value_col].mean(), go[value_col].std(), len(go))
    return pct, err, len(gw), len(go)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pred = pd.read_csv(os.path.join(DATA_DIR, "task4_predictions.csv"))
    ref = pd.read_csv(os.path.join(DATA_DIR, "task4_refractory_element_effect.csv"))
    lit = pd.read_csv(os.path.join(DATA_DIR, "kimi_task4_literature.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), dpi=500)

    # ---------------- (a) light: Be effect on density & B ----------------
    ax = axes[0]
    light = pred[pred["system"] == "light"]
    d_rho = model_pair(light, "has_Be", "density")
    d_bm = model_pair(light, "has_Be", "bm_GPa")
    xs = [0, 1]
    labels = ["Density", "Bulk modulus"]
    vals = [d_rho[0], d_bm[0]]
    errs = [d_rho[1], d_bm[1]]
    ax.bar(xs, vals, yerr=errs, width=0.45, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
           capsize=5, error_kw=dict(elinewidth=1.6, capthick=1.6), label="This work (model)",
           zorder=2)
    # literature: pure Be vs pure Al; BeAl vs Al
    lit_light = lit[lit["group"] == "light_Be"]
    for _, r in lit_light.iterrows():
        x = 0 if r["property"] == "density" else 1
        ax.scatter(x + 0.28, r["pct_change"], marker="*", s=420, color=LIT_COLOR,
                   edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax.scatter([], [], marker="*", s=420, color=LIT_COLOR, edgecolors="#7b241c",
               label="Literature")
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(-45, 88)
    ax.set_ylabel("Change with Be present [%]")
    ax.set_title("(a) Light alloys: effect of Be", pad=12)
    ax.legend(fontsize=11, frameon=False, loc="upper left")

    # ---------------- (b) low_act: micro elements -> hardness ----------------
    ax = axes[1]
    low = pred[pred["system"] == "low_act"]
    d_micro = model_pair(low, "has_micro", "bm_GPa")
    ax.bar([0], [d_micro[0]], yerr=[d_micro[1]], width=0.4, color=MODEL_COLOR,
           edgecolor=MODEL_EDGE, capsize=5,
           error_kw=dict(elinewidth=1.6, capthick=1.6), label="This work (model, B as proxy)",
           zorder=2)
    lit_low = lit[(lit["group"] == "low_act_micro") & lit["pct_change"].notna()]
    for _, r in lit_low.iterrows():
        ax.scatter(0.28, r["pct_change"], marker="*", s=420, color=LIT_COLOR,
                   edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xlim(-0.55, 0.6)
    ax.set_ylim(-25, 125)
    ax.set_xticks([0])
    ax.set_xticklabels(["Micro elements\n(Ta/Ti/Zr/Si ...)"])
    ax.set_ylabel("Change with micro elements [%]")
    ax.set_title("(b) Low-activation alloys: micro additions", pad=12)
    ax.legend(fontsize=11, frameon=False, loc="upper left")

    # ---------------- (c) refractory: Mo / W / Ti presence ----------------
    ax = axes[2]
    xs = np.arange(3)
    vals = ref.set_index("element").loc[["Mo", "W", "Ti"], "d_bm_pct"].values
    errs = ref.set_index("element").loc[["Mo", "W", "Ti"], "d_bm_pct_err"].values
    ax.bar(xs, vals, yerr=errs, width=0.45, color=MODEL_COLOR, edgecolor=MODEL_EDGE,
           capsize=5, error_kw=dict(elinewidth=1.6, capthick=1.6),
           label="This work (model, B)", zorder=2)
    lit_ref = {"Mo": 62.8, "Ti": -0.8}  # Tseng 2019 Table 3, RT compressive YS
    for i, e in enumerate(["Mo", "W", "Ti"]):
        if e in lit_ref:
            ax.scatter(i + 0.27, lit_ref[e], marker="*", s=420, color=LIT_COLOR,
                       edgecolors="#7b241c", linewidths=1.0, zorder=3)
    ax.scatter([], [], marker="*", s=420, color=LIT_COLOR, edgecolors="#7b241c",
               label="Literature (yield strength)")
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Mo", "W", "Ti"])
    ax.set_ylim(-18, 78)
    ax.set_ylabel("Change with element present [%]")
    ax.set_title("(c) Refractory alloys: Mo / W / Ti", pad=12)
    ax.legend(fontsize=11, frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", ls=":", color="#dddddd", lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Element-presence effects: model vs literature", fontsize=20, y=1.0)
    fig.text(0.5, -0.06,
             "Literature points — (a) pure Be vs Al (elemental data), BeAl alloy vs Al (datasheet); "
             "(b) VCrFeTa$_x$W$_x$ x: 0.1→1, hardness (Zhang et al., Entropy 2018); "
             "(c) Hf-Mo-Nb-Ta-Ti-Zr series RT yield strength (Tseng et al., Entropy 2019, Table 3; W not reported).",
             ha="center", fontsize=11, color="#555555")
    fig_out = os.path.join(FIG_DIR, "fig_element_presence_effects.png")
    fig.savefig(fig_out, dpi=600, bbox_inches="tight", pad_inches=0.25)
    print("saved:", fig_out)

    # console summary for the report
    print(f"light  Be: density {d_rho[0]:+.1f}% (n={d_rho[2]}/{d_rho[3]}), "
          f"B {d_bm[0]:+.1f}% +/- {d_bm[1]:.1f}")
    print(f"lowact micro: B {d_micro[0]:+.1f}% +/- {d_micro[1]:.1f} "
          f"(n={d_micro[2]}/{d_micro[3]})")
    print(ref[["element", "d_bm_pct", "d_bm_pct_err", "d_ltc_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
