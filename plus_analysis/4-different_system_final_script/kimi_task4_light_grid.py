# -*- coding: utf-8 -*-
"""
kimi_task4_light_grid.py
Task 4, light part (literature protocol): Sorkin et al. 2023 (Sci. Rep. 13,
22549) grid design for the Al-Be-Mg-Ti-Si quinary family (BCC solid solution).

Each element molar fraction in {5%, 10%, ..., 35%}, sum = 100%.
For every grid composition: 1 random-substitution model (BCC), predict B,
kappa_l, density. Then compute the MEAN trend of density / B / kappa_l /
specific stiffness (B/rho) as a function of Be molar fraction -- the same
statistic as Fig. 3c and Fig. 4d-f of Sorkin 2023.

Output: ../final_data/task4_light_grid_predictions.csv
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TASK1_SCRIPT_DIR = os.path.abspath(os.path.join(
    TASK4_DIR, "..", "1-reference_plus", "final_script"))
sys.path.insert(0, TASK1_SCRIPT_DIR)
os.chdir(TASK1_SCRIPT_DIR)

from kimi_inference_core import build_random_structure, build_datasets, \
    predict_from_data, map_value                                   # noqa: E402

ELEMENTS = ["Al", "Be", "Mg", "Ti", "Si"]
STEP = 5  # percent


def grid_compositions():
    """All (Al,Be,Mg,Ti,Si) in {5..35}% summing to 100%."""
    out = []
    units = list(range(1, 8))  # 5%..35% in 5% units
    for a in units:
        for b in units:
            for c in units:
                for d in units:
                    e = 20 - (a + b + c + d)
                    if e in units:
                        out.append([a, b, c, d, e])
    return out


def main():
    comps = grid_compositions()
    print(f"grid compositions: {len(comps)}")
    structures, metas = [], []
    for fr in comps:
        ratios = [f * STEP / 100.0 for f in fr]
        sd = zlib.crc32("_".join(map(str, fr)).encode()) % 100000
        s = build_random_structure(ELEMENTS, ratios, "bcc", seed=sd)
        structures.append(s)
        metas.append((fr, s.density))
    datasets = build_datasets(structures, n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")

    rows = []
    for (fr, dens), br, lr in zip(metas, bm_raw, ltc_raw):
        rows.append({"Al": fr[0], "Be": fr[1], "Mg": fr[2], "Ti": fr[3], "Si": fr[4],
                     "density": dens, "bm_raw": br, "ltc_raw": lr,
                     "bm_GPa": map_value(br, "bm", "bcc", 5),
                     "ltc_WmK": map_value(lr, "ltc", "bcc", 5)})
    df = pd.DataFrame(rows)
    df["B_over_rho"] = df["bm_GPa"] / df["density"]
    out_dir = os.path.join(TASK4_DIR, "final_data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "task4_light_grid_predictions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv)

    print("\n=== mean trend vs Be fraction (Sorkin protocol) ===")
    g = df.groupby("Be").agg(n=("Be", "size"),
                             density=("density", "mean"),
                             bm=("bm_GPa", "mean"), bm_std=("bm_GPa", "std"),
                             ltc=("ltc_WmK", "mean"),
                             B_over_rho=("B_over_rho", "mean"))
    g["density_rel"] = g["density"] / g["density"].iloc[0]
    g["bm_rel"] = g["bm"] / g["bm"].iloc[0]
    g["B_over_rho_rel"] = g["B_over_rho"] / g["B_over_rho"].iloc[0]
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
