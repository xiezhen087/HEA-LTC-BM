# -*- coding: utf-8 -*-
"""
kimi_task4_controlled_predict.py
Task 4 (redesigned): controlled element add/remove comparison.

Composition sets follow the LITERATURE protocols (per project instruction):
  refractory (BCC): Tseng 2019 mirror -- equiatomic HfMoNbTaTiZr and the five
      minus-one variants (same alloys as Entropy 21 (2019) 15, Table 3).
  light (BCC): Sorkin 2023 protocol -- Al-Be-Mg-Ti-Si family, Be molar
      fraction varied 0 -> 0.35, other elements scaled proportionally to the
      most stable composition [Al 0.31, Be 0.15, Mg 0.14, Ti 0.05, Si 0.35]
      (Sci. Rep. 13 (2023) 22549, Table 1).
  low_act (BCC): Zhang 2018 mirror -- VCrFe + VCrFeTaxWx (x = 0.1..0.4)
      (Entropy 20 (2018) 951).

5 random seeds per composition; B and kappa_l predicted, density from the
structure. Output: ../final_data/task4_controlled_predictions.csv
"""
import os
import sys

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

N_SEEDS = 5


def equiatomic(elements):
    r = [1.0 / len(elements)] * len(elements)
    return elements, r


def build_cases():
    cases = []
    # ---- refractory: Tseng 2019 series (BCC) ----
    base = ["Hf", "Mo", "Nb", "Ta", "Ti", "Zr"]
    variants = {
        "HfMoNbTaTiZr": base,
        "HfNbTaTiZr(-Mo)": [e for e in base if e != "Mo"],
        "HfMoTaTiZr(-Nb)": [e for e in base if e != "Nb"],
        "HfMoNbTiZr(-Ta)": [e for e in base if e != "Ta"],
        "HfMoNbTaZr(-Ti)": [e for e in base if e != "Ti"],
        "HfMoNbTaTi(-Zr)": [e for e in base if e != "Zr"],
    }
    for name, els in variants.items():
        elements, ratios = equiatomic(els)
        cases.append(("refractory", name, elements, ratios, "bcc"))

    # ---- light: Sorkin 2023 Al-Be-Mg-Ti-Si, Be fraction 0 -> 0.35 (BCC) ----
    others = [("Al", 0.31), ("Mg", 0.14), ("Ti", 0.05), ("Si", 0.35)]  # base w/o Be
    for be in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        rest = 1.0 - be
        elements = [e for e, _ in others] + (["Be"] if be > 0 else [])
        ratios = [w * rest / 0.85 for _, w in others] + ([be] if be > 0 else [])
        cases.append(("light", f"AlBeMgTiSi_Be{be:.2f}", elements, ratios, "bcc"))

    # ---- low_act: Zhang 2018 VCrFeTaxWx (BCC) ----
    for x in [0.0, 0.1, 0.2, 0.3, 0.4]:
        total = 3.0 + 2 * x
        elements = ["V", "Cr", "Fe"] + (["Ta", "W"] if x > 0 else [])
        ratios = [1.0 / total] * 3 + ([x / total, x / total] if x > 0 else [])
        cases.append(("low_act", f"VCrFeTaW_x{x:.1f}", elements, ratios, "bcc"))
    return cases


def main():
    cases = build_cases()
    structures, metas = [], []
    for sysname, name, elements, ratios, phase in cases:
        for seed in range(N_SEEDS):
            import zlib
            sd = zlib.crc32(f"{name}_{seed}".encode()) % 100000
            s = build_random_structure(elements, ratios, phase, seed=sd)
            structures.append(s)
            metas.append((sysname, name, seed, s.density, len(elements)))
    print(f"built {len(structures)} structures ({len(cases)} cases x {N_SEEDS} seeds)")
    datasets = build_datasets(structures, n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")

    rows = []
    for (sysname, name, seed, dens, nsp), br, lr in zip(metas, bm_raw, ltc_raw):
        rows.append({"system": sysname, "case": name, "seed": seed,
                     "n_species": nsp, "density": round(dens, 4),
                     "bm_raw": round(br, 4), "ltc_raw": round(lr, 4),
                     "bm_GPa": round(map_value(br, "bm", "bcc", nsp), 3),
                     "ltc_WmK": round(map_value(lr, "ltc", "bcc", nsp), 3)})
    df = pd.DataFrame(rows)
    out_dir = os.path.join(TASK4_DIR, "final_data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "task4_controlled_predictions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv)

    agg = df.groupby(["system", "case"]).agg(
        n_species=("n_species", "first"),
        density=("density", "mean"), density_std=("density", "std"),
        bm=("bm_GPa", "mean"), bm_std=("bm_GPa", "std"),
        ltc=("ltc_WmK", "mean"), ltc_std=("ltc_WmK", "std")).reset_index()
    agg_csv = os.path.join(out_dir, "task4_controlled_summary.csv")
    agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    print("saved:", agg_csv)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
