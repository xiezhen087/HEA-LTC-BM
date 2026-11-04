# -*- coding: utf-8 -*-
"""
kimi_task4v2_build_all.py — Task 4 (v2) modeling & POSCAR saving.

  structure/1  refractory: HfMoNbTaTiZr base + (-Mo) + (-Ta) + (-Zr),
               10 seeds each (error bars for the Mo/Ta/Zr figure)
  structure/2  AlxCoCrFeNi (x = 0..2), phase(s) per Chou 2009 Table 1,
               duplex alloys = FCC+BCC models weighted by the tabulated
               weight fractions; 5 seeds
  structure/3  light grids: AlBeMgTiLi / AlBeMgTiSi / AlBeMgTiCu, each
               element 5-35% step 5% (Sorkin 2023 protocol), 1 seed
  structure/4  low_act: VCrFe + VCrFeTaxWx (x = 0.1..0.4), 5 seeds

Output CSVs in ../final_data/:
  task4v2_refractory.csv, task4v2_alxcocrfeni.csv,
  task4v2_light_grid_<family>.csv, task4v2_lowact.csv
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

STRUCT_DIR = os.path.join(TASK4_DIR, "structure")
OUT_DATA = os.path.join(TASK4_DIR, "final_data")


def save_poscar(structure, group, case_name, tag):
    d = os.path.join(STRUCT_DIR, group, case_name)
    os.makedirs(d, exist_ok=True)
    fname = "POSCAR" if not tag else f"POSCAR_{tag}"
    structure.get_sorted_structure().to(os.path.join(d, fname), fmt="poscar")


def run_cases(case_list, group, n_seeds, save=True):
    """case_list: [(case_name, elements, ratios, phase)]. Returns per-structure df."""
    phases_of = {}
    for name, _, _, phase in case_list:
        phases_of.setdefault(name, set()).add(phase)
    structures, metas = [], []
    for name, elements, ratios, phase in case_list:
        for seed in range(n_seeds):
            sd = zlib.crc32(f"{group}_{name}_{phase}_{seed}".encode()) % 100000
            s = build_random_structure(elements, ratios, phase, seed=sd)
            structures.append(s)
            metas.append((name, phase, seed, s.density, len(elements)))
            if save and seed == 0:
                tag = phase.upper() if len(phases_of[name]) > 1 else ""
                save_poscar(s, group, name, tag)
    print(f"[{group}] built {len(structures)} structures from {len(case_list)} cases x {n_seeds} seeds")
    datasets = build_datasets(structures, n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")
    rows = []
    for (name, phase, seed, dens, nsp), br, lr in zip(metas, bm_raw, ltc_raw):
        rows.append({"case": name, "phase": phase, "seed": seed,
                     "density": round(dens, 4), "n_species": nsp,
                     "bm_GPa": round(map_value(br, "bm", phase, nsp), 3),
                     "ltc_WmK": round(map_value(lr, "ltc", phase, nsp), 3)})
    return pd.DataFrame(rows)


def equiatomic(elements):
    return elements, [1.0 / len(elements)] * len(elements)


def part1_refractory():
    base = ["Hf", "Mo", "Nb", "Ta", "Ti", "Zr"]
    variants = {
        "HfMoNbTaTiZr": base,
        "HfNbTaTiZr(-Mo)": [e for e in base if e != "Mo"],
        "HfMoNbTiZr(-Ta)": [e for e in base if e != "Ta"],
        "HfMoNbTaTi(-Zr)": [e for e in base if e != "Zr"],
    }
    cases = []
    for name, els in variants.items():
        elements, ratios = equiatomic(els)
        cases.append((name, elements, ratios, "bcc"))
    df = run_cases(cases, "1", n_seeds=10)
    df.to_csv(os.path.join(OUT_DATA, "task4v2_refractory.csv"), index=False, encoding="utf-8-sig")
    print(df.groupby("case")[["bm_GPa", "ltc_WmK"]].agg(["mean", "std"]).round(2).to_string())


# Chou 2009 Table 1: x -> (weight% FCC, weight% BCC); single-phase if only one
ALX_PHASES = {
    0.0:   {"fcc": 100.0},
    0.25:  {"fcc": 100.0},
    0.375: {"fcc": 100.0},
    0.5:   {"fcc": 82.41, "bcc": 17.59},
    0.75:  {"fcc": 73.68, "bcc": 26.32},
    0.875: {"fcc": 43.98, "bcc": 56.02},
    1.0:   {"fcc": 15.61, "bcc": 84.39},
    1.25:  {"bcc": 100.0},
    1.5:   {"bcc": 100.0},
    2.0:   {"bcc": 100.0},
}
# Chou 2009 Table 2: phonon (lattice) part of thermal conductivity, ranges
ALX_LIT_KPH = {  # x: (min, max) W/mK
    0.0: (4.7, 5.7), 0.25: (5.7, 7.0), 0.375: (5.6, 7.1), 0.5: (5.8, 7.2),
    0.75: (6.0, 7.0), 0.875: (6.4, 6.6), 1.0: (6.5, 7.1),
    1.25: (14.6, 15.3), 1.5: (14.3, 14.5), 2.0: (11.0, 12.2),
}


def part2_alxcocrfeni():
    cases = []
    for x, phases in ALX_PHASES.items():
        elements = ["Al", "Co", "Cr", "Fe", "Ni"]
        counts = np.array([x, 1, 1, 1, 1])
        ratios = (counts / counts.sum()).tolist()
        for ph in phases:
            cases.append((f"Al{x}CoCrFeNi", elements, ratios, ph))
    df = run_cases(cases, "2", n_seeds=5)
    # aggregate per (case, phase) then weight by phase fraction
    agg = df.groupby(["case", "phase"]).agg(
        bm=("bm_GPa", "mean"), bm_std=("bm_GPa", "std"),
        ltc=("ltc_WmK", "mean"), ltc_std=("ltc_WmK", "std"),
        density=("density", "mean")).reset_index()
    rows = []
    for x, phases in ALX_PHASES.items():
        name = f"Al{x}CoCrFeNi"
        sub = agg[agg["case"] == name]
        bm_w = ltc_w = 0.0
        bm_var = ltc_var = 0.0
        for ph, w in phases.items():
            r = sub[sub["phase"] == ph].iloc[0]
            w = w / 100.0
            bm_w += w * r["bm"]
            ltc_w += w * r["ltc"]
            bm_var += (w * r["bm_std"]) ** 2
            ltc_var += (w * r["ltc_std"]) ** 2
        lo, hi = ALX_LIT_KPH[x]
        rows.append({"x": x, "case": name,
                     "bm_weighted": bm_w, "bm_std": np.sqrt(bm_var),
                     "ltc_weighted": ltc_w, "ltc_std": np.sqrt(ltc_var),
                     "lit_kph_mid": (lo + hi) / 2, "lit_kph_err": (hi - lo) / 2})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DATA, "task4v2_alxcocrfeni.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(OUT_DATA, "task4v2_alxcocrfeni_per_phase.csv"), index=False, encoding="utf-8-sig")
    print(out.round(2).to_string(index=False))


def part3_light_grids():
    families = {"AlBeMgTiLi": "Li", "AlBeMgTiSi": "Si", "AlBeMgTiCu": "Cu"}
    step = 5
    units = list(range(1, 8))
    for fam, fifth in families.items():
        out_csv = os.path.join(OUT_DATA, f"task4v2_light_grid_{fam}.csv")
        cases = []
        for a in units:
            for b in units:
                for c in units:
                    for d in units:
                        e = 20 - (a + b + c + d)
                        if e in units:
                            elements = ["Al", "Be", "Mg", "Ti", fifth]
                            ratios = [v * step / 100.0 for v in (a, b, c, d, e)]
                            cases.append((f"{fam}_{a}-{b}-{c}-{d}-{e}", elements, ratios, "bcc"))
        df = run_cases(cases, "3", n_seeds=3)
        df["B_over_rho"] = df["bm_GPa"] / df["density"]
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        g = df.groupby(df["case"].str.split("-").str[1].astype(int) * 5)["B_over_rho"].mean()
        print(fam, "mean B/rho vs Be%:", g.round(2).to_dict())


def part4_lowact():
    cases = []
    for x in [0.0, 0.1, 0.2, 0.3, 0.4]:
        total = 3.0 + 2 * x
        elements = ["V", "Cr", "Fe"] + (["Ta", "W"] if x > 0 else [])
        ratios = [1.0 / total] * 3 + ([x / total, x / total] if x > 0 else [])
        cases.append((f"VCrFeTaW_x{x:.1f}", elements, ratios, "bcc"))
    df = run_cases(cases, "4", n_seeds=5)
    df.to_csv(os.path.join(OUT_DATA, "task4v2_lowact.csv"), index=False, encoding="utf-8-sig")
    print(df.groupby("case")[["bm_GPa", "ltc_WmK"]].agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    part1_refractory()
    part2_alxcocrfeni()
    part3_light_grids()
    part4_lowact()
