# -*- coding: utf-8 -*-
"""
kimi_task4_predict.py
Task 4 (plus/4.different_system): batch prediction for the light / low_act
alloy POSCAR models, and group statistics for the with/without-element
comparison.

  light/:    Light-* (Be-bearing, FCC) vs Nobe-* (Be-free, FCC)
             -> effect of Be on density and bulk modulus
  low_act/:  Low-*  (BCC); alloys containing micro elements (Ta, Ti, Zr, Si,
             per the authors' micro-element list) vs those without
             -> effect of micro additions on bulk modulus (hardness proxy)
  refractory/:  (currently empty -- pending)

Outputs: ../final_data/task4_predictions.csv  + printed group statistics.
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
os.chdir(TASK1_SCRIPT_DIR)  # normalization .pt files load relative to CWD

from pymatgen.io.vasp import Poscar                       # noqa: E402
from kimi_inference_core import build_datasets, predict_from_data, map_value  # noqa: E402

MICRO_ELEMENTS = ["W", "Ta", "Sc", "Ti", "Y", "Zr", "Ru", "Rh", "Zn", "Si"]

# (subfolder, phase for mapping, system tag)
GROUPS = [("light", "fcc"), ("low_act", "bcc")]


def collect():
    rows = []
    for sub, phase in GROUPS:
        base = os.path.join(TASK4_DIR, sub)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name, "POSCAR")
            if not os.path.isfile(p):
                continue
            structure = Poscar.from_file(p).structure
            elements = [str(e) for e in structure.elements]
            rows.append({"system": sub, "name": name, "phase": phase,
                         "structure": structure,
                         "elements": "-".join(elements),
                         "n_atoms": len(structure),
                         "has_micro": any(e in MICRO_ELEMENTS for e in elements),
                         "has_Be": "Be" in elements,
                         "density": structure.density})
    return rows


def main():
    rows = collect()
    print(f"collected {len(rows)} structures")
    datasets = build_datasets([r["structure"] for r in rows], n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")
    for r, br, lr in zip(rows, bm_raw, ltc_raw):
        n_species = r["elements"].count("-") + 1
        r["bm_GPa"] = map_value(br, "bm", r["phase"], n_species)
        r["ltc_WmK"] = map_value(lr, "ltc", r["phase"], n_species)
        r["bm_raw"] = br
        r["ltc_raw"] = lr
        del r["structure"]
    df = pd.DataFrame(rows)
    out_dir = os.path.join(TASK4_DIR, "final_data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "task4_predictions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv)

    print("\n=== group statistics (mean +/- std) ===")
    for sub, df_sub in df.groupby("system"):
        if sub == "light":
            for flag, g in df_sub.groupby("has_Be"):
                tag = "Be-bearing" if flag else "Be-free"
                print(f"light {tag} (n={len(g)}): "
                      f"B = {g['bm_GPa'].mean():.1f} +/- {g['bm_GPa'].std():.1f} GPa, "
                      f"density = {g['density'].mean():.2f} +/- {g['density'].std():.2f} g/cm3, "
                      f"kappa = {g['ltc_WmK'].mean():.2f} +/- {g['ltc_WmK'].std():.2f} W/mK")
        if sub == "low_act":
            for flag, g in df_sub.groupby("has_micro"):
                tag = "with micro (Ta/Ti/Zr/Si)" if flag else "micro-free"
                print(f"low_act {tag} (n={len(g)}): "
                      f"B = {g['bm_GPa'].mean():.1f} +/- {g['bm_GPa'].std():.1f} GPa, "
                      f"density = {g['density'].mean():.2f} +/- {g['density'].std():.2f} g/cm3, "
                      f"kappa = {g['ltc_WmK'].mean():.2f} +/- {g['ltc_WmK'].std():.2f} W/mK")


if __name__ == "__main__":
    main()
