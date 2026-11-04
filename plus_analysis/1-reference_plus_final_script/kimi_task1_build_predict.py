# -*- coding: utf-8 -*-
"""
kimi_task1_build_predict.py
Task 1 (plus/1-reference_plus): model-universality test on literature alloys.

For every alloy in 组分和相.csv:
  1) parse composition + phase;
  2) build random-substitution models (3x3x3 supercell of pure-metal framework,
     Vegard-estimated lattice) -- one model per listed phase;
  3) save POSCARs to ../final_alloys_structure_random/<Index>/;
  4) Mix-GNN inference + de-normalization + linear value mapping;
  5) multi-phase alloys: average the per-phase predictions;
  6) write CSV to ../final_data/ and a reference_painting.py-style figure to ../final_fig/.

Run from this folder:  python kimi_task1_build_predict.py
"""
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from pymatgen.core.periodic_table import Element
from kimi_inference_core import (build_random_structure, predict_raw,
                                 map_value, choose_framework, TASK1_DIR)

CSV_PATH = os.path.join(TASK1_DIR, "组分和相.csv")
OUT_STRUCT_DIR = os.path.join(TASK1_DIR, "final_alloys_structure_random")
OUT_DATA_DIR = os.path.join(TASK1_DIR, "final_data")
OUT_FIG_DIR = os.path.join(TASK1_DIR, "final_fig")

# number of random-substitution realizations per (alloy, phase);
# predictions are reported as mean +/- std over seeds
N_SEEDS = 5

# Compositions not numerically given in 组分和相.csv (wt.%, balance Mg).
# From Xiang et al., Mater. Charact. 202 (2023) 113028, Table 1 (ICP-measured, mass%):
#   AXM:  Al 0.86, Ca 0.12, Mn 0.35
#   AXMC: Al 0.88, Ca 0.08, Mn 0.36, Cu 0.15
#   AZ31: Al 2.82, Zn 0.88, Mn 0.33
SPECIAL_COMPOSITIONS_WT = {
    "Mg-Al-Ca-Mn-Cu (AXMC)": {"Al": 0.88, "Ca": 0.08, "Mn": 0.36, "Cu": 0.15},
    "Mg-Al-Ca-Mn (AXM)": {"Al": 0.86, "Ca": 0.12, "Mn": 0.35},
    "Mg-Al-Zn (AZ31)": {"Al": 2.82, "Zn": 0.88, "Mn": 0.33},
}

ATOMIC_MASS = {e.symbol: float(e.atomic_mass) for e in Element}


def wt_to_at(wt_dict):
    mol = {e: w / ATOMIC_MASS[e] for e, w in wt_dict.items()}
    tot = sum(mol.values())
    return {e: v / tot for e, v in mol.items()}


def parse_composition(comp):
    """Return (element_list, atomic_ratio_list, note)."""
    note = "atomic ratio parsed from formula"
    comp = comp.strip()
    comp = re.sub(r"\(TiC\).*", "", comp)          # TiC composites: model metal matrix only
    comp = re.sub(r"-HEBM$", "", comp)              # HEBM variants share the base composition
    comp = re.sub(r"\((Cantor)\)", "", comp)
    comp = comp.strip()

    if comp in SPECIAL_COMPOSITIONS_WT:
        wt = dict(SPECIAL_COMPOSITIONS_WT[comp])
        wt["Mg"] = 100.0 - sum(wt.values())
        at = wt_to_at(wt)
        elements = list(at.keys())
        ratios = [at[e] for e in elements]
        note = ("Xiang 2023 Mater. Charact. 202:113028 Table 1 (mass%, ICP), "
                "converted to at.%")
        return elements, ratios, note

    if comp.startswith("Mg-"):
        # hyphenated wt.% form, e.g. Mg-2Zn-3.6Cu-0.5Ce
        parts = comp.split("-")
        wt = {}
        for p in parts:
            m = re.match(r"^(\d*\.?\d*)([A-Z][a-z]?)$", p)
            num, ele = m.groups()
            wt[ele] = float(num) if num else 0.0
        wt["Mg"] = 100.0 - sum(wt.values())
        at = wt_to_at(wt)
        elements = list(at.keys())
        ratios = [at[e] for e in elements]
        note = "wt.% parsed from formula (balance Mg), converted to at.%"
        return elements, ratios, note

    # plain formula with atomic proportions, e.g. Al0.25CoCrFeNi, Cu5.3Cr36.8Fe36.8V21
    pairs = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", comp)
    elements = [p[0] for p in pairs]
    counts = np.array([float(p[1]) if p[1] else 1.0 for p in pairs])
    ratios = (counts / counts.sum()).tolist()
    return elements, ratios, note


def framework_for(index, elements, phase):
    """Framework selection rules (per project instruction 2026-08-05):
      refractory: contains W -> bcc/W; else contains Mo -> bcc/Mo; else -> bcc/Ta
      3d families (3d/al_3d/cu_3d): fcc -> fcc/Cu; bcc -> bcc/Fe
      mg alloys: hcp/Mg (fcc/bcc sub-models of mg-1 fall back to fcc/Cu, bcc/Nb)
      TiC composites: bcc -> bcc/Ta; fcc -> fcc/Ti
    """
    fw_dir = os.path.join(TASK1_DIR, "framework")
    fam = alloy_type(index)
    ele = None
    if fam == "refractory":
        ele = "W" if "W" in elements else ("Mo" if "Mo" in elements else "Ta")
    elif fam == "tic":
        ele = "Ta" if phase == "bcc" else "Ti"
    elif fam == "mg":
        ele = "Mg"
    else:  # 3d families
        ele = "Cu" if phase == "fcc" else "Fe"
    cand = os.path.join(fw_dir, phase, "POSCAR_" + ele)
    if os.path.exists(cand):
        return cand, ele
    # fallbacks for phase/element combinations without a matching framework
    # (only mg-1 builds fcc/bcc models: no fcc/bcc Mg framework exists)
    # note: hcp/POSCAR_Ti is a non-standard 3-atom cell (c/a=0.62) -- use Zr instead
    fallback = {"fcc": "Cu", "bcc": "Nb", "hcp": "Zr"}[phase]
    return os.path.join(fw_dir, phase, "POSCAR_" + fallback), fallback


def parse_phases(phase_str):
    """Map the Phase column to the list of frameworks to build."""
    s = str(phase_str).upper()
    if "其他" in s or "多相" in s:
        return ["hcp"]  # Mg alloys: alpha-Mg (HCP) matrix only, per project instruction
    phases = []
    for tag in ("FCC", "BCC", "HCP"):
        if tag in s:
            phases.append(tag.lower())
    return phases


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
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lighter = [c + (1.0 - c) * blend for c in rgb]
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in lighter)


def main():
    df = pd.read_csv(CSV_PATH, encoding="gbk")
    os.makedirs(OUT_STRUCT_DIR, exist_ok=True)
    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    records = []
    structures_meta = []  # (index, phase, seed, structure)
    for _, row in df.iterrows():
        index = str(row["Index"]).strip()
        comp = str(row["Composition"]).strip()
        phase_str = str(row["Phase"]).strip()
        elements, ratios, note = parse_composition(comp)
        phases = parse_phases(phase_str)
        print(f"[{index}] {comp} -> {elements} {np.round(ratios,4).tolist()} phases={phases}")
        for ph in phases:
            for seed in range(N_SEEDS):
                sd = zlib.crc32(f"{index}_{ph}_{seed}".encode()) % 100000
                structure = build_random_structure(elements, ratios, ph, seed=sd)
                if seed == 0:  # save the representative model
                    sub_dir = os.path.join(OUT_STRUCT_DIR, index)
                    os.makedirs(sub_dir, exist_ok=True)
                    fname = "POSCAR" if len(phases) == 1 else f"POSCAR_{ph.upper()}"
                    # sort by species so identical elements are merged in the
                    # element/count lines of the POSCAR
                    structure.get_sorted_structure().to(os.path.join(sub_dir, fname), fmt="poscar")
                structures_meta.append((index, ph, seed, structure))
        records.append({"index": index, "composition": comp, "phase_str": phase_str,
                        "elements": elements, "ratios": ratios, "phases": phases, "note": note})

    # ---- batch inference (CrystalNN graph built once per structure) ----
    all_structures = [m[3] for m in structures_meta]
    print(f"\nBuilding graphs for {len(all_structures)} structures "
          f"({len(records)} alloys x phases x {N_SEEDS} seeds) ...")
    from kimi_inference_core import build_datasets, predict_from_data
    datasets = build_datasets(all_structures, n_workers=8)
    print("Running inference ...")
    bm_raw_all = predict_from_data(datasets["bm"], "bm")
    ltc_raw_all = predict_from_data(datasets["ltc"], "ltc")

    # ---- per-seed detailed CSV ----
    det_rows = []
    for (index, ph, seed, structure), br, lr in zip(structures_meta, bm_raw_all, ltc_raw_all):
        n_species = len(set(structure.species))
        det_rows.append({"Index": index, "Phase_modeled": ph.upper(), "Seed": seed,
                         "BM_raw": round(br, 4), "LTC_raw": round(lr, 4),
                         "BM_GPa": round(map_value(br, "bm", ph, n_species), 3),
                         "LTC_WmK": round(map_value(lr, "ltc", ph, n_species), 3)})
    det_df = pd.DataFrame(det_rows)
    det_csv = os.path.join(OUT_DATA_DIR, "predicted_per_seed_random.csv")
    det_df.to_csv(det_csv, index=False, encoding="utf-8-sig")
    print("saved:", det_csv)

    # ---- assemble main CSV (mean +/- std over seeds and phases) ----
    rows = []
    for rec in records:
        sub = det_df[det_df["Index"] == rec["index"]]
        row = {"Index": rec["index"], "Composition": rec["composition"],
               "Phase": rec["phase_str"], "Phases_modeled": "+".join(p.upper() for p in rec["phases"]),
               "Elements": "-".join(rec["elements"]),
               "Atomic_ratios": "-".join(f"{r:.4f}" for r in rec["ratios"]),
               "Frameworks": "+".join(f"{p.upper()}:{os.path.basename(choose_framework(rec['elements'], p)).replace('POSCAR_','')}"
                                       for p in rec["phases"]),
               "Note": rec["note"]}
        for ph in rec["phases"]:
            sub_ph = sub[sub["Phase_modeled"] == ph.upper()]
            row[f"BM_{ph.upper()}_GPa"] = round(sub_ph["BM_GPa"].mean(), 3)
            row[f"BM_{ph.upper()}_std"] = round(sub_ph["BM_GPa"].std(), 3)
            row[f"LTC_{ph.upper()}_WmK"] = round(sub_ph["LTC_WmK"].mean(), 3)
            row[f"LTC_{ph.upper()}_std"] = round(sub_ph["LTC_WmK"].std(), 3)
        row["BM_pred_GPa"] = round(sub["BM_GPa"].mean(), 3)
        row["BM_pred_std"] = round(sub["BM_GPa"].std(), 3)
        row["LTC_pred_WmK"] = round(sub["LTC_WmK"].mean(), 3)
        row["LTC_pred_std"] = round(sub["LTC_WmK"].std(), 3)
        rows.append(row)
    out_df = pd.DataFrame(rows)
    csv_out = os.path.join(OUT_DATA_DIR, "predicted_bulk_modulus_and_ltc_random.csv")
    out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print("saved:", csv_out)
    print("\nDone. Now run kimi_task1_plot.py to render the figure.")


if __name__ == "__main__":
    main()
