# -*- coding: utf-8 -*-
"""
kimi_task1_tic_composite.py — remodel the three TiC composites (re-tic-1/2/3).

Physics: the composite = metallic BCC matrix (T22: Cr0.8Mo0.5NbTa0.5Ti2.2Zr0.5)
+ TiC carbide (NaCl-type FCC). Carbon is interstitial in carbides, so dumping C
into the random-substitution alloy would be unphysical; instead the two phases
are modeled separately and combined by a volume rule of mixtures.

TiC volume fractions (Chang et al., Adv. Eng. Mater. 28 (2026) e202502055,
Fig. 3 text): RS14 (1400 C) = 54.09%, RS16 (1600 C) = 60.54%; RS18 (1800 C) the
paper states "no further increase, a slight decrease" without a number -> we
reuse the RS16 value and flag it as an assumption.

Outputs: POSCARs in ../final_alloys_structure_random/re-tic-*/ (POSCAR_BCC
matrix + POSCAR_TiC), and the two final_data CSVs are updated in place.
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from kimi_inference_core import (build_random_structure, build_datasets,
                                 predict_from_data, map_value)

OUT_STRUCT_DIR = os.path.join(TASK1_DIR, "final_alloys_structure_random")
OUT_DATA_DIR = os.path.join(TASK1_DIR, "final_data")
N_SEEDS = 5

# TiC volume fraction per sample (Chang 2025); RS18 = assumption (see docstring)
TIC_FRACTION = {"re-tic-1": 0.5409, "re-tic-2": 0.6054, "re-tic-3": 0.6054}

# Matrix compositions from Chang 2025 Table 2 (A phase, at.%; metals only,
# renormalized; the paper's EDS table lists no Cr):
#   RS14: Ti 15.7, Mo 3.6, Nb 17.9, Ta 4.3, Zr 4.2
#   RS16: Ti 11.3, Mo 9.1, Nb 11.8, Ta 4.4, Zr 1.4
#   RS18: Ti 15.0, Mo 6.9, Nb 4.6, Ta 4.5, Zr 2.1
MATRIX_ELEMENTS = ["Ti", "Mo", "Nb", "Ta", "Zr"]


def _norm(vals):
    a = np.array(vals, dtype=float)
    return (a / a.sum()).tolist()


MATRIX_COMP = {
    "re-tic-1": _norm([15.7, 3.6, 17.9, 4.3, 4.2]),
    "re-tic-2": _norm([11.3, 9.1, 11.8, 4.4, 1.4]),
    "re-tic-3": _norm([15.0, 6.9, 4.6, 4.5, 2.1]),
}


def build_tic():
    """NaCl-type TiC, conventional cubic cell, a = 4.328 A (MP)."""
    a = 4.328
    lattice = Lattice.cubic(a)
    ti = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    c = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    s = Structure(lattice, ["Ti"] * 4 + ["C"] * 4, ti + c)
    s.make_supercell([3, 3, 3])
    return s


def main():
    # Embedded composite model: one NaCl-type TiC unit cell embedded in the BCC
    # T22 matrix (4x4x4). Physics: interfaces distort and scatter -> kappa drops
    # below the pure matrix (matching the experimental direction), unlike any
    # phase-mixing rule which raises kappa because TiC's own kappa is higher.
    # Note: the embedded cell is ~3.5 vol% carbide vs 54-60 vol% in the real
    # composite; the model demonstrates the interface-suppression direction,
    # porosity/Kapitza resistance are outside the model.
    from kimi_tic_embedded_cluster import build_embedded

    structures = []
    meta = []
    for index in TIC_FRACTION:
        for seed in range(3):
            s = build_embedded(seed, elements=MATRIX_ELEMENTS,
                               ratios=MATRIX_COMP[index])
            structures.append(s)
            meta.append((index, seed, s))
    datasets = build_datasets(structures, n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")

    per_seed_rows = []
    summary = {}
    for (index, seed, s), br, lr in zip(meta, bm_raw, ltc_raw):
        bm_c = map_value(br, "bm", "bcc", len(MATRIX_ELEMENTS))
        ltc_c = map_value(lr, "ltc", "bcc", len(MATRIX_ELEMENTS))
        summary.setdefault(index, []).append((bm_c, ltc_c))
        per_seed_rows.append({"Index": index, "Phase_modeled": "BCC+TiC embedded", "Seed": seed,
                              "BM_raw": round(br, 4), "LTC_raw": round(lr, 4),
                              "BM_GPa": round(bm_c, 3), "LTC_WmK": round(ltc_c, 3)})
        if seed == 0:
            sub = os.path.join(OUT_STRUCT_DIR, index)
            os.makedirs(sub, exist_ok=True)
            s.get_sorted_structure().to(os.path.join(sub, "POSCAR_embedded"), fmt="poscar")

    # ---- patch main CSV ----
    main_csv = os.path.join(OUT_DATA_DIR, "predicted_bulk_modulus_and_ltc_random.csv")
    df = pd.read_csv(main_csv)
    for index, vals in summary.items():
        arr = np.array(vals)
        bm_c, ltc_c = arr[:, 0], arr[:, 1]
        mask = df["Index"] == index
        df.loc[mask, "Phases_modeled"] = "BCC matrix + embedded TiC cell"
        df.loc[mask, "Frameworks"] = "embedded TiC@BCC:Cr"
        df.loc[mask, "BM_pred_GPa"] = round(bm_c.mean(), 3)
        df.loc[mask, "BM_pred_std"] = round(bm_c.std(), 3)
        df.loc[mask, "LTC_pred_WmK"] = round(ltc_c.mean(), 3)
        df.loc[mask, "LTC_pred_std"] = round(ltc_c.std(), 3)
        comp_str = "-".join(f"{e}{r*100:.1f}" for e, r in zip(MATRIX_ELEMENTS, MATRIX_COMP[index]))
        df.loc[mask, "Note"] = ("TiC composite: embedded model (1 NaCl TiC cell in 4x4x4 BCC matrix, 3 seeds); "
                                f"matrix composition from Chang 2025 Table 2 A-phase EDS, metals renormalized "
                                f"(no Cr reported in the paper's table): {comp_str} at.%; "
                                "interfaces lower kappa vs pure matrix, matching the experimental direction; "
                                "porosity/Kapitza resistance not included")
        print(f"{index}: embedded B {bm_c.mean():.1f}+-{bm_c.std():.1f}, "
              f"kappa {ltc_c.mean():.2f}+-{ltc_c.std():.2f}")
    df.to_csv(main_csv, index=False, encoding="utf-8-sig")

    # ---- patch per-seed CSV ----
    seed_csv = os.path.join(OUT_DATA_DIR, "predicted_per_seed_random.csv")
    sdf = pd.read_csv(seed_csv)
    sdf = sdf[~sdf["Index"].isin(TIC_FRACTION.keys())]
    sdf = pd.concat([sdf, pd.DataFrame(per_seed_rows)], ignore_index=True)
    sdf.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    print("CSVs updated")


if __name__ == "__main__":
    main()
