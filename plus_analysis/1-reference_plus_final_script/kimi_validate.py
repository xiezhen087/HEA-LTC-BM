# -*- coding: utf-8 -*-
"""kimi_validate.py — end-to-end sanity checks for the inference pipeline.
1) Build Cantor alloy (CrMnFeCoNi, FCC) by random substitution and predict.
2) Inference on existing NSGA-II POSCARs (task-2 folder) whose filenames contain
   the raw model outputs -> must reproduce those numbers.
3) Inference on a Pr-containing low-activation POSCAR (feature coverage check).
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)  # normalization pt files are loaded relative to CWD

from pymatgen.io.vasp import Poscar
from kimi_inference_core import predict_raw, build_random_structure, predict_alloy

TASK1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PLUS_DIR = os.path.abspath(os.path.join(TASK1_DIR, ".."))

print("=== check 1: Cantor CrMnFeCoNi FCC, random substitution ===")
res = predict_alloy(["Cr", "Mn", "Fe", "Co", "Ni"], [0.2] * 5, ["fcc"], seed=0, keep_structures=True)
s = res["structures"][0]
print("n_atoms:", len(s), "lattice a:", round(s.lattice.a, 4))
print("B raw %.3f -> mapped %.2f GPa" % (res["per_phase"]["fcc"]["bm_raw"], res["per_phase"]["fcc"]["bm"]))
print("k raw %.3f -> mapped %.2f W/mK" % (res["per_phase"]["fcc"]["ltc_raw"], res["per_phase"]["fcc"]["ltc"]))

print("\n=== check 2: reproduce raw values stored in task-2 POSCAR filenames ===")
for subspace, folder, fname in [
    ("bcc", "BCC", "POSCAR_1540_30.01_23.07"),
    ("fcc", "FCC", "POSCAR_1999_56.75_36.54"),
]:
    p = os.path.join(PLUS_DIR, "2-Mo-Ti-Os-Ir_plus", folder, fname)
    structure = Poscar.from_file(p).structure
    bm_raw = predict_raw([structure], "bm")[0]
    ltc_raw = predict_raw([structure], "ltc")[0]
    parts = fname.split("_")
    bm_ref, ltc_ref = float(parts[-2]), float(parts[-1])
    print(f"{fname}: bm {bm_raw:.2f} (ref {bm_ref}), ltc {ltc_raw:.2f} (ref {ltc_ref})")

print("\n=== check 3: Pr-containing low-activation alloy POSCAR ===")
p = os.path.join(PLUS_DIR, "4.different_system", "low_act",
                 "Low-11-V0.35-Cr0.35-Pr0.21-Ti0.05-Si0.04", "POSCAR")
structure = Poscar.from_file(p).structure
bm_raw = predict_raw([structure], "bm")[0]
ltc_raw = predict_raw([structure], "ltc")[0]
print("Low-11: bm raw %.3f, ltc raw %.3f" % (bm_raw, ltc_raw))
print("elements:", [str(e) for e in structure.elements])
print("\nALL CHECKS DONE")
