# -*- coding: utf-8 -*-
"""
kimi_tic_embedded_prototype.py — prototype: embedded matrix/TiC sandwich model.

BCC T22 matrix slab | TiC (100) slab | BCC matrix, periodic along z.
In-plane co-strained to a common lattice (split the ~7% mismatch).
Purpose: test whether the Mix-GNN predicts a LOWER kappa_l for the embedded
composite than for the pure matrix (the expected experimental direction).
"""
import os
import sys
import zlib

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from kimi_inference_core import (build_random_structure, build_datasets,
                                 predict_from_data, map_value,
                                 _weighted_mean_radius)

MATRIX_ELEMENTS = ["Cr", "Mo", "Nb", "Ta", "Ti", "Zr"]
MATRIX_RATIOS = np.array([0.8, 0.5, 1.0, 0.5, 2.2, 0.5])
MATRIX_RATIOS = (MATRIX_RATIOS / MATRIX_RATIOS.sum()).tolist()


def build_sandwich(seed, n_mat_xy=3, n_mat_z=4, n_tic_layers=4):
    r_bar = _weighted_mean_radius(MATRIX_ELEMENTS, MATRIX_RATIOS)
    a_mat = 4.0 * r_bar / np.sqrt(3.0)          # BCC lattice constant of matrix
    a_tic = 4.328                                # TiC NaCl lattice constant
    ip_mat = n_mat_xy * a_mat                    # matrix in-plane box
    ip_tic = n_mat_xy * a_tic / np.sqrt(2.0)     # TiC (100) in-plane box
    a_ip = 0.5 * (ip_mat + ip_tic) / n_mat_xy    # compromise in-plane cell

    # --- matrix slab (BCC conventional cell replicated n_mat_xy x n_mat_xy x n_mat_z)
    lat_m = Lattice.cubic(a_ip)
    species = ["W"] * (2 * n_mat_xy * n_mat_xy * n_mat_z)
    base = [[0, 0, 0], [0.5, 0.5, 0.5]]
    coords = []
    for i in range(n_mat_xy):
        for j in range(n_mat_xy):
            for k in range(n_mat_z):
                for b in base:
                    coords.append([(b[0] + i) / n_mat_xy, (b[1] + j) / n_mat_xy,
                                   (b[2] + k) / n_mat_z])
    mat = Structure(lat_m, species, coords)
    # random substitution of the T22 composition
    rng = np.random.RandomState(seed)
    total = len(mat)
    counts = [int(total * r) for r in MATRIX_RATIOS]
    counts[-1] += total - sum(counts)
    order = rng.permutation(total)
    pos = 0
    for ele, c in zip(MATRIX_ELEMENTS, counts):
        for idx in order[pos:pos + c]:
            mat.replace(int(idx), Element(ele))
        pos += c

    # --- TiC (100) slab: layers spaced a_ip_z = a_tic/2 along z, in-plane net a_tic/sqrt(2)
    d_layer = a_tic / 2.0
    gap = 2.0  # interface distance (A), typical metal-carbide contact
    z0 = gap + n_mat_z * a_ip  # gap after matrix top
    tic_species, tic_coords = [], []
    inplane_step = a_tic / np.sqrt(2.0)
    for l in range(n_tic_layers):
        z = z0 + l * d_layer
        for i in range(n_mat_xy):
            for j in range(n_mat_xy):
                x = i * inplane_step
                y = j * inplane_step
                if l % 2 == 0:
                    tic_species += ["Ti", "C"]
                    tic_coords += [[x, y, z], [x + inplane_step / 2, y + inplane_step / 2, z]]
                else:
                    tic_species += ["C", "Ti"]
                    tic_coords += [[x, y, z], [x + inplane_step / 2, y + inplane_step / 2, z]]
    z_tot = z0 + n_tic_layers * d_layer + gap  # gap before periodic matrix image
    box = Lattice.from_parameters(n_mat_xy * a_ip, n_mat_xy * a_ip, z_tot, 90, 90, 90)
    all_species = [str(s) for s in mat.species] + tic_species
    all_coords = [s.coords for s in mat.sites] + tic_coords
    sand = Structure(box, all_species, all_coords, coords_are_cartesian=True)
    return sand, a_mat, a_ip


def main():
    # pure matrix reference (same seed protocol as task 1)
    sd = zlib.crc32("re-tic-1_bcc_0".encode()) % 100000
    mat = build_random_structure(MATRIX_ELEMENTS, MATRIX_RATIOS, "bcc", seed=sd)
    structures = [mat]
    labels = ["T22 matrix"]
    for seed in range(3):
        s, a_mat, a_ip = build_sandwich(seed)
        structures.append(s)
        labels.append(f"sandwich seed{seed}")
        print(f"seed{seed}: a_mat={a_mat:.3f} a_ip={a_ip:.3f} atoms={len(s)}")
    ds = build_datasets(structures, n_workers=4)
    bm = predict_from_data(ds["bm"], "bm")
    ltc = predict_from_data(ds["ltc"], "ltc")
    for lab, br, lr in zip(labels, bm, ltc):
        bmv = map_value(br, "bm", "bcc", 6)
        ltcv = map_value(lr, "ltc", "bcc", 6)
        print(f"{lab}: B = {bmv:.1f} GPa, kappa_l = {ltcv:.2f} W/mK")


if __name__ == "__main__":
    main()
