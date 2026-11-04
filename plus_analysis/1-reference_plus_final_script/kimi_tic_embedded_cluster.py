# -*- coding: utf-8 -*-
"""
kimi_tic_embedded_cluster.py — prototype: one TiC unit cell embedded in the
BCC T22 matrix bulk (4x4x4 conventional cells, 128 metal atoms).

Carve the central ~4.3 A cube of metal sites, insert an NaCl-type TiC cell
(a = 4.328 A). The mismatch strain is local (incoherent particle), as in the
real composite. Compare B / kappa_l against the pure matrix.
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
A_TIC = 4.328


def substitute_t22(structure, seed, elements=None, ratios=None):
    elements = elements or MATRIX_ELEMENTS
    ratios = ratios or MATRIX_RATIOS
    rng = np.random.RandomState(seed)
    total = len(structure)
    counts = [int(total * r) for r in ratios]
    counts[-1] += total - sum(counts)
    # at least one atom per present element
    for i, c in enumerate(counts):
        if c == 0:
            j = max(range(len(counts)), key=lambda k: counts[k])
            counts[j] -= 1
            counts[i] = 1
    order = rng.permutation(total)
    pos = 0
    for ele, c in zip(elements, counts):
        for idx in order[pos:pos + c]:
            structure.replace(int(idx), Element(ele))
        pos += c
    return structure


def build_embedded(seed, elements=None, ratios=None, n=4):
    elements = elements or MATRIX_ELEMENTS
    ratios = ratios or MATRIX_RATIOS
    r_bar = _weighted_mean_radius(elements, ratios)
    a_mat = 4.0 * r_bar / np.sqrt(3.0)
    mat = Structure(Lattice.cubic(a_mat), ["W", "W"],
                    [[0, 0, 0], [0.5, 0.5, 0.5]]).copy()
    mat.make_supercell([n, n, n])
    mat = substitute_t22(mat, seed, elements, ratios)

    center = np.array([n * a_mat / 2.0] * 3)
    keep = [s for s in mat.sites
            if np.any(np.abs(s.coords - center) > A_TIC / 2.0 + 0.3)]
    species = [str(s.specie) for s in keep]
    coords = [s.coords for s in keep]
    ti_frac = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    c_frac = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    origin = center - A_TIC / 2.0
    for f in ti_frac:
        species.append("Ti")
        coords.append(origin + np.array(f) * A_TIC)
    for f in c_frac:
        species.append("C")
        coords.append(origin + np.array(f) * A_TIC)
    return Structure(mat.lattice, species, coords, coords_are_cartesian=True)


def main():
    sd = zlib.crc32("re-tic-1_bcc_0".encode()) % 100000
    mat = build_random_structure(MATRIX_ELEMENTS, MATRIX_RATIOS, "bcc", seed=sd)
    structures = [mat]
    labels = ["T22 matrix"]
    for seed in range(3):
        s = build_embedded(seed)
        structures.append(s)
        labels.append(f"embedded seed{seed}")
        print(f"seed{seed}: atoms={len(s)} (removed {128 - (len(s) - 8)} matrix atoms)")
    ds = build_datasets(structures, n_workers=4)
    bm = predict_from_data(ds["bm"], "bm")
    ltc = predict_from_data(ds["ltc"], "ltc")
    for lab, br, lr in zip(labels, bm, ltc):
        print(f"{lab}: B = {map_value(br, 'bm', 'bcc', 6):.1f} GPa, "
              f"kappa_l = {map_value(lr, 'ltc', 'bcc', 6):.2f} W/mK")


if __name__ == "__main__":
    main()
