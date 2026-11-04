# -*- coding: utf-8 -*-
"""
kimi_inference_core.py
Core utilities for the "plus" tasks:
  1) build random-substitution alloy structures (FCC/BCC/HCP frameworks, 3x3x3 supercell)
  2) run Mix-GNN inference for bulk modulus / lattice thermal conductivity
  3) de-normalize + linear value mapping (correction) to physical values

Pipeline reconstructed from the authors' scripts:
  mixmodel.py / data_combine.py / mlp_data_processing.py / cgcnn_data_processing.py
  nsga_2_for_core_loss_save_poscar_batch_processing_light_and_low.py
  value_mapping_example.py

Run all scripts from this folder (final_script) because features_normalization_for_running.py
loads ./bulk_modulus/data_min_list.pt etc. relative to CWD.
"""
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar
from torch_geometric.loader import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from mixmodel import MixModel            # noqa: E402
from data_combine import output_data, mlp_data, cgcnn_data, package_data     # noqa: E402
from cgcnn_data_processing import get_cgcnn_data   # noqa: E402
from mlp_data_processing import get_mlp_data       # noqa: E402

TASK1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BM_MODEL_PATH = os.path.join(
    TASK1_DIR, "pre_trained_models", "bulk_modulus", "result",
    "0625_loop_save_time_1_r_square_0.8799",
    "model_mix_best_bulk_modulus_0625_except_pure_metal_loop_save_time_1_simple.pth")
LTC_MODEL_PATH = os.path.join(
    TASK1_DIR, "pre_trained_models", "thermal_conduct", "result",
    "0802_loop_save_time_0_r_square_0.872",
    "model_mix_best_thermal_conduct_0802_except_pure_metal_loop_save_time_0.pth")

FRAMEWORK_DIR = os.path.join(TASK1_DIR, "framework")
# default pure-metal framework per phase (only used for lattice geometry / fractional sites)
FRAMEWORK_DEFAULT = {"fcc": "POSCAR_Ni", "bcc": "POSCAR_W", "hcp": "POSCAR_Ti"}

# linear correction: value_mapped = (value + b) / a   (from value_mapping_example.py)
VALUE_MAPPING = {
    ("bm", "fcc"): (9.04, 0.25),    # bm_quinary_fcc
    ("bm", "fcc_ternary"): (2.60, 0.16),  # bm_ternary_fcc
    ("bm", "bcc"): (2.38, 0.13),    # bm_quinary_bcc
    ("ltc", "fcc"): (7.75, 5.88),   # ltc_quinary_fcc
    ("ltc", "bcc"): (5.03, 1.28),   # ltc_quinary_bcc
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE = {}


def get_model(prop):
    """prop: 'bm' (bulk modulus) or 'ltc' (lattice thermal conductivity)."""
    if prop in _MODEL_CACHE:
        return _MODEL_CACHE[prop]
    if prop == "bm":
        model_path, norm_path = BM_MODEL_PATH, os.path.join(os.path.dirname(BM_MODEL_PATH), "normalization.pt")
    elif prop == "ltc":
        model_path, norm_path = LTC_MODEL_PATH, os.path.join(os.path.dirname(LTC_MODEL_PATH), "normalization.pt")
    else:
        raise ValueError(prop)
    model = MixModel()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    normalization = torch.load(norm_path, map_location="cpu")
    _MODEL_CACHE[prop] = (model, normalization)
    return _MODEL_CACHE[prop]


def predict_raw(structures, prop):
    """Run inference. structures: list of pymatgen Structure.
    Returns list of de-normalized raw outputs (same convention as the NSGA-II
    POSCAR filename values, i.e. BEFORE the linear value mapping)."""
    material_property = "bulk_modulus" if prop == "bm" else "thermal_conduct"
    data_list = [output_data(s, material_property) for s in structures]
    return predict_from_data(data_list, prop)


def _build_one(structure):
    """Top-level worker: build [mlp, gcn] data for both properties of one
    structure (picklable for multiprocessing)."""
    gcn = cgcnn_data(get_cgcnn_data(structure))
    return (package_data(mlp_data(get_mlp_data(structure, "bulk_modulus")), gcn),
            package_data(mlp_data(get_mlp_data(structure, "thermal_conduct")), gcn))


def build_datasets(structures, n_workers=1, log_every=20):
    """Build the [mlp, gcn] data objects for both properties in one pass.
    The expensive CrystalNN graph is constructed only once per structure and
    shared by the bm / ltc entries. n_workers > 1 parallelizes graph building
    across CPU processes."""
    if n_workers and n_workers > 1 and len(structures) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = [None] * len(structures)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_build_one, s): i for i, s in enumerate(structures)}
            done = 0
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if log_every and (done % log_every == 0 or done == len(structures)):
                    print(f"  graphs: {done}/{len(structures)} "
                          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    else:
        results = []
        for i, s in enumerate(structures):
            results.append(_build_one(s))
            if log_every and ((i + 1) % log_every == 0 or i + 1 == len(structures)):
                print(f"  graphs: {i+1}/{len(structures)}", flush=True)
    ds_bm = [r[0] for r in results]
    ds_ltc = [r[1] for r in results]
    return {"bm": ds_bm, "ltc": ds_ltc}


def predict_from_data(data_list, prop):
    """Inference on pre-built data objects (see build_datasets)."""
    model, normalization = get_model(prop)
    loader = DataLoader(data_list, batch_size=min(512, len(data_list)), shuffle=False)
    y_list = []
    with torch.no_grad():
        for batch in loader:
            for i in range(1, len(batch[0])):
                batch[0][i] = batch[0][i].to(device)
            batch[1] = batch[1].to(device)
            out = model(batch[0], batch[1].x.float(), batch[1].edge_index.long(),
                        batch[1].edge_attr.float(), batch[1].batch)
            out = out.cpu().tolist()
            y_list.extend([float(o[0]) * float(normalization[1]) + float(normalization[0]) for o in out])
    return y_list


def map_value(raw_value, prop, phase, n_species=None):
    """Linear correction to physical units.
    prop: 'bm' -> GPa ; 'ltc' -> W m-1 K-1.
    Mapping choice (validated subspaces of the manuscript):
      FCC 3-species bm -> ternary_fcc mapping; other FCC -> quinary_fcc mapping;
      BCC (any) -> quinary_bcc mapping;
      HCP -> no validated subspace; use the FCC (close-packed) mapping as default.
    """
    key_phase = phase.lower()
    if prop == "bm" and key_phase == "fcc" and n_species == 3:
        key = ("bm", "fcc_ternary")
    elif key_phase in ("fcc", "bcc"):
        key = (prop, key_phase)
    elif key_phase == "hcp":
        key = (prop, "fcc")  # no validated HCP subspace; closest packed analogue
    else:
        raise ValueError(phase)
    b, a = VALUE_MAPPING[key]
    return (raw_value + b) / a


def choose_framework(element_list, phase):
    """Pick the framework POSCAR of the first constituent element that has one
    for this phase; otherwise the phase default. Geometry is identical within a
    phase (lattice is re-scaled by Vegard estimate afterwards)."""
    phase_dir = os.path.join(FRAMEWORK_DIR, phase.lower())
    for ele in element_list:
        cand = os.path.join(phase_dir, "POSCAR_" + ele)
        if os.path.exists(cand):
            return cand
    return os.path.join(phase_dir, FRAMEWORK_DEFAULT[phase.lower()])


def _weighted_mean_radius(element_list, ratio_list):
    # empirical (metallic) atomic radii give Vegard estimates closest to real
    # alloy lattice constants; fall back to calculated radii when missing
    r = 0.0
    for ele, w in zip(element_list, ratio_list):
        radius = Element(ele).atomic_radius or Element(ele).atomic_radius_calculated
        r += w * float(radius)
    return r


def vegard_lattice(element_list, ratio_list, phase):
    """Estimate lattice parameter(s) by Vegard's law on calculated atomic radii."""
    r = _weighted_mean_radius(element_list, ratio_list)
    phase = phase.lower()
    if phase == "fcc":
        a = 2.0 * np.sqrt(2.0) * r
        return Lattice.cubic(a)
    if phase == "bcc":
        a = 4.0 * r / np.sqrt(3.0)
        return Lattice.cubic(a)
    if phase == "hcp":
        a = 2.0 * r
        return Lattice.hexagonal(a, 1.633 * a)
    raise ValueError(phase)


def build_random_structure(element_list, ratio_list, phase, seed=0, supercell=(3, 3, 3),
                           framework_path=None):
    """Random-substitution ('random sprinkling') alloy model on a pure-metal framework.

    element_list: e.g. ["Cr", "Mn", "Fe", "Co", "Ni"]
    ratio_list:   atomic fractions summing to 1, same order as element_list
    phase:        'fcc' / 'bcc' / 'hcp'
    framework_path: optional explicit framework POSCAR; default = first
        constituent element that has a framework for this phase, else the
        phase default (Ni/W/Ti).
    Lattice: Vegard estimate (2*sqrt(2)*r_bar for fcc, 4/sqrt(3)*r_bar for bcc,
    a=2*r_bar, c=1.633a for hcp) scaled up to the supercell dimensions.
    """
    assert abs(sum(ratio_list) - 1.0) < 1e-6, sum(ratio_list)
    # drop zero-fraction elements entirely (no phantom single atoms)
    nonzero = [(e, r) for e, r in zip(element_list, ratio_list) if r > 0]
    element_list = [e for e, _ in nonzero]
    ratio_list = [r for _, r in nonzero]
    rng = random.Random(seed)
    framework = framework_path or choose_framework(element_list, phase)
    unit = Poscar.from_file(framework).structure
    unit.make_supercell(list(supercell))
    structure = unit.copy()

    total = len(structure)
    counts = [int(total * r) for r in ratio_list]
    counts[-1] += total - sum(counts)  # remainder to last element
    # guarantee at least one atom per element (mirrors the authors' convention)
    for i, c in enumerate(counts):
        if c == 0:
            j = max(range(len(counts)), key=lambda k: counts[k])
            counts[j] -= 1
            counts[i] = 1

    index_list = list(range(total))
    rng.shuffle(index_list)
    pos = 0
    for ele, c in zip(element_list, counts):
        for idx in index_list[pos:pos + c]:
            structure.replace(idx, Element(ele))
        pos += c

    new_lattice = vegard_lattice(element_list, ratio_list, phase)
    # scale the unit-cell Vegard lattice up to the supercell dimensions
    scale = np.array(supercell, dtype=float)
    new_lattice = Lattice(new_lattice.matrix * scale[:, None])
    structure = Structure(new_lattice, structure.species, structure.frac_coords)
    return structure


def predict_alloy(element_list, ratio_list, phases, seed=0, keep_structures=False):
    """Predict (B [GPa], kappa_l [W/mK]) for one alloy.
    phases: list like ['fcc'] or ['fcc','bcc'] -> predictions averaged over phases.
    Returns dict with per-phase and mean values."""
    per_phase = {}
    structures = []
    for ph in phases:
        structures.append(build_random_structure(element_list, ratio_list, ph, seed=seed))
    datasets = build_datasets(structures)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")
    n = len(element_list)
    for k, ph in enumerate(phases):
        per_phase[ph] = {
            "bm_raw": bm_raw[k], "ltc_raw": ltc_raw[k],
            "bm": map_value(bm_raw[k], "bm", ph, n),
            "ltc": map_value(ltc_raw[k], "ltc", ph, n),
        }
    out = {
        "per_phase": per_phase,
        "bm_mean": float(np.mean([v["bm"] for v in per_phase.values()])),
        "ltc_mean": float(np.mean([v["ltc"] for v in per_phase.values()])),
    }
    if keep_structures:
        out["structures"] = structures
    return out
