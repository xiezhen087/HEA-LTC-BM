# -*- coding: utf-8 -*-
"""
optimize.py — command-line entry for HEA composition optimization.

Task division:
  - NSGA-II and TPE optimization algorithms: (author; see module docstrings)
  - CLI packaging: Kimi

Examples
--------
Multi-objective optimization of bulk modulus B and lattice thermal
conductivity kappa_l with NSGA-II (quinary FCC+BCC search):

    python code/optimize.py --method nsga2 --n-components 5 --frameworks fcc bcc \
        --elements Ti V Cr Mn Fe Co Ni Cu Nb Mo Al Y \
        --generations 500 --popnum 100 --out-dir ./opt_run

Single-objective TPE (Bayesian) optimization of one property:

    python code/optimize.py --method tpe --property thermal_conduct \
        --n-components 4 --frameworks fcc --trials 2000 --out-dir ./tpe_run

Pre-trained model weights default to pre_trained_models/ in this repository;
override with --bm-model / --ltc-model (file paths) if needed.
"""
import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

_CODE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR / "optimization"))


def split_path_arg(s):
    """'dir/file.pth' -> (dir, file); a bare filename keeps the default dir."""
    d, f = os.path.split(s)
    return d, f


def parse_args():
    ap = argparse.ArgumentParser(
        description="HEA composition optimization with the pre-trained "
                    "Mix-GNN models (NSGA-II multi-objective or TPE "
                    "single-objective).")
    ap.add_argument("--method", choices=["nsga2", "tpe"], default="nsga2")
    ap.add_argument("--elements", nargs="+", default=None,
                    help="candidate element pool, e.g. --elements Ti V Cr Mn Fe")
    ap.add_argument("--n-components", type=int, default=None,
                    help="number of components in each candidate alloy")
    ap.add_argument("--frameworks", nargs="+", default=None,
                    choices=["fcc", "bcc", "hcp"],
                    help="lattice frameworks to consider")
    ap.add_argument("--ratio-min", type=float, default=None,
                    help="lower bound of each atomic fraction")
    ap.add_argument("--ratio-max", type=float, default=None,
                    help="upper bound of each atomic fraction "
                         "(default: 1/n-components)")
    ap.add_argument("--out-dir", default=None, help="output directory")

    # NSGA-II knobs
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--popnum", type=int, default=None)
    ap.add_argument("--eta", type=float, default=None,
                    help="mutation distribution parameter")

    # TPE knobs
    ap.add_argument("--property", choices=["bulk_modulus", "thermal_conduct"],
                    default=None, help="target property (tpe method)")
    ap.add_argument("--trials", type=int, default=None, help="number of TPE trials")
    ap.add_argument("--ratios", nargs="+", type=float, default=None,
                    help="tpe only: fix the first n-1 atomic fractions")

    # model overrides
    ap.add_argument("--bm-model", default=None,
                    help="path to the bulk-modulus model .pth")
    ap.add_argument("--ltc-model", default=None,
                    help="path to the lattice-thermal-conductivity model .pth")
    return ap.parse_args()


def apply_model_overrides(mod, args):
    if args.bm_model:
        d, f = split_path_arg(args.bm_model)
        if d:
            mod.MODEL_DIR_BM = d
        mod.MODEL_FILE_BM = f
    if args.ltc_model:
        d, f = split_path_arg(args.ltc_model)
        if d:
            mod.MODEL_DIR_LTC = d
        mod.MODEL_FILE_LTC = f


def run_nsga2(args):
    import nsga_2_for_core_loss as N

    N.configure_run(elements=args.elements, n_components=args.n_components,
                    frameworks=args.frameworks, ratio_min=args.ratio_min,
                    ratio_max=args.ratio_max)
    apply_model_overrides(N, args)
    N.main(generations=args.generations, popnum=args.popnum, eta=args.eta,
           out_dir=args.out_dir, headless=True)


def run_tpe(args):
    import bayes_optimization as B

    if args.elements is not None:
        B.SELECTED_ELEMENTS = args.elements
    if args.n_components is not None:
        B.ELEMENT_NUM = args.n_components
    if args.frameworks is not None:
        B.FRAMEWORKS = args.frameworks
    if args.property is not None:
        B.PROPERTY = args.property
    if args.ratios is not None:
        B.RATIOS = args.ratios
    if args.trials is not None:
        B.N_TRIALS = args.trials
    if args.out_dir is not None:
        B.OUT_DIR = args.out_dir
    apply_model_overrides(B, args)
    B.main()


def main():
    args = parse_args()
    if args.method == "nsga2":
        run_nsga2(args)
    else:
        run_tpe(args)


if __name__ == "__main__":
    main()
