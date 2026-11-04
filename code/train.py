# -*- coding: utf-8 -*-
"""
train.py — command-line entry for Mix-GNN model training (final training loop).

Task division:
  - model architecture and training pipeline: (author)
  - CLI packaging: Kimi

Example
-------
    python code/train.py --property thermal_conduct --data-dir ./data_set/thermal_conduct \
        --save-root ./runs/ltc --lr 1e-3 --batch-size 2048 --max-epochs 12000

The data directory must contain the two products of the data-processing chain
(see README): features_normalized_<property>_log.csv and
cgcnn-voronoi-<property>-HEA.pt.
"""
import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # training curves are saved to disk, never shown

_CODE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR / "training"))

import mix_model_regress_loop as M  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train the Mix-GNN property-prediction model "
                    "(the loop that produced the released pre-trained models).")
    ap.add_argument("--property", choices=["bulk_modulus", "thermal_conduct"],
                    required=True, help="target property")
    ap.add_argument("--data-dir", required=True,
                    help="folder with features_normalized_<property>_log.csv "
                         "and cgcnn-voronoi-<property>-HEA.pt")
    ap.add_argument("--save-root", default=None,
                    help="output root (default: ./runs/<property>/result/)")
    ap.add_argument("--lr", type=float, default=M.LR, help="learning rate")
    ap.add_argument("--weight-decay", type=float, default=M.WEIGHT_DECAY)
    ap.add_argument("--batch-size", type=int, default=M.BATCH_SIZE)
    ap.add_argument("--train-ratio", type=float, default=M.TRAIN_RATIO,
                    help="train/test split ratio")
    ap.add_argument("--min-epochs", type=int, default=M.MIN_EPOCHS,
                    help="train at least this many epochs per outer loop")
    ap.add_argument("--max-epochs", type=int, default=M.MAX_EPOCHS,
                    help="hard cap of epochs per outer loop")
    ap.add_argument("--r2-target", type=float, default=M.R2_TARGET,
                    help="stop an outer loop once test R^2 (2 dp) reaches this")
    ap.add_argument("--r2-save", type=float, default=M.R2_SAVE,
                    help="save the model once test R^2 (2 dp) reaches this")
    ap.add_argument("--max-loops", type=int, default=M.MAX_LOOPS,
                    help="cap on outer loops (re-initializations)")
    ap.add_argument("--max-models", type=int, default=M.MAX_MODELS,
                    help="stop after saving this many qualified models")
    return ap.parse_args()


def main():
    args = parse_args()
    save_root = (args.save_root
                 or os.path.join("./runs", args.property, "result"))
    save_root = os.path.join(save_root, "")  # normalize trailing separator

    M.PROPERTY = args.property
    M.INPUT_PATH = args.data_dir
    M.SAVE_ROOT = save_root
    M.SAVE_TEMP = os.path.join(save_root, "model_temp", "")
    M.LR = args.lr
    M.WEIGHT_DECAY = args.weight_decay
    M.BATCH_SIZE = args.batch_size
    M.TRAIN_RATIO = args.train_ratio
    M.MIN_EPOCHS = args.min_epochs
    M.MAX_EPOCHS = args.max_epochs
    M.R2_TARGET = args.r2_target
    M.R2_SAVE = args.r2_save
    M.MAX_LOOPS = args.max_loops
    M.MAX_MODELS = args.max_models

    print("Training configuration:")
    for k in ("PROPERTY", "INPUT_PATH", "SAVE_ROOT", "LR", "WEIGHT_DECAY",
              "BATCH_SIZE", "TRAIN_RATIO", "MIN_EPOCHS", "MAX_EPOCHS",
              "R2_TARGET", "R2_SAVE", "MAX_LOOPS", "MAX_MODELS"):
        print("  {} = {}".format(k, getattr(M, k)))
    M.main()


if __name__ == "__main__":
    main()
