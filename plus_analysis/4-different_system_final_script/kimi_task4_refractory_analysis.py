# -*- coding: utf-8 -*-
"""
kimi_task4_refractory_analysis.py
Task 4, refractory part: the refractory alloys are the quinary BCC alloys of
task 2 (plus/2-Mo-Ti-Os-Ir_plus/BCC, filenames carry raw model outputs).
Group statistics for the presence/absence of Mo, W, Ti:

  for element E in {Mo, W, Ti}:
      group_with    = alloys containing E
      group_without = alloys without E
      -> mean/std of predicted B and kappa_l in each group
      -> relative change  dB% = (mean_with - mean_without)/mean_without

Output: ../final_data/task4_refractory_element_effect.csv
The comparison figure is rendered by kimi_task4_plot.py once the literature
values (final_data/kimi_task4_literature.csv) are available.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BCC_DIR = os.path.abspath(os.path.join(TASK4_DIR, "..", "2-Mo-Ti-Os-Ir_plus", "BCC"))

TARGETS = ["Mo", "W", "Ti"]
# BCC linear mapping (value_mapping_example.py)
BM_B, BM_A = 2.38, 0.13
LTC_B, LTC_A = 5.03, 1.28


def parse_poscar_header(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()[:10]]
    elements = lines[5].split()
    counts = [int(float(x)) for x in lines[6].split()]
    return elements, counts


def main():
    rows = []
    for fname in os.listdir(BCC_DIR):
        if not fname.startswith("POSCAR"):
            continue
        parts = fname.split("_")
        bm_raw, ltc_raw = float(parts[-2]), float(parts[-1])
        elements, counts = parse_poscar_header(os.path.join(BCC_DIR, fname))
        total = sum(counts)
        frac = {e: c / total for e, c in zip(elements, counts)}
        row = {"file": fname,
               "bm": (bm_raw + BM_B) / BM_A,
               "ltc": (ltc_raw + LTC_B) / LTC_A}
        for e in TARGETS:
            row[f"frac_{e}"] = frac.get(e, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"BCC alloys: {len(df)}")
    for e in TARGETS:
        print(f"  with {e}: {(df[f'frac_{e}'] > 0).sum()}, without: {(df[f'frac_{e}'] == 0).sum()}")

    out_rows = []
    for e in TARGETS:
        gw = df[df[f"frac_{e}"] > 0]
        go = df[df[f"frac_{e}"] == 0]
        rec = {"element": e, "n_with": len(gw), "n_without": len(go),
               "bm_with": gw["bm"].mean(), "bm_with_std": gw["bm"].std(),
               "bm_without": go["bm"].mean(), "bm_without_std": go["bm"].std(),
               "ltc_with": gw["ltc"].mean(), "ltc_with_std": gw["ltc"].std(),
               "ltc_without": go["ltc"].mean(), "ltc_without_std": go["ltc"].std()}
        rec["d_bm_pct"] = 100.0 * (rec["bm_with"] - rec["bm_without"]) / rec["bm_without"]
        rec["d_ltc_pct"] = 100.0 * (rec["ltc_with"] - rec["ltc_without"]) / rec["ltc_without"]
        # error of the percentage change (independent means)
        rec["d_bm_pct_err"] = 100.0 * np.sqrt(
            (rec["bm_with_std"] / np.sqrt(max(len(gw), 1)))**2 +
            (rec["bm_without_std"] / np.sqrt(max(len(go), 1)))**2) / rec["bm_without"]
        rec["d_ltc_pct_err"] = 100.0 * np.sqrt(
            (rec["ltc_with_std"] / np.sqrt(max(len(gw), 1)))**2 +
            (rec["ltc_without_std"] / np.sqrt(max(len(go), 1)))**2) / rec["ltc_without"]
        out_rows.append(rec)
        print(f"{e}: B {rec['bm_without']:.1f} -> {rec['bm_with']:.1f} GPa "
              f"({rec['d_bm_pct']:+.2f}%), kappa {rec['ltc_without']:.2f} -> "
              f"{rec['ltc_with']:.2f} W/mK ({rec['d_ltc_pct']:+.2f}%)")

    out_df = pd.DataFrame(out_rows)
    out_dir = os.path.join(TASK4_DIR, "final_data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "task4_refractory_element_effect.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
