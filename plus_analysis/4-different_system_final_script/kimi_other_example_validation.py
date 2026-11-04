# -*- coding: utf-8 -*-
"""
kimi_other_example_validation.py — validate the model's element add/remove
trends against ADDITIONAL refractory HEA literature series (beyond Tseng 2019).

Pairs tested (all BCC, equiatomic unless noted):
  1. NbMoTaW -> NbMoTaWV  (+V)   [Hu 2020 JAC 827:153963; Senkov 2011]
  2. NbMoTaW -> TiNbMoTaW  (+Ti)  [Han 2018 MSEA 712:380]
  3. HfNbTaTiZr -> HfNbTaTiZrW (+W)   [Wang 2019 JAC 803:778]
  4. HfNbTaTiZr -> HfMoNbTaTiZrW (+Mo+W) [Wang 2019]
  5. NbTaV -> NbTaTiV -> NbTaTiVW  (+Ti, then +W) [Yao 2016 MSEA 674:203]

Output: ../other_example/validation_results.csv + fig_other_refractory_series.png
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
os.chdir(TASK1_SCRIPT_DIR)

from kimi_inference_core import build_random_structure, build_datasets, \
    predict_from_data, map_value                                   # noqa: E402

N_SEEDS = 5

# alloy name -> element list (equiatomic)
ALLOYS = {
    "NbMoTaW": ["Nb", "Mo", "Ta", "W"],
    "NbMoTaWV": ["Nb", "Mo", "Ta", "W", "V"],
    "TiNbMoTaW": ["Ti", "Nb", "Mo", "Ta", "W"],
    "HfNbTaTiZr": ["Hf", "Nb", "Ta", "Ti", "Zr"],
    "HfNbTaTiZrW": ["Hf", "Nb", "Ta", "Ti", "Zr", "W"],
    "HfMoNbTaTiZrW": ["Hf", "Mo", "Nb", "Ta", "Ti", "Zr", "W"],
    "NbTaV": ["Nb", "Ta", "V"],
    "NbTaTiV": ["Nb", "Ta", "Ti", "V"],
    "NbTaTiVW": ["Nb", "Ta", "Ti", "V", "W"],
}

# (pair_name, base, variant, element, literature direction note)
PAIRS = [
    ("+V (NbMoTaW)", "NbMoTaW", "NbMoTaWV", "V",
     "Hu2020 B -4.5% (VCA); Senkov2011 YS +18%, HV +21%"),
    ("+Ti (NbMoTaW)", "NbMoTaW", "TiNbMoTaW", "Ti",
     "Han2018 YS +46%, ductility up"),
    ("+W (HfNbTaTiZr)", "HfNbTaTiZr", "HfNbTaTiZrW", "W",
     "Wang2019 YS@1200C +275%"),
    ("+Mo+W (HfNbTaTiZr)", "HfNbTaTiZr", "HfMoNbTaTiZrW", "Mo,W",
     "Wang2019 YS@1200C +664%"),
    ("+Ti (NbTaV)", "NbTaV", "NbTaTiV", "Ti", "Yao2016"),
    ("+W (NbTaTiV)", "NbTaTiV", "NbTaTiVW", "W", "Yao2016 YS 1420 MPa"),
]


def main():
    import zlib
    structures, metas = [], []
    for name, elements in ALLOYS.items():
        ratios = [1.0 / len(elements)] * len(elements)
        for seed in range(N_SEEDS):
            sd = zlib.crc32(f"other_{name}_{seed}".encode()) % 100000
            s = build_random_structure(elements, ratios, "bcc", seed=sd)
            structures.append(s)
                # save one representative POSCAR for inspection
            if seed == 0:
                d = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "other_example", name))
                os.makedirs(d, exist_ok=True)
                s.get_sorted_structure().to(os.path.join(d, "POSCAR"), fmt="poscar")
            metas.append((name, seed, len(elements)))
    print(f"built {len(structures)} structures")
    datasets = build_datasets(structures, n_workers=8)
    bm_raw = predict_from_data(datasets["bm"], "bm")
    ltc_raw = predict_from_data(datasets["ltc"], "ltc")

    rows = []
    for (name, seed, nsp), br, lr in zip(metas, bm_raw, ltc_raw):
        rows.append({"alloy": name, "seed": seed, "n_species": nsp,
                     "bm_GPa": map_value(br, "bm", "bcc", nsp),
                     "ltc_WmK": map_value(lr, "ltc", "bcc", nsp)})
    df = pd.DataFrame(rows)
    agg = df.groupby("alloy").agg(bm=("bm_GPa", "mean"), bm_std=("bm_GPa", "std"),
                                  ltc=("ltc_WmK", "mean"), ltc_std=("ltc_WmK", "std")).reset_index()

    out_rows = []
    for pair, base, variant, ele, lit_note in PAIRS:
        b0 = agg.loc[agg["alloy"] == base, "bm"].iloc[0]
        b1 = agg.loc[agg["alloy"] == variant, "bm"].iloc[0]
        l0 = agg.loc[agg["alloy"] == base, "ltc"].iloc[0]
        l1 = agg.loc[agg["alloy"] == variant, "ltc"].iloc[0]
        out_rows.append({"pair": pair, "base": base, "variant": variant,
                         "bm_base": round(b0, 2), "bm_variant": round(b1, 2),
                         "d_bm_pct": round(100 * (b1 - b0) / b0, 2),
                         "ltc_base": round(l0, 2), "ltc_variant": round(l1, 2),
                         "d_ltc_pct": round(100 * (l1 - l0) / l0, 2),
                         "literature": lit_note})
    out = pd.DataFrame(out_rows)
    out_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "other_example"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "validation_results.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    agg.to_csv(os.path.join(out_dir, "alloy_predictions.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
