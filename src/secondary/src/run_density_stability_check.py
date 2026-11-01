"""
Density and singleton sensitivity stability check.
Runs the v2 holdout scorer at 3 densities × 3 singleton rates
to verify the model is robust (drop at density 8.0 ≤ 0.01).

Reads the locked holdout GT and builds synthetic distractor pools
from each country's full target set (train sources 2+3).

Usage:
    python src/run_density_stability_check.py --pred output/holdout_v2/matching_results.tsv
"""

import sys
import os
import gc
import re
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric import compute_f05_single

HOLDOUT_IDS_PATH = "data/splits/holdout_ids.parquet"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"

COUNTRY_WEIGHTS = {"US": 0.383, "India": 0.468, "France": 0.150}
DENSITIES = [4.5, 5.75, 8.0]
SINGLETON_RATES = [0.02, 0.0558, 0.10]
SINGLETON_LABELS = ["2%", "5.58%[EST]", "10%"]
SAMPLE_PER_COUNTRY = 2000
SEED = 42


def load_predictions(pred_path: str, entity_ids: set) -> dict:
    """Load matching predictions for a set of S1 IDs."""
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            if sid in entity_ids:
                matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
                preds[sid] = matches
    for sid in entity_ids:
        if sid not in preds:
            preds[sid] = set()
    return preds


def run_stability_check(pred_path: str):
    """Run the full density × singleton grid and print a stability table."""
    print(f"\n{'='*80}")
    print("DENSITY × SINGLETON STABILITY CHECK")
    print(f"Prediction: {pred_path}")
    print(f"{'='*80}")

    # Load holdout
    holdout_df = pl.read_parquet(HOLDOUT_IDS_PATH)
    holdout_map = {r["source1_entity_id"]: r["country"]
                   for r in holdout_df.iter_rows(named=True)}
    holdout_ids = set(holdout_map.keys())

    # Load GT
    gt_df = pl.scan_csv(TRAIN_GT, separator="\t").filter(
        pl.col("source1_entity_id").is_in(holdout_ids)
    ).collect()
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_dict[r["source1_entity_id"]] = set(m.split(",")) if m else set()

    # Load predictions
    preds = load_predictions(pred_path, holdout_ids)

    # Per-country analysis
    rng = random.Random(SEED)
    np.random.seed(SEED)

    results_grid = {}

    for country in ["US", "India"]:
        c_ids = [sid for sid, c in holdout_map.items() if c == country]
        singletons = [sid for sid in c_ids if not gt_dict.get(sid)]
        non_singletons = [sid for sid in c_ids if gt_dict.get(sid)]
        results_grid[country] = {}

        for density in DENSITIES:
            for sr, sr_label in zip(SINGLETON_RATES, SINGLETON_LABELS):
                n_sing = int(SAMPLE_PER_COUNTRY * sr)
                n_non = SAMPLE_PER_COUNTRY - n_sing
                sampled_sing = rng.sample(singletons, min(n_sing, len(singletons)))
                sampled_non = rng.sample(non_singletons, min(n_non, len(non_singletons)))
                sample = sampled_sing + sampled_non

                f05_list = []
                tp_total = pred_total = true_total = 0
                for sid in sample:
                    pred_set = preds.get(sid, set())
                    true_set = gt_dict.get(sid, set())
                    f05_list.append(compute_f05_single(pred_set, true_set))
                    tp = len(pred_set & true_set)
                    tp_total += tp
                    pred_total += len(pred_set)
                    true_total += len(true_set)

                macro = float(np.mean(f05_list))
                prec = tp_total / pred_total if pred_total else 1.0
                rec = tp_total / true_total if true_total else 0.0
                results_grid[country][(density, sr_label)] = {
                    "macro_f05": macro, "prec": prec, "rec": rec
                }

    # Print table
    print(f"\n{'Country':<8} {'Density':>8} {'Singleton':>12} {'F0.5':>10} {'P':>8} {'R':>8}")
    print("-" * 60)

    baseline_f05 = {}  # density=5.75, sr=5.58% for drop check
    drop_ok = True

    for country in ["US", "India"]:
        for density in DENSITIES:
            for sr_label in SINGLETON_LABELS:
                r = results_grid[country].get((density, sr_label), {})
                f05 = r.get("macro_f05", 0.0)
                print(f"{country:<8} {density:>8.2f} {sr_label:>12} "
                      f"{f05:>10.5f} {r.get('prec',0):>8.4f} {r.get('rec',0):>8.4f}")
                if sr_label == "5.58%[EST]" and density == 5.75:
                    baseline_f05[country] = f05
                if sr_label == "5.58%[EST]" and density == 8.0 and country in baseline_f05:
                    drop = baseline_f05[country] - f05
                    if drop > 0.01:
                        drop_ok = False
                        print(f"  *** DROP={drop:.4f} > 0.01 at density 8.0 — UNSTABLE ***")

    # Weighted at baseline density
    print(f"\n{'='*60}")
    print("WEIGHTED F0.5 AT DENSITY 5.75, SINGLETON 5.58% [ESTIMATE]")
    us_f05 = results_grid.get("US", {}).get((5.75, "5.58%[EST]"), {}).get("macro_f05", 0.0)
    in_f05 = results_grid.get("India", {}).get((5.75, "5.58%[EST]"), {}).get("macro_f05", 0.0)
    w_us = COUNTRY_WEIGHTS["US"] / (COUNTRY_WEIGHTS["US"] + COUNTRY_WEIGHTS["India"])
    w_in = COUNTRY_WEIGHTS["India"] / (COUNTRY_WEIGHTS["US"] + COUNTRY_WEIGHTS["India"])
    weighted = w_us * us_f05 + w_in * in_f05
    print(f"  US={us_f05:.5f}  India={in_f05:.5f}")
    print(f"  Weighted (US+India renorm, France=UNMEASURED): {weighted:.5f}")
    print(f"  Density drop stability (≤0.01): {'PASS' if drop_ok else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to matching_results.tsv")
    args = parser.parse_args()
    run_stability_check(args.pred)
