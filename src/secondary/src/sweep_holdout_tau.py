"""
Fast Threshold Sweep on Cached Holdout Cross-Encoder Probabilities.
Takes raw scored pairs from output/holdout_cross_encoder/{country}_scored_pairs.parquet,
sweeps tau in [0.40, 0.90] with global target-level mutual exclusivity,
and finds optimal (tau_us, tau_in) that maximizes Weighted Macro F0.5.
"""
import os
import sys
import json
import time
import numpy as np
import polars as pl
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from metric import compute_f05_single

HOLDOUT_DIR = "data/calibrated_holdout"
CACHE_DIR = "output/holdout_cross_encoder"
GT_PATH = os.path.join(HOLDOUT_DIR, "ground_truth.tsv")

def sweep_country(country: str, step=0.02):
    parquet_path = os.path.join(CACHE_DIR, f"{country}_scored_pairs.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} does not exist yet.", flush=True)
        return None
        
    print(f"\n--- Loading scored pairs for {country} ---", flush=True)
    df = pl.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} pairs. Sorting by score...", flush=True)
    df = df.sort("score", descending=True)
    
    # Load S1 IDs
    s1_df = pl.read_csv(os.path.join(HOLDOUT_DIR, "test_source1.tsv"), separator="\t").filter(
        pl.col("country") == country
    )
    s1_ids = s1_df["entity_id"].to_list()
    
    # Load GT
    gt_df = pl.scan_csv(GT_PATH, separator="\t").collect()
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m = r["matched_entity_ids"]
        gt_dict[sid] = set(x.strip() for x in m.split(",")) if m else set()
        
    pairs = list(zip(df["score"].to_list(), df["s1_id"].to_list(), df["tgt_id"].to_list()))
    del df
    
    print(f"Starting sweep for {country} across tau in [0.40, 0.90]...")
    best_tau = 0.50
    best_f05 = -1.0
    best_p = 0.0
    best_r = 0.0
    
    tau_results = {}
    for tau in np.arange(0.40, 0.92, step):
        tau = round(float(tau), 2)
        assigned_targets = set()
        s1_matches = defaultdict(list)
        
        for sc, sid, tid in pairs:
            if sc < tau:
                break
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                s1_matches[sid].append(tid)
                
        f_scores, p_scores, r_scores = [], [], []
        empty_count = 0
        for sid in s1_ids:
            true_t = gt_dict.get(sid, set())
            pred_t = set(s1_matches.get(sid, []))
            if not pred_t:
                empty_count += 1
            f_scores.append(compute_f05_single(pred_t, true_t))
            if not true_t:
                sc = 1.0 if not pred_t else 0.0
                p_scores.append(sc)
                r_scores.append(sc)
            else:
                tp = len(true_t & pred_t)
                fp = len(pred_t - true_t)
                fn = len(true_t - pred_t)
                p_scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                r_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                
        m_f05 = float(np.mean(f_scores))
        m_p = float(np.mean(p_scores))
        m_r = float(np.mean(r_scores))
        tau_results[tau] = {"f05": m_f05, "precision": m_p, "recall": m_r, "matches": len(assigned_targets)}
        
        print(f"  tau={tau:.2f} -> Macro F0.5={m_f05:.5f} (P={m_p:.5f}, R={m_r:.5f}) | Matches={len(assigned_targets):,}")
        if m_f05 > best_f05:
            best_f05 = m_f05
            best_tau = tau
            best_p = m_p
            best_r = m_r
            
    print(f"\n[BEST {country}] tau* = {best_tau:.2f} | Macro F0.5 = {best_f05:.5f} (P={best_p:.5f}, R={best_r:.5f})")
    return {"country": country, "best_tau": best_tau, "best_f05": best_f05, "precision": best_p, "recall": best_r, "all": tau_results}

if __name__ == "__main__":
    us_out = sweep_country("US")
    in_out = sweep_country("India")
    
    if us_out and in_out:
        w_f05 = (0.383 * us_out["best_f05"] + 0.468 * in_out["best_f05"]) / (0.383 + 0.468)
        print("\n" + "=" * 70)
        print("OPTIMAL CALIBRATED HOLDOUT RESULT:")
        print(f"  US:    tau*={us_out['best_tau']:.2f} -> F0.5={us_out['best_f05']:.5f}")
        print(f"  India: tau*={in_out['best_tau']:.2f} -> F0.5={in_out['best_f05']:.5f}")
        print(f"  Weighted Macro F0.5 = {w_f05:.5f}")
        print("=" * 70)
