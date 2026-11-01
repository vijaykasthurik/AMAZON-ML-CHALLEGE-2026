"""
Step B5: Re-tune per-country tau on calibrated DEV split (5.75 density).
Evaluates threshold sweep tau in [0.40, 0.95] in steps of 0.02 for:
  - US
  - India
Tests:
  1. Global mutual exclusivity vs per-S1 mutual exclusivity
  2. Optimal tau* per country
  3. F0.5 vs tau curves, precision, recall, and empty prediction percentage
"""
import os
import sys
import time
import numpy as np
import polars as pl
from collections import defaultdict
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from metric import compute_f05_single
from infer_ensemble_optimized import (
    run_country_inference, LGBM_MODEL_PATH, CB_MODEL_PATH
)

def evaluate_tau_sweep(country: str, density="5.75"):
    split_dir = f"data/calibrated_dev/{country}_density_{density}"
    print(f"\n{'='*70}")
    print(f"RUNNING SCORING & TAU SWEEP: {country} (Density={density})")
    print(f"{'='*70}")
    
    # Load GT
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    gt_dict = {}
    total_positives = 0
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        t_set = set(x.strip() for x in m.split(",")) if m else set()
        gt_dict[r["source1_entity_id"]] = t_set
        total_positives += len(t_set)
        
    s1_all_scores = {}
    out_m = {}
    out_c = {}
    
    lgbm_booster = lgb.Booster(model_file=LGBM_MODEL_PATH)
    cb_booster = CatBoostClassifier()
    cb_booster.load_model(CB_MODEL_PATH)
    
    run_country_inference(
        country, lgbm_booster, cb_booster, out_m, out_c, split_dir,
        override_tau=0.0, # collect all scored candidates
        out_all_candidate_scores=s1_all_scores
    )
    
    s1_ids = list(gt_dict.keys())
    print(f"Collected raw candidate probabilities for {len(s1_all_scores):,} S1 entities.")
    
    # Flatten all scored pairs: (prob, sid, tg_id)
    all_pairs = []
    for sid, cands in s1_all_scores.items():
        for tg_id, p in cands:
            all_pairs.append((p, sid, tg_id))
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n--- Strategy 1: Global Mutual Exclusivity (Greedy Target Assignment) ---")
    print(f"{'tau':<6s} | {'Macro F0.5':<12s} | {'Precision':<10s} | {'Recall':<10s} | {'Empty %':<10s} | {'Matched Pairs':<15s}")
    print("-" * 70)
    
    best_tau_global = 0.50
    best_f05_global = -1.0
    best_p_global = 0.0
    best_r_global = 0.0
    
    for tau in np.arange(0.40, 0.96, 0.02):
        tau = round(float(tau), 2)
        used_targets = set()
        s1_matches = defaultdict(list)
        
        for prob, sid, tg_id in all_pairs:
            if prob < tau:
                break
            if tg_id not in used_targets:
                s1_matches[sid].append(tg_id)
                used_targets.add(tg_id)
                
        f05_list, p_list, r_list = [], [], []
        empty_count = 0
        for sid in s1_ids:
            t_set = gt_dict.get(sid, set())
            m_set = set(s1_matches.get(sid, []))
            if not m_set:
                empty_count += 1
            f05_list.append(compute_f05_single(m_set, t_set))
            if len(t_set) == 0:
                sc = 1.0 if len(m_set) == 0 else 0.0
                p_list.append(sc)
                r_list.append(sc)
            else:
                tp = len(t_set & m_set)
                fp = len(m_set - t_set)
                fn = len(t_set - m_set)
                p_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                r_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                
        m_f05 = np.mean(f05_list)
        m_p = np.mean(p_list)
        m_r = np.mean(r_list)
        empty_pct = (empty_count / len(s1_ids)) * 100
        
        print(f"{tau:<6.2f} | {m_f05:<12.5f} | {m_p:<10.5f} | {m_r:<10.5f} | {empty_pct:<9.1f}% | {len(used_targets):<15d}")
        
        if m_f05 > best_f05_global:
            best_f05_global = m_f05
            best_tau_global = tau
            best_p_global = m_p
            best_r_global = m_r
            
    print(f"\n[GLOBAL WINNER] {country}: tau* = {best_tau_global:.2f} | Macro F0.5 = {best_f05_global:.5f} (P={best_p_global:.5f}, R={best_r_global:.5f})")
    
    return {
        "country": country,
        "best_tau": best_tau_global,
        "best_f05": best_f05_global,
        "precision": best_p_global,
        "recall": best_r_global,
    }

if __name__ == "__main__":
    us_res = evaluate_tau_sweep("US", "5.75")
    in_res = evaluate_tau_sweep("India", "5.75")
    
    weighted_f05 = (0.383 * us_res["best_f05"] + 0.468 * in_res["best_f05"]) / (0.383 + 0.468)
    print(f"\n{'='*70}")
    print(f"STEP B5 FINAL MEASURED RESULTS (Calibrated DEV Folds):")
    print(f"  US:    Optimal tau* = {us_res['best_tau']:.2f} -> Macro F0.5 = {us_res['best_f05']:.5f} (P={us_res['precision']:.5f}, R={us_res['recall']:.5f})")
    print(f"  India: Optimal tau* = {in_res['best_tau']:.2f} -> Macro F0.5 = {in_res['best_f05']:.5f} (P={in_res['precision']:.5f}, R={in_res['recall']:.5f})")
    print(f"  Blended (US + India): Macro F0.5 = {weighted_f05:.5f}")
    print(f"{'='*70}")
