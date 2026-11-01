"""
Evaluate candidate generation & model scoring on Calibrated DEV Split (5.75 density).
Measures:
  1. Candidate recall ceiling
  2. Average, median, max candidates per S1
  3. Reduction ratio
  4. Precision, Recall, and Macro F0.5
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("src")
from metric import compute_f05_single
from infer_ensemble_optimized import run_country_inference, LGBM_MODEL_PATH, CB_MODEL_PATH

def evaluate_calibrated_country(country="India", density="5.75"):
    split_dir = f"data/calibrated_dev/{country}_density_{density}"
    print(f"\n{'='*70}")
    print(f"EVALUATING CALIBRATED DEV: {country} (Density={density})")
    print(f"{'='*70}")
    
    # Load Ground Truth
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_dict[r["source1_entity_id"]] = set(m.split(",")) if m else set()
        
    lgbm_booster = lgb.Booster(model_file=LGBM_MODEL_PATH)
    cb_booster = CatBoostClassifier()
    cb_booster.load_model(CB_MODEL_PATH)
    
    out_matching_map = {}
    out_candidate_map = {}
    
    t0 = time.time()
    run_country_inference(country, lgbm_booster, cb_booster, out_matching_map, out_candidate_map, split_dir)
    elapsed = time.time() - t0
    
    # Compute candidate stats
    s1_ids = list(out_matching_map.keys())
    cand_counts = [len(out_candidate_map.get(s, [])) for s in s1_ids]
    
    # Compute recall ceiling (fraction of true targets present in candidate set)
    true_targets_total = 0
    true_targets_in_candidates = 0
    for s in s1_ids:
        t_set = gt_dict.get(s, set())
        c_set = set(out_candidate_map.get(s, []))
        true_targets_total += len(t_set)
        true_targets_in_candidates += len(t_set & c_set)
        
    recall_ceiling = true_targets_in_candidates / true_targets_total if true_targets_total > 0 else 1.0
    
    # Compute F0.5, P, R
    f05_list, p_list, r_list = [], [], []
    for s in s1_ids:
        t_set = gt_dict.get(s, set())
        m_set = set(out_matching_map.get(s, []))
        score = compute_f05_single(m_set, t_set)
        f05_list.append(score)
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
            
    print(f"\n--- RESULTS: {country} (Density {density}) ---")
    print(f"  Candidate Recall Ceiling : {recall_ceiling*100:.2f}%")
    print(f"  Avg Candidates per S1    : {np.mean(cand_counts):.2f}")
    print(f"  Median Candidates per S1 : {np.median(cand_counts):.1f}")
    print(f"  Max Candidates per S1    : {np.max(cand_counts)}")
    print(f"  Precision                : {np.mean(p_list):.5f}")
    print(f"  Recall                   : {np.mean(r_list):.5f}")
    print(f"  Macro F0.5               : {np.mean(f05_list):.5f}")
    print(f"  Inference speed          : {len(s1_ids)/elapsed:.1f} S1/sec ({elapsed:.1f}s total)")
    
    return {
        "country": country,
        "recall_ceiling": recall_ceiling,
        "avg_cands": np.mean(cand_counts),
        "f05": np.mean(f05_list),
        "precision": np.mean(p_list),
        "recall": np.mean(r_list),
    }

if __name__ == "__main__":
    us_res = evaluate_calibrated_country("US", "5.75")
    in_res = evaluate_calibrated_country("India", "5.75")
    
    weighted_f05 = (0.383 * us_res["f05"] + 0.468 * in_res["f05"]) / (0.383 + 0.468)
    print(f"\n=======================================================")
    print(f"CALIBRATED DEV (5.75 DENSITY) WEIGHTED F0.5: {weighted_f05:.5f}")
    print(f"=======================================================")
