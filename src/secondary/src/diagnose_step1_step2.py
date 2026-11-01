"""
Step 1 & Step 2 Diagnostic and Threshold Tuning Script.
Amazon ML Challenge 2026.

1. DIAGNOSIS on DEV/Validation split:
   - Candidate recall @ 20, 50, 100
   - Loss breakdown:
     * Blocking miss (never generated)
     * Model score below threshold (tau)
     * Removed by mutual exclusivity
     * Cross-script (non-Latin/Indic) vs Latin
   - F0.5 vs tau curve per country (0.30 to 0.80)
   - Empty predictions analysis (predicted empty vs true singletons)

2. PER-COUNTRY THRESHOLD TUNING:
   - Optimal tau for US and India independently
   - Comparison: standard multi-match vs mutual exclusivity vs top-N above tau
"""

import os
import sys
import gc
import time
import pickle
from collections import defaultdict
import numpy as np
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append("src")
from metric import compute_f05_single, evaluate_f05
from infer_baseline import LEGAL_STOPS, GENERIC_ADDR

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

VAL_SPLIT_PATH = "data/splits/val_mini_20k.parquet"
MODEL_PATH = "models/lgbm_reranker.txt"
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"


def get_char_ngrams(text: str, n: int = 3) -> set:
    if not text:
        return set()
    cleaned = "".join(c for c in text.lower() if c.isalnum())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i+n] for i in range(len(cleaned)-n+1)}


def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def run_diagnostics():
    print("=" * 80)
    print("RUNNING STEP 1 (DIAGNOSIS) & STEP 2 (PER-COUNTRY THRESHOLD TUNING)")
    print("=" * 80)
    
    # 1. Load validation set
    val_meta = pl.read_parquet(VAL_SPLIT_PATH)
    val_s1_ids = set(val_meta["source1_entity_id"].to_list())
    
    s1_full = pl.read_csv(TRAIN_S1, separator="\t").filter(
        pl.col("entity_id").is_in(val_s1_ids)
    )
    gt = pl.read_csv(TRAIN_GT, separator="\t").filter(
        pl.col("source1_entity_id").is_in(val_s1_ids)
    )
    
    ground_truth_dict = {}
    for r in gt.iter_rows(named=True):
        mids = r["matched_entity_ids"]
        ground_truth_dict[r["source1_entity_id"]] = set(mids.split(",")) if mids else set()
        
    print(f"Loaded validation set: {len(val_s1_ids)} S1 entities")
    
    booster = lgb.Booster(model_file=MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    
    # Store results per country
    per_country_results = {}
    
    for country in ["India", "US"]:
        print(f"\n" + "-" * 60)
        print(f"ANALYZING COUNTRY: {country}")
        print("-" * 60)
        
        s1_c = s1_full.filter(pl.col("country") == country)
        s1_ids_list = s1_c["entity_id"].to_list()
        n_entities = len(s1_ids_list)
        print(f"S1 count for {country}: {n_entities}")
        
        s2_c = pl.read_csv(TRAIN_S2, separator="\t").filter(pl.col("country") == country)
        s3_c = pl.read_csv(TRAIN_S3, separator="\t").filter(pl.col("country") == country)
        targets_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()
        
        df_tgt = targets_c.select([
            pl.col("entity_id"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"https?://|www\.|[@#\.]|(\.com|\.in|\.org|\.net|\.fr|\.co|cdp)\b", " ").str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("name_clean"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("addr_clean"),
            pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
            pl.col("business_name").fill_null("").alias("raw_name"),
        ])
        del targets_c
        gc.collect()
        
        tgt_ids = df_tgt["entity_id"].to_list()
        tgt_names = df_tgt["name_clean"].to_list()
        tgt_addrs = df_tgt["addr_clean"].to_list()
        tgt_nums_raw = df_tgt["nums"].to_list()
        tgt_raw_names = df_tgt["raw_name"].to_list()
        del df_tgt
        gc.collect()
        
        # Build inverted index
        idx_w2_sorted = defaultdict(list)
        idx_w1 = defaultdict(list)
        idx_w1_num = defaultdict(list)
        idx_aw_num = defaultdict(list)
        
        for i in range(len(tgt_ids)):
            n = tgt_names[i]
            a = tgt_addrs[i]
            words = [w for w in n.split() if len(w) >= 3 and w not in LEGAL_STOPS] if n else []
            nums = [num.lstrip("0") for num in tgt_nums_raw[i] if num.lstrip("0")] if tgt_nums_raw[i] else []
            awords = [w for w in a.split() if len(w) >= 4 and w not in GENERIC_ADDR] if a else []
            
            if len(words) >= 2:
                w_sorted = sorted(words[:3])
                idx_w2_sorted[f"{w_sorted[0]}_{w_sorted[1]}"].append(i)
                if len(words) >= 3:
                    idx_w2_sorted[f"{w_sorted[0]}_{w_sorted[2]}"].append(i)
                    idx_w2_sorted[f"{w_sorted[1]}_{w_sorted[2]}"].append(i)
            if len(words) >= 1:
                idx_w1[words[0]].append(i)
                if nums:
                    for num in nums[:2]:
                        idx_w1_num[f"{words[0]}_{num}"].append(i)
            if awords and nums:
                for aw in awords[:3]:
                    for num in nums[:2]:
                        idx_aw_num[f"{aw}_{num}"].append(i)
                        
        df_s1 = s1_c.select([
            pl.col("entity_id"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"https?://|www\.|[@#\.]|(\.com|\.in|\.org|\.net|\.fr|\.co|cdp)\b", " ").str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("name_clean"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("addr_clean"),
            pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
            pl.col("business_name").fill_null("").alias("raw_name"),
        ])
        del s1_c
        gc.collect()
        
        s1_ids = df_s1["entity_id"].to_list()
        s1_names = df_s1["name_clean"].to_list()
        s1_addrs = df_s1["addr_clean"].to_list()
        s1_nums_raw = df_s1["nums"].to_list()
        s1_raw_names = df_s1["raw_name"].to_list()
        del df_s1
        gc.collect()
        
        # Track recall at K
        total_true_pairs_country = 0
        hits_k20 = 0
        hits_k50 = 0
        hits_k100 = 0
        hits_all_retrieved = 0
        
        # Feature extraction
        all_features = []
        pair_metadata = [] # (sid, tid, is_true, is_cross_script)
        
        for idx in range(len(s1_ids)):
            sid = s1_ids[idx]
            n = s1_names[idx]
            a = s1_addrs[idx]
            rn = s1_raw_names[idx]
            nums = [num.lstrip("0") for num in s1_nums_raw[idx] if num.lstrip("0")] if s1_nums_raw[idx] else []
            words = [w for w in n.split() if len(w) >= 3 and w not in LEGAL_STOPS] if n else []
            awords = [w for w in a.split() if len(w) >= 4 and w not in GENERIC_ADDR] if a else []
            
            true_mids = ground_truth_dict.get(sid, set())
            total_true_pairs_country += len(true_mids)
            
            # Blocking query
            cand_indices = set()
            if len(words) >= 2:
                w_sorted = sorted(words[:3])
                p = idx_w2_sorted.get(f"{w_sorted[0]}_{w_sorted[1]}")
                if p: cand_indices.update(p[:100])
                if len(words) >= 3:
                    p2 = idx_w2_sorted.get(f"{w_sorted[0]}_{w_sorted[2]}")
                    if p2: cand_indices.update(p2[:50])
                    p3 = idx_w2_sorted.get(f"{w_sorted[1]}_{w_sorted[2]}")
                    if p3: cand_indices.update(p3[:50])
            if len(words) >= 1 and nums:
                for num in nums[:2]:
                    p = idx_w1_num.get(f"{words[0]}_{num}")
                    if p: cand_indices.update(p[:50])
            if awords and nums:
                for aw in awords[:3]:
                    for num in nums[:2]:
                        p = idx_aw_num.get(f"{aw}_{num}")
                        if p: cand_indices.update(p[:50])
            if len(cand_indices) < 20 and len(words) >= 1:
                p = idx_w1.get(words[0])
                if p and len(p) <= 300:
                    cand_indices.update(p[:100])
                    
            cand_tids_all = [tgt_ids[c] for c in cand_indices]
            
            # Rank candidates by simple heuristic to compute recall@K
            c_list = list(cand_indices)
            quick_scores = []
            for c_idx in c_list:
                tg_n = tgt_names[c_idx]
                tg_a = tgt_addrs[c_idx]
                ns = fuzz.token_sort_ratio(n, tg_n) if n and tg_n else 0
                as_ = fuzz.token_set_ratio(a, tg_a) if a and tg_a else 0
                quick_scores.append((c_idx, max(ns, as_)))
            quick_scores.sort(key=lambda x: x[1], reverse=True)
            
            ranked_c_idxs = [x[0] for x in quick_scores]
            ranked_tids = [tgt_ids[c] for c in ranked_c_idxs]
            
            for t in true_mids:
                if t in set(cand_tids_all):
                    hits_all_retrieved += 1
                if t in set(ranked_tids[:100]):
                    hits_k100 += 1
                if t in set(ranked_tids[:50]):
                    hits_k50 += 1
                if t in set(ranked_tids[:20]):
                    hits_k20 += 1
                        
            # Use top 25 for LightGBM reranker (same as production pipeline)
            top_for_model = ranked_c_idxs[:25]
            
            s1_ng = get_char_ngrams(n, 3)
            s1_ang = get_char_ngrams(a, 3)
            s1_compact = "".join(c for c in n.lower() if c.isalnum())
            s1_num_set = set(nums)
            s1_is_non_ascii = any(ord(c) > 127 for c in rn)
            
            for c_idx in top_for_model:
                tid = tgt_ids[c_idx]
                tg_n = tgt_names[c_idx]
                tg_a = tgt_addrs[c_idx]
                tg_rn = tgt_raw_names[c_idx]
                
                n_sort = fuzz.token_sort_ratio(n, tg_n) / 100.0 if n and tg_n else 0.0
                n_set = fuzz.token_set_ratio(n, tg_n) / 100.0 if n and tg_n else 0.0
                a_set = fuzz.token_set_ratio(a, tg_a) / 100.0 if a and tg_a else 0.0
                a_sort = fuzz.token_sort_ratio(a, tg_a) / 100.0 if a and tg_a else 0.0
                
                if max(n_sort, a_set) >= 0.35:
                    tg_ng = get_char_ngrams(tg_n, 3)
                    tg_ang = get_char_ngrams(tg_a, 3)
                    n_jacc = jaccard(s1_ng, tg_ng)
                    a_jacc = jaccard(s1_ang, tg_ang)
                else:
                    n_jacc = 0.0
                    a_jacc = 0.0
                    
                tg_nums = [num.lstrip("0") for num in tgt_nums_raw[c_idx] if num.lstrip("0")] if tgt_nums_raw[c_idx] else []
                tg_num_set = set(tg_nums)
                num_overlap = len(s1_num_set & tg_num_set)
                num_jacc = jaccard(s1_num_set, tg_num_set)
                
                is_non_ascii = int(s1_is_non_ascii or any(ord(c) > 127 for c in tg_rn))
                
                tg_compact = "".join(c for c in tg_n.lower() if c.isalnum())
                domain_match = 0
                if s1_compact and tg_compact and (s1_compact in tg_compact or tg_compact in s1_compact):
                    if len(min(s1_compact, tg_compact, key=len)) >= 5:
                        domain_match = 1
                        
                feat = [
                    n_sort, n_set, a_set, a_sort, n_jacc, a_jacc,
                    num_overlap, num_jacc, is_non_ascii, domain_match,
                    abs(len(n) - len(tg_n)), abs(len(a) - len(tg_a))
                ]
                all_features.append(feat)
                is_true = 1 if tid in true_mids else 0
                pair_metadata.append((sid, tid, is_true, is_non_ascii))
                
        # Predict probabilities
        feats_arr = np.array(all_features, dtype=np.float32)
        probs = booster.predict(feats_arr)
        
        # Build pair evaluation table
        pairs_df = []
        for i, (sid, tid, is_true, is_non_ascii) in enumerate(pair_metadata):
            pairs_df.append({
                "s1_id": sid,
                "tgt_id": tid,
                "prob": float(probs[i]),
                "is_true": is_true,
                "is_cross_script": is_non_ascii
            })
            
        print(f"Total True Positive Target Pairs: {total_true_pairs_country}")
        print(f"Candidate Recall @ 20 : {hits_k20 / total_true_pairs_country * 100:.2f}% ({hits_k20}/{total_true_pairs_country})")
        print(f"Candidate Recall @ 50 : {hits_k50 / total_true_pairs_country * 100:.2f}% ({hits_k50}/{total_true_pairs_country})")
        print(f"Candidate Recall @ 100: {hits_k100 / total_true_pairs_country * 100:.2f}% ({hits_k100}/{total_true_pairs_country})")
        print(f"Total Retrieved Recall: {hits_all_retrieved / total_true_pairs_country * 100:.2f}% ({hits_all_retrieved}/{total_true_pairs_country})")
        
        per_country_results[country] = {
            "s1_ids": s1_ids_list,
            "pairs": pairs_df,
            "total_true": total_true_pairs_country,
            "hits_k20": hits_k20,
            "hits_k50": hits_k50,
            "hits_k100": hits_k100,
            "hits_all": hits_all_retrieved,
            "blocking_misses": total_true_pairs_country - hits_all_retrieved,
        }
        
        del tgt_ids, tgt_names, tgt_addrs, tgt_nums_raw, tgt_raw_names, idx_w2_sorted, idx_w1, idx_w1_num, idx_aw_num
        del s1_ids, s1_names, s1_addrs, s1_nums_raw, s1_raw_names, all_features, pair_metadata, feats_arr, probs
        gc.collect()

    print("\n" + "=" * 80)
    print("DETAILED ERROR BREAKDOWN AT TAU = 0.55")
    print("=" * 80)
    
    tau_default = 0.55
    for country in ["India", "US"]:
        res = per_country_results[country]
        total_true = res["total_true"]
        blocking_misses = res["blocking_misses"]
        pairs = res["pairs"]
        
        # Pairs retrieved with prob < 0.55
        below_tau_count = 0
        cross_script_below_tau = 0
        
        # True pairs above tau
        above_tau_true = set()
        for p in pairs:
            if p["is_true"] == 1:
                if p["prob"] < tau_default:
                    below_tau_count += 1
                    if p["is_cross_script"] == 1:
                        cross_script_below_tau += 1
                else:
                    above_tau_true.add((p["s1_id"], p["tgt_id"]))
                    
        # Simulate mutual exclusivity
        sorted_pairs = sorted([p for p in pairs if p["prob"] >= tau_default], key=lambda x: x["prob"], reverse=True)
        assigned_targets = set()
        kept_pairs = set()
        removed_by_excl = 0
        
        for p in sorted_pairs:
            tid = p["tgt_id"]
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                kept_pairs.add((p["s1_id"], tid))
            else:
                if p["is_true"] == 1:
                    removed_by_excl += 1
                    
        print(f"\n--- Country: {country} (Total True Matches = {total_true}) ---")
        print(f"1. Blocking Misses (never generated as candidates): {blocking_misses} ({blocking_misses/total_true*100:.2f}%)")
        print(f"2. Retrieved but Model Prob < {tau_default} (False Rejects): {below_tau_count} ({below_tau_count/total_true*100:.2f}%)")
        print(f"3. Removed by Mutual Exclusivity: {removed_by_excl} ({removed_by_excl/total_true*100:.2f}%)")
        print(f"4. Cross-script component of False Rejects: {cross_script_below_tau} / {below_tau_count} ({cross_script_below_tau/max(1,below_tau_count)*100:.1f}%)")
        print(f"5. Successfully Matched (True Positives): {len(kept_pairs & above_tau_true)} ({len(kept_pairs & above_tau_true)/total_true*100:.2f}%)")

    print("\n" + "=" * 80)
    print("EMPTY PREDICTIONS ANALYSIS (Why 9.8% in Test vs ~5.58% Singletons)")
    print("=" * 80)
    
    for country in ["India", "US"]:
        res = per_country_results[country]
        s1_ids = res["s1_ids"]
        pairs = res["pairs"]
        
        s1_to_preds = defaultdict(set)
        for p in pairs:
            if p["prob"] >= tau_default:
                s1_to_preds[p["s1_id"]].add(p["tgt_id"])
                
        true_singletons = 0
        pred_empty_count = 0
        true_singleton_and_pred_empty = 0
        non_singleton_pred_empty_blocking = 0
        non_singleton_pred_empty_below_tau = 0
        
        s1_to_cands = defaultdict(set)
        for p in pairs:
            s1_to_cands[p["s1_id"]].add(p["tgt_id"])
            
        for sid in s1_ids:
            true_set = ground_truth_dict.get(sid, set())
            pred_set = s1_to_preds.get(sid, set())
            is_true_singleton = (len(true_set) == 0)
            is_pred_empty = (len(pred_set) == 0)
            
            if is_true_singleton:
                true_singletons += 1
            if is_pred_empty:
                pred_empty_count += 1
                if is_true_singleton:
                    true_singleton_and_pred_empty += 1
                else:
                    cands = s1_to_cands.get(sid, set())
                    overlap_with_cands = true_set & cands
                    if len(overlap_with_cands) == 0:
                        non_singleton_pred_empty_blocking += 1
                    else:
                        non_singleton_pred_empty_below_tau += 1
                        
        print(f"\n--- {country} Entity Breakdown (N={len(s1_ids)}) ---")
        print(f"True Singletons (Ground Truth = empty): {true_singletons} ({true_singletons/len(s1_ids)*100:.2f}%)")
        print(f"Total Predicted Empty: {pred_empty_count} ({pred_empty_count/len(s1_ids)*100:.2f}%)")
        print(f"  • True Singletons correctly predicted empty: {true_singleton_and_pred_empty} ({true_singleton_and_pred_empty/len(s1_ids)*100:.2f}%)")
        print(f"  • Non-Singletons wrongly predicted empty due to 100% Blocking Miss: {non_singleton_pred_empty_blocking} ({non_singleton_pred_empty_blocking/len(s1_ids)*100:.2f}%)")
        print(f"  • Non-Singletons wrongly predicted empty because candidates scored < {tau_default}: {non_singleton_pred_empty_below_tau} ({non_singleton_pred_empty_below_tau/len(s1_ids)*100:.2f}%)")

    print("\n" + "=" * 80)
    print("STEP 2: F0.5 vs TAU CURVES & THRESHOLD TUNING")
    print("=" * 80)
    
    tau_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    tuning_results = {}
    for country in ["India", "US"]:
        print(f"\nEvaluating tau grid for {country}...")
        res = per_country_results[country]
        s1_ids = res["s1_ids"]
        pairs = res["pairs"]
        
        tuning_results[country] = []
        
        for tau in tau_grid:
            # 1. Multi-match (allow all above tau)
            preds_multi = defaultdict(set)
            for p in pairs:
                if p["prob"] >= tau:
                    preds_multi[p["s1_id"]].add(p["tgt_id"])
            f05_multi = evaluate_f05(dict(preds_multi), ground_truth_dict, s1_ids)
            
            # 2. Mutual exclusivity
            preds_excl = defaultdict(set)
            sorted_p = sorted([p for p in pairs if p["prob"] >= tau], key=lambda x: x["prob"], reverse=True)
            assigned_t = set()
            for p in sorted_p:
                if p["tgt_id"] not in assigned_t:
                    preds_excl[p["s1_id"]].add(p["tgt_id"])
                    assigned_t.add(p["tgt_id"])
            f05_excl = evaluate_f05(dict(preds_excl), ground_truth_dict, s1_ids)
            
            tuning_results[country].append({
                "tau": tau,
                "f05_multi": f05_multi,
                "f05_excl": f05_excl
            })
            print(f"  {country} | tau={tau:.2f} -> Multi: {f05_multi:.5f} | Excl: {f05_excl:.5f}")

    best_tau_us = max(tuning_results["US"], key=lambda x: x["f05_excl"])
    best_tau_in = max(tuning_results["India"], key=lambda x: x["f05_excl"])
    
    print("\n" + "=" * 80)
    print("THRESHOLD TUNING SUMMARY")
    print("=" * 80)
    print(f"US: Baseline tau=0.55 Excl F0.5 = {[x for x in tuning_results['US'] if x['tau']==0.55][0]['f05_excl']:.5f}")
    print(f"    Optimal tau={best_tau_us['tau']:.2f} Excl F0.5 = {best_tau_us['f05_excl']:.5f} (gain: {best_tau_us['f05_excl'] - [x for x in tuning_results['US'] if x['tau']==0.55][0]['f05_excl']:+.5f})")
    
    print(f"India: Baseline tau=0.55 Excl F0.5 = {[x for x in tuning_results['India'] if x['tau']==0.55][0]['f05_excl']:.5f}")
    print(f"       Optimal tau={best_tau_in['tau']:.2f} Excl F0.5 = {best_tau_in['f05_excl']:.5f} (gain: {best_tau_in['f05_excl'] - [x for x in tuning_results['India'] if x['tau']==0.55][0]['f05_excl']:+.5f})")
    
    # Combined weighted CV calculation
    base_us = [x for x in tuning_results['US'] if x['tau']==0.55][0]['f05_excl']
    base_in = [x for x in tuning_results['India'] if x['tau']==0.55][0]['f05_excl']
    
    w_us = 0.383 / (0.383 + 0.468)
    w_in = 0.468 / (0.383 + 0.468)
    
    cv_base = w_us * base_us + w_in * base_in
    cv_tuned = w_us * best_tau_us['f05_excl'] + w_in * best_tau_in['f05_excl']
    print(f"\nWeighted Normalized CV (US+India):")
    print(f"  Baseline (tau=0.55): {cv_base:.5f}")
    print(f"  Tuned per-country : {cv_tuned:.5f} (gain: {cv_tuned - cv_base:+.5f})")
    
    multi_us = best_tau_us['f05_multi']
    multi_in = best_tau_in['f05_multi']
    cv_multi = w_us * multi_us + w_in * multi_in
    print(f"  Multi-match at tuned tau: {cv_multi:.5f} (difference vs mutual excl: {cv_multi - cv_tuned:+.5f})")
    print("\nDIAGNOSTICS & TUNING COMPLETE.")


if __name__ == "__main__":
    run_diagnostics()
