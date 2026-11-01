"""
Track C1: Build Cross-Encoder Training Pairs from Calibrated DEV Splits.
Extracts:
  - All True Positive Pairs (label=1.0) from ground_truth.tsv
  - Hard Negative Pairs (label=0.0): Top candidate distractors from hybrid retrieval
  - Formatted text:
      query: "{business_name} | {business_address} | {country}"
      passage: "{business_name} | {business_address} | {country}"
  - 5-fold GroupKFold by source1_entity_id (0 leakage)
Saves:
  - data/cross_encoder_pairs.parquet
  - kaggle_dataset/cross_encoder_pairs.parquet
"""
import os
import sys
import time
import numpy as np
import polars as pl
from collections import defaultdict
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from infer_ensemble_optimized import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    get_char_ngrams, jaccard, name_tokens_no_stops,
    query_index, build_inverted_index,
)
from rapidfuzz import fuzz

def build_pairs_for_country(country: str, density="5.75"):
    split_dir = f"data/calibrated_dev/{country}_density_{density}"
    print(f"\n[C1] Generating cross-encoder pairs for {country} (Density={density})...", flush=True)
    t0 = time.time()
    
    s1_df = pl.read_csv(os.path.join(split_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(split_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(split_dir, "test_source3.tsv"), separator="\t")
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_dict[r["source1_entity_id"]] = set(x.strip() for x in m.split(",")) if m else set()
        
    s23_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    target_ids = s23_df["entity_id"].to_list()
    target_names_raw = [r["business_name"] or "" for r in s23_df.iter_rows(named=True)]
    target_addrs_raw = [r["business_address"] or "" for r in s23_df.iter_rows(named=True)]
    
    target_info = {}
    for tid, nm, ad in zip(target_ids, target_names_raw, target_addrs_raw):
        target_info[tid] = (nm, ad)
        
    # Preprocessing targets for indexing
    target_names_norm = [normalize_name(n) for n in target_names_raw]
    target_addrs_norm = [normalize_name(a) for a in target_addrs_raw]
    target_nums = []
    for a in target_addrs_raw:
        nums = [x.lstrip("0") for x in a.split() if x.isdigit()] if a else []
        target_nums.append(nums)
        
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(
        target_names_norm, target_addrs_norm, target_nums
    )
    
    idx_trans = defaultdict(list)
    if country == "India":
        for i, tn in enumerate(target_names_norm):
            if any(ord(c) > 127 for c in target_names_raw[i]):
                trans = normalize_name(transliterate(target_names_raw[i]))
                if LOANWORD_MAP:
                    trans = apply_loanword_map(trans, LOANWORD_MAP)
                ws = name_tokens_no_stops(trans)
                if ws:
                    idx_trans[ws[0]].append(i)
                    if len(ws) >= 2:
                        idx_trans[f"{ws[0]}_{ws[1]}"].append(i)
                        
    records = []
    pos_count = 0
    neg_count = 0
    
    for r1 in s1_df.iter_rows(named=True):
        sid = r1["entity_id"]
        s1_name = r1["business_name"] or ""
        s1_addr = r1["business_address"] or ""
        true_tgts = gt_dict.get(sid, set())
        
        s1_query = f"{s1_name} | {s1_addr} | {country}".strip()
        
        # 1. Add True Positives
        for tid in true_tgts:
            if tid in target_info:
                tn, ta = target_info[tid]
                t_passage = f"{tn} | {ta} | {country}".strip()
                records.append({
                    "s1_id": sid,
                    "target_id": tid,
                    "query": s1_query,
                    "passage": t_passage,
                    "label": 1.0,
                    "country": country,
                })
                pos_count += 1
                
        # 2. Retrieve Candidate Distractors (Hard Negatives)
        s1_norm = normalize_name(s1_name)
        s1_a_norm = normalize_name(s1_addr)
        s1_nums = [x.lstrip("0") for x in s1_addr.split() if x.isdigit()]
        
        cands = query_index(s1_norm, s1_a_norm, s1_nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)
        if country == "India":
            ws = name_tokens_no_stops(s1_norm)
            if ws:
                cands.update(idx_trans.get(ws[0], []))
                if len(ws) >= 2:
                    cands.update(idx_trans.get(f"{ws[0]}_{ws[1]}", []))
                    
        # Score candidates to pick the hardest negatives
        cands_scored = []
        for ci in cands:
            tid = target_ids[ci]
            if tid in true_tgts:
                continue # not a negative
            tn = target_names_norm[ci]
            ta = target_addrs_norm[ci]
            sim_n = fuzz.token_sort_ratio(s1_norm, tn) if s1_norm and tn else 0
            sim_a = fuzz.token_set_ratio(s1_a_norm, ta) if s1_a_norm and ta else 0
            score = max(sim_n, sim_a)
            cands_scored.append((tid, score))
            
        cands_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Pick top 6 hardest negatives per entity
        max_negs = 6 if true_tgts else 3 # fewer negatives for singletons
        for tid, _ in cands_scored[:max_negs]:
            tn, ta = target_info[tid]
            t_passage = f"{tn} | {ta} | {country}".strip()
            records.append({
                "s1_id": sid,
                "target_id": tid,
                "query": s1_query,
                "passage": t_passage,
                "label": 0.0,
                "country": country,
            })
            neg_count += 1
            
    print(f"  {country} pairs built in {time.time()-t0:.1f}s: {pos_count:,} Positives, {neg_count:,} Hard Negatives.")
    return records

def main():
    print("=" * 70)
    print("STEP C1: BUILDING CROSS-ENCODER TRAINING PAIRS")
    print("=" * 70)
    
    us_records = build_pairs_for_country("US")
    in_records = build_pairs_for_country("India")
    
    all_records = us_records + in_records
    df = pl.DataFrame(all_records)
    print(f"\nTotal pairs: {len(df):,} ({df.filter(pl.col('label') == 1.0).shape[0]:,} Pos, {df.filter(pl.col('label') == 0.0).shape[0]:,} Neg)")
    
    # 5-fold GroupKFold by s1_id
    gkf = GroupKFold(n_splits=5)
    s1_ids = df["s1_id"].to_numpy()
    folds = np.zeros(len(df), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=s1_ids)):
        folds[val_idx] = fold
        
    df = df.with_columns(pl.Series("fold", folds))
    
    # Save locally
    os.makedirs("data", exist_ok=True)
    df.write_parquet("data/cross_encoder_pairs.parquet")
    print(f"Saved to data/cross_encoder_pairs.parquet ({os.path.getsize('data/cross_encoder_pairs.parquet') / 1024 / 1024:.2f} MB)")
    
    # Copy to kaggle_dataset/
    os.makedirs("kaggle_dataset", exist_ok=True)
    df.write_parquet("kaggle_dataset/cross_encoder_pairs.parquet")
    print(f"Saved to kaggle_dataset/cross_encoder_pairs.parquet")

if __name__ == "__main__":
    main()
