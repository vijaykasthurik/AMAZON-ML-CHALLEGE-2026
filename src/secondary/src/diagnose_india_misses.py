"""
Diagnose India False Negatives:
Analyzes true pairs present in candidates that score below tau (tau = 0.44).
Categorizes the errors:
  1. Cross-script (one side non-ASCII / Indic script, other side Latin/English)
  2. Address-only match (name has low overlap, address has high overlap)
  3. Abbreviation / Initials mismatch
  4. Number/pin code mismatch
  5. Other phonetic / loanword variations
"""
import os
import sys
import json
import numpy as np
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    extract_features,
    get_char_ngrams, extract_street_core, extract_house_number,
    name_tokens_no_stops,
)
import lightgbm as lgb
from catboost import CatBoostClassifier

def main():
    print("=" * 70)
    print("DIAGNOSING INDIA FALSE NEGATIVES (True Pairs Scoring Below Tau)")
    print("=" * 70)
    
    dev_dir = "data/calibrated_dev/India_density_5.75"
    s1_df = pl.read_csv(os.path.join(dev_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(dev_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(dev_dir, "test_source3.tsv"), separator="\t")
    gt_df = pl.read_csv(os.path.join(dev_dir, "ground_truth.tsv"), separator="\t")
    
    tgt_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    tgt_map = {r["entity_id"]: r for r in tgt_df.iter_rows(named=True)}
    
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        if m:
            gt_dict[r["source1_entity_id"]] = [x.strip() for x in m.split(",") if x.strip()]
            
    print(f"Loaded {len(s1_df):,} S1 entities, {len(gt_dict):,} with true matches.")
    
    # Load models
    lgbm_booster = lgb.Booster(model_file="models/lgbm_v2.txt")
    cb_booster = CatBoostClassifier()
    cb_booster.load_model("models/catboost_v2.cbm")
    
    tau = 0.44
    categories = defaultdict(list)
    
    all_true_pairs = []
    for r1 in s1_df.iter_rows(named=True):
        sid = r1["entity_id"]
        true_tgts = gt_dict.get(sid, [])
        for tid in true_tgts:
            if tid in tgt_map:
                r2 = tgt_map[tid]
                all_true_pairs.append((r1, r2))
                
    print(f"Total true pairs in calibrated DEV: {len(all_true_pairs):,}")
    
    # Extract features and score all true pairs
    feats_list = []
    for r1, r2 in all_true_pairs:
        s1n = normalize_name(r1["business_name"] or "")
        s1a = normalize_name(r1["business_address"] or "")
        s1r = r1["business_name"] or ""
        s1nums = [x.lstrip("0") for x in (r1["business_address"] or "").split() if x.isdigit()]
        
        t2n = normalize_name(r2["business_name"] or "")
        t2a = normalize_name(r2["business_address"] or "")
        t2r = r2["business_name"] or ""
        t2nums = [x.lstrip("0") for x in (r2["business_address"] or "").split() if x.isdigit()]
        
        # Features
        f = extract_features(
            s1n, s1a, s1r,
            None, None, None, None, set(s1nums),
            None, None, set(), "".join(c for c in s1n.lower() if c.isalnum()),
            any(ord(c) > 127 for c in s1r),
            t2n, t2a, t2r, t2nums,
            1, 1, LOANWORD_MAP
        )
        feats_list.append(f)
        
    X = np.array(feats_list, dtype=np.float32)
    probs_lgb = lgbm_booster.predict(X)
    probs_cb = cb_booster.predict_proba(X)[:, 1]
    probs = 0.7 * probs_lgb + 0.3 * probs_cb
    
    below_tau_count = 0
    for idx, ((r1, r2), prob) in enumerate(zip(all_true_pairs, probs)):
        if prob < tau:
            below_tau_count += 1
            s1_name = r1["business_name"] or ""
            s1_addr = r1["business_address"] or ""
            t_name = r2["business_name"] or ""
            t_addr = r2["business_address"] or ""
            
            # Categorize
            s1_non_ascii = any(ord(c) > 127 for c in s1_name)
            t_non_ascii = any(ord(c) > 127 for c in t_name)
            
            sim_name = fuzz.token_sort_ratio(normalize_name(s1_name), normalize_name(t_name))
            sim_addr = fuzz.token_set_ratio(normalize_name(s1_addr), normalize_name(t_addr))
            
            pair_info = {
                "s1_id": r1["entity_id"],
                "target_id": r2["entity_id"],
                "prob": float(prob),
                "s1_name": s1_name,
                "target_name": t_name,
                "s1_addr": s1_addr,
                "target_addr": t_addr,
                "sim_name": sim_name,
                "sim_addr": sim_addr,
            }
            
            if s1_non_ascii != t_non_ascii:
                categories["cross_script"].append(pair_info)
            elif sim_name < 50 and sim_addr >= 75:
                categories["address_only_match"].append(pair_info)
            elif sim_name < 50 and sim_addr < 50:
                categories["heavy_abbreviation_or_phonetic"].append(pair_info)
            elif 50 <= sim_name < 75:
                categories["moderate_name_variation"].append(pair_info)
            else:
                categories["other"].append(pair_info)
                
    total = len(all_true_pairs)
    print("\n" + "=" * 60)
    print(f"INDIA ERROR BREAKDOWN (True Pairs Below Tau = {tau}):")
    print(f"Total True Pairs: {total:,} | Below Tau: {below_tau_count:,} ({below_tau_count/total*100:.2f}%)")
    print("-" * 60)
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat:<35}: {len(items):>5} pairs ({len(items)/below_tau_count*100:.1f}%)")
    print("=" * 60)
    
    # Show top 5 examples for largest categories
    for cat in ["cross_script", "address_only_match", "heavy_abbreviation_or_phonetic", "moderate_name_variation"]:
        if categories[cat]:
            print(f"\n--- Top 3 Examples for: {cat.upper()} ---")
            for ex in categories[cat][:3]:
                print(f"  Prob={ex['prob']:.4f} | Name Sim={ex['sim_name']} | Addr Sim={ex['sim_addr']}")
                print(f"    S1: {ex['s1_name']} | {ex['s1_addr']}")
                print(f"    T2: {ex['target_name']} | {ex['target_addr']}")

if __name__ == "__main__":
    main()
