"""
Production Ensemble Inference Pipeline (LightGBM v2 + CatBoost v2).
Blends LightGBM (0.70) + CatBoost (0.30) predictions on 24 enhanced features.
Features:
  - Lexical similarities & n-gram Jaccard (2, 3, 4)
  - Number & distinctive token overlaps
  - Anti-false-merge (street core similarity, house number matching)
  - Cross-script & Indic transliteration with loanword correction
  - Chain frequency detection
  - Mutual exclusivity post-processing
Produces:
  matching_results.tsv
  candidate_pairs.tsv
"""
import os
import sys
import gc
import re
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from train_lgbm_v2 import (
    FEATURE_COLS, LEGAL_STOPS, GENERIC_ADDR,
    normalize_name, extract_features, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    get_char_ngrams, jaccard, name_tokens_no_stops,
    query_index, build_inverted_index, get_name_freq,
    extract_street_core, extract_house_number,
)

LGBM_MODEL_PATH = "models/lgbm_v2.txt"
CB_MODEL_PATH = "models/catboost_v2.cbm"
TEST_DIR = "student_resource/dataset/test"
MAX_CANDIDATES = 20

# Per-country thresholds tuned on DEV OOF
COUNTRY_THRESHOLDS = {
    "France": 0.85,
    "US": 0.90,
    "India": 0.90,
}

LGBM_WEIGHT = 0.70
CB_WEIGHT = 0.30

def run_country_inference(country: str, lgbm_booster, cb_booster, out_matching_map, out_candidate_map, data_dir: str):
    print(f"\n{'='*60}", flush=True)
    print(f"ENSEMBLE INFERENCE: {country}", flush=True)
    print(f"{'='*60}", flush=True)
    tau = COUNTRY_THRESHOLDS.get(country, 0.90)
    t0 = time.time()

    # Load S1
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s1_df = pl.scan_csv(s1_path, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    
    if len(s1_df) == 0:
        print(f"  No S1 entities for {country}.", flush=True)
        return
        
    s1_df = s1_df.with_columns([
        pl.col("business_name").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_clean"),
        pl.col("business_address").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("addr_clean"),
        pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
    ])
    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["name_clean"].to_list()
    s1_addrs = s1_df["addr_clean"].to_list()
    s1_nums_raw = s1_df["nums"].to_list()
    s1_raw_names = s1_df["business_name"].fill_null("").to_list()
    del s1_df
    gc.collect()
    print(f"  Loaded {len(s1_ids):,} S1 entities ({time.time()-t0:.1f}s)", flush=True)

    # Load targets
    t1 = time.time()
    s2 = pl.scan_csv(os.path.join(data_dir, "test_source2.tsv"), separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    s3 = pl.scan_csv(os.path.join(data_dir, "test_source3.tsv"), separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    
    tgt_df = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

    tgt_df = tgt_df.with_columns([
        pl.col("business_name").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_clean"),
        pl.col("business_address").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("addr_clean"),
        pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
    ])
    tgt_ids = tgt_df["entity_id"].to_list()
    tgt_names = tgt_df["name_clean"].to_list()
    tgt_addrs = tgt_df["addr_clean"].to_list()
    tgt_nums_raw = tgt_df["nums"].to_list()
    tgt_raw_names = tgt_df["business_name"].fill_null("").to_list()
    del tgt_df
    gc.collect()
    print(f"  Loaded {len(tgt_ids):,} targets ({time.time()-t1:.1f}s)", flush=True)

    # Build name frequency map
    t2 = time.time()
    freq_map = defaultdict(int)
    for n in tgt_names:
        key = " ".join(sorted(name_tokens_no_stops(n)))
        if key:
            freq_map[key] += 1

    # Build inverted index
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw)

    # India transliterated index
    idx_trans = defaultdict(list)
    if country == "India":
        for i, rn in enumerate(tgt_raw_names):
            if rn and any(ord(c) > 127 for c in rn):
                trans = normalize_name(transliterate(rn))
                if LOANWORD_MAP:
                    trans = apply_loanword_map(trans, LOANWORD_MAP)
                trans_words = name_tokens_no_stops(trans)
                if trans_words:
                    idx_trans[trans_words[0]].append(i)
                    if len(trans_words) >= 2:
                        ws = sorted(trans_words[:2])
                        idx_trans[f"{ws[0]}_{ws[1]}"].append(i)

    print(f"  Index built ({time.time()-t2:.1f}s)", flush=True)

    # Precompute target frequencies
    t2_freq = time.time()
    tgt_freqs = [get_name_freq(n, freq_map) for n in tgt_names]
    print(f"  Target frequencies computed ({time.time()-t2_freq:.1f}s)", flush=True)

    # Batch prediction
    t3 = time.time()
    BATCH_SIZE = 4000
    
    # Store all candidate pairs above tau for global mutual exclusivity
    all_scored_pairs = [] # (prob, s1_id, tgt_id)
    s1_top_cands = {}     # s1_id -> list of top 20 candidate tgt_ids

    for batch_start in range(0, len(s1_ids), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(s1_ids))
        batch_feats = []
        batch_map = []  # (s1_id, tg_id)
        batch_entity_cands = defaultdict(list)

        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            s1n = s1_names[idx]
            s1a = s1_addrs[idx]
            s1r = s1_raw_names[idx]
            s1nums = [x.lstrip("0") for x in (s1_nums_raw[idx] or []) if x.lstrip("0")]
            s1_freq = get_name_freq(s1n, freq_map)

            # Precompute S1 properties once per entity
            s1_ng2 = get_char_ngrams(s1n, 2)
            s1_ng3 = get_char_ngrams(s1n, 3)
            s1_ng4 = get_char_ngrams(s1n, 4)
            s1_ang3 = get_char_ngrams(s1a, 3)
            s1_num_set = set(s1nums)
            s1_sc = extract_street_core(s1a)
            s1_hn = extract_house_number(s1a)
            s1_dist = {w for w in name_tokens_no_stops(s1n) if len(w) >= 5}
            s1_compact = "".join(c for c in s1n.lower() if c.isalnum())
            s1_is_non_ascii = any(ord(c) > 127 for c in s1r)

            cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)

            # India cross-script: query transliterated index
            if country == "India":
                is_ascii_s1 = not any(ord(c) > 127 for c in s1r)
                if is_ascii_s1:
                    s1_words = name_tokens_no_stops(s1n)
                    if s1_words:
                        cands.update(idx_trans.get(s1_words[0], [])[:30])
                        if len(s1_words) >= 2:
                            ws = sorted(s1_words[:2])
                            cands.update(idx_trans.get(f"{ws[0]}_{ws[1]}", [])[:30])

            # Limit candidates per entity for fast inference
            if len(cands) > 30:
                quick = []
                for c_idx in cands:
                    ns = fuzz.token_sort_ratio(s1n, tgt_names[c_idx]) if s1n and tgt_names[c_idx] else 0
                    as_ = fuzz.token_set_ratio(s1a, tgt_addrs[c_idx]) if s1a and tgt_addrs[c_idx] else 0
                    quick.append((c_idx, max(ns, as_)))
                quick.sort(key=lambda x: x[1], reverse=True)
                cands = {x[0] for x in quick[:30]}

            for c_idx in cands:
                tg_id = tgt_ids[c_idx]
                feats = extract_features(
                    s1n, s1a, s1r,
                    s1_ng2, s1_ng3, s1_ng4, s1_ang3, s1_num_set,
                    s1_sc, s1_hn, s1_dist, s1_compact, s1_is_non_ascii,
                    tgt_names[c_idx], tgt_addrs[c_idx], tgt_raw_names[c_idx], tgt_nums_raw[c_idx],
                    s1_freq, tgt_freqs[c_idx],
                    LOANWORD_MAP,
                )
                batch_feats.append(feats)
                batch_map.append((sid, tg_id))

        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs_lgb = lgbm_booster.predict(X_batch)
            if cb_booster is not None:
                probs_cb = cb_booster.predict_proba(X_batch)[:, 1]
                probs = LGBM_WEIGHT * probs_lgb + CB_WEIGHT * probs_cb
            else:
                probs = probs_lgb
                
            for (sid, tg_id), prob in zip(batch_map, probs):
                batch_entity_cands[sid].append((tg_id, float(prob)))
                if prob >= tau:
                    all_scored_pairs.append((float(prob), sid, tg_id))

        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            cands_sorted = sorted(batch_entity_cands.get(sid, []), key=lambda x: x[1], reverse=True)
            s1_top_cands[sid] = [c[0] for c in cands_sorted[:MAX_CANDIDATES]]

        if batch_end % 50000 == 0 or batch_end == len(s1_ids):
            print(f"  Processed {batch_end:,}/{len(s1_ids):,} entities ({time.time()-t3:.0f}s)", flush=True)

    # Global Mutual Exclusivity: sort all pairs above tau by probability descending
    print("  Applying global mutual exclusivity...", flush=True)
    all_scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_targets = set()
    s1_matches = defaultdict(list)
    
    for prob, sid, tg_id in all_scored_pairs:
        if tg_id not in used_targets:
            s1_matches[sid].append(tg_id)
            used_targets.add(tg_id)

    # Store results in maps
    for sid in s1_ids:
        matches = s1_matches.get(sid, [])
        top_cands = s1_top_cands.get(sid, [])
        # Candidate pairs MUST contain all matched entities
        full_cands = list(dict.fromkeys(matches + top_cands))
        
        out_matching_map[sid] = matches
        out_candidate_map[sid] = full_cands

    print(f"  {country} completed in {time.time()-t0:.1f}s (matched: {len(used_targets):,} target links)", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Ensemble (LightGBM + CatBoost) Production Inference")
    parser.add_argument("--out-dir", default="output/holdout_ensemble", help="Output directory")
    parser.add_argument("--data-dir", default=TEST_DIR, help="Data directory (test or holdout)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    matching_out = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.out_dir, "candidate_pairs.tsv")

    print(f"=== Running Ensemble Inference ===")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Out dir:  {args.out_dir}")
    t_start = time.time()

    print(f"Loading LightGBM model from {LGBM_MODEL_PATH}...")
    lgbm_booster = lgb.Booster(model_file=LGBM_MODEL_PATH)
    
    cb_booster = None
    if os.path.exists(CB_MODEL_PATH):
        print(f"Loading CatBoost model from {CB_MODEL_PATH}...")
        cb_booster = CatBoostClassifier()
        cb_booster.load_model(CB_MODEL_PATH)
    else:
        print("[WARN] CatBoost model not found, falling back to pure LightGBM.")

    out_matching_map = {}
    out_candidate_map = {}

    s1_all = pl.scan_csv(os.path.join(args.data_dir, "test_source1.tsv"), separator="\t").select(["country"]).collect()
    available_countries = s1_all["country"].unique().to_list()
    
    # Run inference for all countries present
    for country in ["France", "US", "India"]:
        if country in available_countries:
            run_country_inference(country, lgbm_booster, cb_booster, out_matching_map, out_candidate_map, args.data_dir)

    print(f"\nWriting results to {matching_out} and {candidate_out}...")
    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for sid, matches in out_matching_map.items():
            f_match.write(f"{sid}\t{','.join(matches)}\n")
            cands = out_candidate_map.get(sid, matches)
            f_cand.write(f"{sid}\t{','.join(cands)}\n")

    print(f"Done in {time.time()-t_start:.1f}s.")
    print(f"  {matching_out}: {os.path.getsize(matching_out):,} bytes")
    print(f"  {candidate_out}: {os.path.getsize(candidate_out):,} bytes")

    # Run check_submission
    print("\nValidating submission structure with check_submission.py...")
    import check_submission
    ok = check_submission.check_submission_files(
        matching_path=matching_out,
        candidate_path=candidate_out,
        test_dir=args.data_dir,
        check_ids=False,
    )
    if not ok:
        print("ERROR: check_submission FAILED!")
        sys.exit(1)
    print("SUCCESS: Submission passed all validation checks!")

if __name__ == "__main__":
    main()
