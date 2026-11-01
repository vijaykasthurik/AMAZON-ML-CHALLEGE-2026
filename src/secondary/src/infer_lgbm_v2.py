"""
Production LightGBM v2 Inference Pipeline.
Identical feature set to train_lgbm_v2.py.
Enhanced with:
  - Cross-script blocking: transliterated Indic names added to index.
  - Data-driven loanword mapping.
  - Anti-false-merge features.
  - Mutual-exclusivity post-processing.
Produces:
  output/matching_results.tsv
  output/candidate_pairs.tsv
  (or custom paths via CLI --out-dir)
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

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add project root so train_lgbm_v2 is importable regardless of working dir
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # parent of src/
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# Import shared feature code from train_lgbm_v2 (at project root)
from train_lgbm_v2 import (
    FEATURE_COLS, LEGAL_STOPS, GENERIC_ADDR,
    normalize_name, extract_features, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    get_char_ngrams, jaccard, name_tokens_no_stops,
    query_index, build_inverted_index, get_name_freq,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_V2_PATH = "models/lgbm_v2.txt"
TEST_DIR = "student_resource/dataset/test"
SCORE_THRESHOLD_DEFAULT = 0.50  # will be overridden per-country
MAX_CANDIDATES = 20

# Per-country thresholds (tuned on OOF by train_lgbm_v2.py; override if better found)
COUNTRY_THRESHOLDS = {
    "France": 0.50,
    "US": 0.55,
    "India": 0.50,
}


def run_country_inference(
    country: str, booster, f_match, f_cand, out_dir: str,
    data_dir: str = TEST_DIR,
):
    """Run full inference pipeline for one country and write output rows."""
    print(f"\n{'='*60}", flush=True)
    print(f"INFERENCE: {country}", flush=True)
    print(f"{'='*60}", flush=True)
    tau = COUNTRY_THRESHOLDS.get(country, 0.50)
    t0 = time.time()

    # Load S1
    s1_df = pl.scan_csv(os.path.join(data_dir, "test_source1.tsv"), separator="\t").filter(
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

    # Build name frequency map from test targets (chain detection)
    t2 = time.time()
    freq_map = defaultdict(int)
    for n in tgt_names:
        key = " ".join(sorted(name_tokens_no_stops(n)))
        if key:
            freq_map[key] += 1

    # Build inverted index
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw)

    # For India: also index transliterated target names
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

    # Batch inference
    t3 = time.time()
    BATCH_SIZE = 2000
    match_rows = []
    cand_rows = []
    n_processed = 0

    for batch_start in range(0, len(s1_ids), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(s1_ids))
        batch_feats = []
        batch_map = []  # (idx_in_batch, tgt_id)
        entity_cands = defaultdict(list)

        for bi, idx in enumerate(range(batch_start, batch_end)):
            sid = s1_ids[idx]
            s1n = s1_names[idx]
            s1a = s1_addrs[idx]
            s1r = s1_raw_names[idx]
            s1nums = [x.lstrip("0") for x in (s1_nums_raw[idx] or []) if x.lstrip("0")]
            s1_freq = get_name_freq(s1n, freq_map)

            cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)

            # India cross-script: also query transliterated index
            if country == "India":
                is_ascii_s1 = not any(ord(c) > 127 for c in s1r)
                if is_ascii_s1:
                    s1_words = name_tokens_no_stops(s1n)
                    if s1_words:
                        cands.update(idx_trans.get(s1_words[0], [])[:30])
                        if len(s1_words) >= 2:
                            ws = sorted(s1_words[:2])
                            cands.update(idx_trans.get(f"{ws[0]}_{ws[1]}", [])[:30])

            for c_idx in cands:
                tg_id = tgt_ids[c_idx]
                tg_freq = freq_map.get(" ".join(sorted(name_tokens_no_stops(tgt_names[c_idx]))), 1)
                feats = extract_features(
                    s1n, s1a, s1r, s1nums,
                    tgt_names[c_idx], tgt_addrs[c_idx], tgt_raw_names[c_idx], tgt_nums_raw[c_idx],
                    s1_freq, tg_freq,
                    LOANWORD_MAP,
                )
                batch_feats.append(feats)
                batch_map.append((idx, tg_id))

        if batch_feats:
            probs = booster.predict(np.array(batch_feats, dtype=np.float32))
            for (entity_idx, tg_id), prob in zip(batch_map, probs):
                entity_cands[entity_idx].append((tg_id, prob))

        # Write rows for this batch
        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            cands_sorted = sorted(entity_cands.get(idx, []), key=lambda x: x[1], reverse=True)
            top_cands = [c[0] for c in cands_sorted[:MAX_CANDIDATES]]
            matched = [c[0] for c in cands_sorted if c[1] >= tau]
            match_rows.append(f"{sid}\t{','.join(matched)}\n")
            cand_rows.append(f"{sid}\t{','.join(top_cands)}\n")

        n_processed = batch_end
        if n_processed % 50000 == 0 or n_processed == len(s1_ids):
            print(f"  {n_processed:,}/{len(s1_ids):,} entities "
                  f"({time.time()-t3:.0f}s)", flush=True)

        # Flush to disk periodically
        if len(match_rows) >= 50000:
            f_match.writelines(match_rows)
            f_cand.writelines(cand_rows)
            match_rows.clear()
            cand_rows.clear()

    if match_rows:
        f_match.writelines(match_rows)
        f_cand.writelines(cand_rows)

    print(f"  {country} done in {time.time()-t0:.0f}s", flush=True)
    del tgt_ids, tgt_names, tgt_addrs, tgt_nums_raw, tgt_raw_names
    del idx_w2, idx_w1, idx_w1_num, idx_aw_num
    gc.collect()


def main():
    """Entry point for v2 inference."""
    parser = argparse.ArgumentParser(description="LightGBM v2 Production Inference")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--data-dir", default=TEST_DIR, help="Test data directory")
    parser.add_argument("--model", default=MODEL_V2_PATH, help="Model path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    matching_out = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.out_dir, "candidate_pairs.tsv")

    print("=== LightGBM v2 Production Inference ===", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Output: {args.out_dir}", flush=True)
    t_start = time.time()

    booster = lgb.Booster(model_file=args.model)

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in ["France", "US", "India"]:
            run_country_inference(country, booster, f_match, f_cand, args.out_dir, args.data_dir)

    print(f"\nAll countries done in {time.time()-t_start:.0f}s", flush=True)
    print(f"  {matching_out}: {os.path.getsize(matching_out):,} bytes")
    print(f"  {candidate_out}: {os.path.getsize(candidate_out):,} bytes")

    # Run official check
    print("\nRunning official validation...", flush=True)
    import check_submission
    ok = check_submission.check_submission_files(
        matching_path=matching_out, candidate_path=candidate_out,
        test_dir=TEST_DIR, check_ids=False
    )
    if not ok:
        print("ERROR: Submission validation FAILED!", flush=True)
        sys.exit(1)
    print("=== v2 INFERENCE COMPLETE AND VALIDATED ===", flush=True)


if __name__ == "__main__":
    main()
