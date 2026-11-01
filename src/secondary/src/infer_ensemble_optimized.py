"""
Optimized Production Ensemble Inference Pipeline (LightGBM v2 + CatBoost v2).
Blends LightGBM (0.70) + CatBoost (0.30) predictions on 24 enhanced features.
Optimized with:
  - Precomputed target features (transliteration, street core, house numbers, compact forms)
  - Memory-safe batching
  - Fast C-level RapidFuzz operations
  - Global mutual exclusivity post-processing
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
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    get_char_ngrams, jaccard, name_tokens_no_stops,
    query_index, build_inverted_index, get_name_freq,
    extract_street_core, extract_house_number,
)

LGBM_MODEL_PATH = "models/lgbm_v2.txt"
CB_MODEL_PATH = "models/catboost_v2.cbm"
TEST_DIR = "student_resource/dataset/test"
MAX_CANDIDATES = 20

COUNTRY_THRESHOLDS = {
    "France": 0.85,
    "US": 0.90,
    "India": 0.90,
}

LGBM_WEIGHT = 0.70
CB_WEIGHT = 0.30

def run_country_inference(country: str, lgbm_booster, cb_booster, out_matching_map, out_candidate_map, data_dir: str, override_tau: float = None, out_all_candidate_scores: dict = None):
    print(f"\n{'='*60}", flush=True)
    print(f"OPTIMIZED ENSEMBLE INFERENCE: {country}", flush=True)
    print(f"{'='*60}", flush=True)
    tau = override_tau if override_tau is not None else COUNTRY_THRESHOLDS.get(country, 0.90)
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

    tgt_freqs = [get_name_freq(n, freq_map) for n in tgt_names]

    # Build inverted index
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw)

    # Precompute non-ASCII targets and transliterations for India
    tgt_is_non_ascii = [False] * len(tgt_ids)
    tgt_trans_norm = [""] * len(tgt_ids)
    tgt_trans_mapped = [""] * len(tgt_ids)
    idx_trans = defaultdict(list)

    if country == "India":
        t_trans = time.time()
        for i, rn in enumerate(tgt_raw_names):
            if rn and any(ord(c) > 127 for c in rn):
                tgt_is_non_ascii[i] = True
                trans = normalize_name(transliterate(rn))
                tgt_trans_norm[i] = trans
                if LOANWORD_MAP:
                    mapped = apply_loanword_map(trans, LOANWORD_MAP)
                    tgt_trans_mapped[i] = mapped
                else:
                    tgt_trans_mapped[i] = trans
                    
                trans_words = name_tokens_no_stops(trans)
                if trans_words:
                    idx_trans[trans_words[0]].append(i)
                    if len(trans_words) >= 2:
                        ws = sorted(trans_words[:2])
                        idx_trans[f"{ws[0]}_{ws[1]}"].append(i)
        print(f"  Transliterated index & targets precomputed ({time.time()-t_trans:.1f}s)", flush=True)

    print(f"  Index built ({time.time()-t2:.1f}s)", flush=True)

    # Precompute target properties lazily / on-demand with caching
    tg_cache = {}  # c_idx -> (tg_sc, tg_hn, tg_dist, tg_compact, tg_num_set, ng2, ng3, ng4, ang3)

    def get_tg_props(c_idx):
        if c_idx in tg_cache:
            return tg_cache[c_idx]
        tn = tgt_names[c_idx]
        ta = tgt_addrs[c_idx]
        sc = extract_street_core(ta)
        hn = extract_house_number(ta)
        dist = {w for w in name_tokens_no_stops(tn) if len(w) >= 5}
        compact = "".join(c for c in tn.lower() if c.isalnum())
        num_set = set(s.lstrip("0") for s in (tgt_nums_raw[c_idx] or []) if s.lstrip("0"))
        ng2 = get_char_ngrams(tn, 2)
        ng3 = get_char_ngrams(tn, 3)
        ng4 = get_char_ngrams(tn, 4)
        ang3 = get_char_ngrams(ta, 3)
        props = (sc, hn, dist, compact, num_set, ng2, ng3, ng4, ang3)
        tg_cache[c_idx] = props
        return props

    # Batch prediction
    t3 = time.time()
    BATCH_SIZE = 5000
    
    all_scored_pairs = [] # (prob, sid, tg_id)
    s1_top_cands = {}     # sid -> list of top 20 candidate tgt_ids

    for batch_start in range(0, len(s1_ids), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(s1_ids))
        batch_feats = []
        batch_map = []
        batch_entity_cands = defaultdict(list)

        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            s1n = s1_names[idx]
            s1a = s1_addrs[idx]
            s1r = s1_raw_names[idx]
            s1nums = [x.lstrip("0") for x in (s1_nums_raw[idx] or []) if x.lstrip("0")]
            s1_freq = get_name_freq(s1n, freq_map)

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
            s1_len_n = len(s1n)
            s1_len_a = len(s1a)
            freq_s1_log = np.log1p(s1_freq)

            s1_trans_norm = ""
            s1_trans_mapped = ""
            if s1_is_non_ascii:
                s1_trans_norm = normalize_name(transliterate(s1r))
                s1_trans_mapped = apply_loanword_map(s1_trans_norm, LOANWORD_MAP) if LOANWORD_MAP else s1_trans_norm

            cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)

            # India cross-script
            if country == "India":
                if not s1_is_non_ascii:
                    s1_words = name_tokens_no_stops(s1n)
                    if s1_words:
                        cands.update(idx_trans.get(s1_words[0], [])[:20])
                        if len(s1_words) >= 2:
                            ws = sorted(s1_words[:2])
                            cands.update(idx_trans.get(f"{ws[0]}_{ws[1]}", [])[:20])

            # Limit to top 25 candidates per entity
            if len(cands) > 25:
                # Fast set-based pruning before expensive string metrics
                cands_scored = []
                for c_idx in cands:
                    tn = tgt_names[c_idx]
                    ta = tgt_addrs[c_idx]
                    ns = fuzz.token_sort_ratio(s1n, tn) if s1n and tn else 0
                    as_ = fuzz.token_set_ratio(s1a, ta) if s1a and ta else 0
                    cands_scored.append((c_idx, max(ns, as_)))
                cands_scored.sort(key=lambda x: x[1], reverse=True)
                cands = [x[0] for x in cands_scored[:25]]

            for c_idx in cands:
                tg_id = tgt_ids[c_idx]
                tgn = tgt_names[c_idx]
                tga = tgt_addrs[c_idx]
                tg_freq = tgt_freqs[c_idx]
                tg_non_ascii = tgt_is_non_ascii[c_idx]

                n_sort = fuzz.token_sort_ratio(s1n, tgn) / 100.0 if s1n and tgn else 0.0
                n_set = fuzz.token_set_ratio(s1n, tgn) / 100.0 if s1n and tgn else 0.0
                a_set = fuzz.token_set_ratio(s1a, tga) / 100.0 if s1a and tga else 0.0
                a_sort = fuzz.token_sort_ratio(s1a, tga) / 100.0 if s1a and tga else 0.0

                freq_tg_log = np.log1p(tg_freq)
                freq_ratio = min(s1_freq, tg_freq) / max(s1_freq, tg_freq, 1)
                n_len_diff = abs(s1_len_n - len(tgn))
                a_len_diff = abs(s1_len_a - len(tga))

                # Early exit for very low similarities
                if max(n_sort, a_set) < 0.25:
                    feats = [
                        n_sort, n_set, a_set, a_sort,
                        0.0, 0.0, 0.0, 0.0,
                        0, 0.0,
                        0, 0.0, 0.0,
                        0.0, 0, 0,
                        0.0, 0,
                        freq_s1_log, freq_tg_log, freq_ratio,
                        n_len_diff, a_len_diff,
                        0
                    ]
                else:
                    tg_sc, tg_hn, tg_dist, tg_compact, tg_num_set, ng2, ng3, ng4, ang3 = get_tg_props(c_idx)

                    n_jacc2 = jaccard(s1_ng2, ng2)
                    n_jacc3 = jaccard(s1_ng3, ng3)
                    n_jacc4 = jaccard(s1_ng4, ng4)
                    a_jacc3 = jaccard(s1_ang3, ang3)

                    num_overlap = len(s1_num_set & tg_num_set)
                    num_jacc = jaccard(s1_num_set, tg_num_set)

                    is_cross_script = int(
                        (not s1_is_non_ascii and tg_non_ascii) or
                        (s1_is_non_ascii and not tg_non_ascii)
                    )

                    trans_sim = 0.0
                    loanword_sim = 0.0
                    if is_cross_script:
                        if not s1_is_non_ascii and tg_non_ascii:
                            trans_sim = fuzz.token_sort_ratio(s1n, tgt_trans_norm[c_idx]) / 100.0
                            loanword_sim = fuzz.token_sort_ratio(s1n, tgt_trans_mapped[c_idx]) / 100.0
                        else:
                            trans_sim = fuzz.token_sort_ratio(s1_trans_norm, tgn) / 100.0
                            loanword_sim = fuzz.token_sort_ratio(s1_trans_mapped, tgn) / 100.0

                    street_core_sim = fuzz.token_sort_ratio(s1_sc, tg_sc) / 100.0 if s1_sc and tg_sc else 0.0
                    house_num_match = int(bool(s1_hn and tg_hn and s1_hn == tg_hn))
                    hn_street_match = int(bool(house_num_match and s1_sc and tg_sc and fuzz.partial_ratio(s1_sc, tg_sc) >= 80))

                    dist_overlap = len(s1_dist & tg_dist) / max(len(s1_dist | tg_dist), 1)
                    generic_mismatch = int(len(s1_dist) >= 1 and len(tg_dist) >= 1 and len(s1_dist & tg_dist) == 0)

                    domain_match = int(
                        bool(s1_compact and tg_compact and
                             (s1_compact in tg_compact or tg_compact in s1_compact) and
                             len(min(s1_compact, tg_compact, key=len)) >= 5)
                    )

                    feats = [
                        n_sort, n_set, a_set, a_sort,
                        n_jacc2, n_jacc3, n_jacc4, a_jacc3,
                        num_overlap, num_jacc,
                        is_cross_script, trans_sim, loanword_sim,
                        street_core_sim, house_num_match, hn_street_match,
                        dist_overlap, generic_mismatch,
                        freq_s1_log, freq_tg_log, freq_ratio,
                        n_len_diff, a_len_diff,
                        domain_match,
                    ]

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
                if out_all_candidate_scores is not None:
                    if sid not in out_all_candidate_scores:
                        out_all_candidate_scores[sid] = []
                    out_all_candidate_scores[sid].append((tg_id, float(prob)))
                if prob >= tau:
                    all_scored_pairs.append((float(prob), sid, tg_id))

        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            cands_sorted = sorted(batch_entity_cands.get(sid, []), key=lambda x: x[1], reverse=True)
            s1_top_cands[sid] = [c[0] for c in cands_sorted[:MAX_CANDIDATES]]

        # Free cache if it gets too large
        if len(tg_cache) > 200000:
            tg_cache.clear()

        if batch_end % 50000 == 0 or batch_end == len(s1_ids):
            print(f"  Processed {batch_end:,}/{len(s1_ids):,} entities ({time.time()-t3:.0f}s)", flush=True)

    # Global Mutual Exclusivity
    print("  Applying global mutual exclusivity...", flush=True)
    all_scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_targets = set()
    s1_matches = defaultdict(list)
    
    for prob, sid, tg_id in all_scored_pairs:
        if tg_id not in used_targets:
            s1_matches[sid].append(tg_id)
            used_targets.add(tg_id)

    for sid in s1_ids:
        matches = s1_matches.get(sid, [])
        top_cands = s1_top_cands.get(sid, [])
        full_cands = list(dict.fromkeys(matches + top_cands))
        out_matching_map[sid] = matches
        out_candidate_map[sid] = full_cands

    print(f"  {country} completed in {time.time()-t0:.1f}s (matched: {len(used_targets):,} target links)", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Optimized Ensemble Inference")
    parser.add_argument("--out-dir", default="submissions/v2_ensemble", help="Output directory")
    parser.add_argument("--data-dir", default=TEST_DIR, help="Data directory (test or holdout)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    matching_out = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.out_dir, "candidate_pairs.tsv")

    print(f"=== Running Optimized Ensemble Inference ===")
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
