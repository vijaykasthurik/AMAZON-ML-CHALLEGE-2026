"""
Production LightGBM Inference Pipeline for Amazon ML Challenge 2026.
Produces official submission files:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
Uses trained LightGBM reranker (Macro F0.5 = 0.8634) with multi-key inverted index blocking.
Caches and streams predictions country-by-country (France, US, India).
"""

import os
import sys
import gc
import time
from collections import defaultdict
import numpy as np
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(__file__))
sys.path.append("src")

# Challenge Legal Stopwords & Generic Address terms (Multilingual: US, India, France)
LEGAL_STOPS = {
    # English / US / Global
    "inc", "corp", "corporation", "llc", "ltd", "limited", "co", "pvt", "private", 
    "enterprises", "enterprise", "group", "holdings", "services", "company", "center", 
    "centre", "associates", "international", "intl", "solutions", "technologies",
    # France
    "sarl", "sasu", "sas", "sa", "eurl", "snc", "gie", "sci", "ste", "societe"
}

GENERIC_ADDR = {
    # US / India
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", 
    "floor", "fl", "building", "bldg", "block", "blk", "plot", "near", "opp", 
    "opposite", "behind", "north", "south", "east", "west", "india", "us", "usa", "france",
    "maharashtra", "karnataka", "delhi", "gujarat", "tamil", "nadu", "telangana",
    "mumbai", "bangalore", "bengaluru", "chennai", "hyderabad", "ahmedabad", "kolkata",
    "ny", "il", "tx", "ca", "fl", "nc", "oh", "pa", "ct", "mo", "va", "ga", "co",
    # France
    "rue", "boulevard", "bd", "bld", "chemin", "ch", "allee", "place", "pl", "cours",
    "zone", "za", "zi", "zac", "batiment", "bat", "etage", "bis", "ter"
}

TEST_DIR = "student_resource/dataset/test"
MATCHING_OUT = "output/matching_results.tsv"
CANDIDATE_OUT = "output/candidate_pairs.tsv"
MODEL_PATH = "models/lgbm_reranker.txt"
SCORE_THRESHOLD = 0.55
MAX_CANDIDATES = 20


def get_char_ngrams(text: str, n=3):
    if not text: return set()
    cleaned = "".join(c for c in text.lower() if c.isalnum())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i+n] for i in range(len(cleaned)-n+1)}


def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def run_country_inference(country: str, booster: lgb.Booster, f_match, f_cand):
    """
    Process one country's records against its candidate pool using LightGBM.
    """
    print(f"\n==========================================", flush=True)
    print(f"PROCESSING COUNTRY: {country}", flush=True)
    print(f"==========================================", flush=True)
    
    t0 = time.time()
    s1_all = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t")
    s1_c = s1_all.filter(pl.col("country") == country)
    del s1_all
    gc.collect()
    print(f"Loaded {len(s1_c)} S1 entities for {country} in {time.time()-t0:.2f}s.", flush=True)
    
    if len(s1_c) == 0:
        return
        
    t0 = time.time()
    s2_c = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country)
    s3_c = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country)
    targets_c = pl.concat([s2_c, s3_c])
    del s2_c, s3_c
    gc.collect()
    print(f"Loaded {len(targets_c)} target entities (S2+S3) in {time.time()-t0:.2f}s.", flush=True)
    
    # Vectorized target cleaning
    t1 = time.time()
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
    print(f"Vectorized target cleaning in {time.time()-t1:.2f}s.", flush=True)
    
    # Build multi-key inverted index
    t2 = time.time()
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
                    
    print(f"Multi-key index built in {time.time()-t2:.2f}s.", flush=True)
    
    # Clean S1
    t3 = time.time()
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
    
    print(f"Querying, feature extraction & LightGBM reranking for {len(s1_ids)} entities in {country}...", flush=True)
    t4 = time.time()
    
    match_rows = []
    cand_rows = []
    
    BATCH_SIZE = 2500
    for batch_start in range(0, len(s1_ids), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(s1_ids))
        
        batch_feats = []
        batch_cand_map = [] # stores (idx_in_batch, tgt_id)
        entity_cands_map = defaultdict(list)
        
        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            n = s1_names[idx]
            a = s1_addrs[idx]
            rn = s1_raw_names[idx]
            nums = [num.lstrip("0") for num in s1_nums_raw[idx] if num.lstrip("0")] if s1_nums_raw[idx] else []
            words = [w for w in n.split() if len(w) >= 3 and w not in LEGAL_STOPS] if n else []
            awords = [w for w in a.split() if len(w) >= 4 and w not in GENERIC_ADDR] if a else []
            
            cand_indices = set()
            if len(words) >= 2:
                w_sorted = sorted(words[:3])
                p = idx_w2_sorted.get(f"{w_sorted[0]}_{w_sorted[1]}")
                if p: cand_indices.update(p[:40])
                if len(words) >= 3:
                    p2 = idx_w2_sorted.get(f"{w_sorted[0]}_{w_sorted[2]}")
                    if p2: cand_indices.update(p2[:20])
                    p3 = idx_w2_sorted.get(f"{w_sorted[1]}_{w_sorted[2]}")
                    if p3: cand_indices.update(p3[:20])
            if len(words) >= 1 and nums:
                for num in nums[:2]:
                    p = idx_w1_num.get(f"{words[0]}_{num}")
                    if p: cand_indices.update(p[:30])
            if awords and nums:
                for aw in awords[:3]:
                    for num in nums[:2]:
                        p = idx_aw_num.get(f"{aw}_{num}")
                        if p: cand_indices.update(p[:30])
            if len(cand_indices) < 20 and len(words) >= 1:
                p = idx_w1.get(words[0])
                if p and len(p) <= 300:
                    cand_indices.update(p[:40])
                    
            if not cand_indices:
                continue
                
            c_list = list(cand_indices)
            if len(c_list) > 25:
                quick_scores = []
                for c_idx in c_list:
                    tg_n = tgt_names[c_idx]
                    tg_a = tgt_addrs[c_idx]
                    ns = fuzz.token_sort_ratio(n, tg_n) if n and tg_n else 0
                    as_ = fuzz.token_set_ratio(a, tg_a) if a and tg_a else 0
                    quick_scores.append((c_idx, max(ns, as_)))
                quick_scores.sort(key=lambda x: x[1], reverse=True)
                c_list = [x[0] for x in quick_scores[:25]]
                
            s1_ng = get_char_ngrams(n, 3)
            s1_ang = get_char_ngrams(a, 3)
            s1_compact = "".join(c for c in n.lower() if c.isalnum())
            s1_num_set = set(nums)
            s1_is_non_ascii = any(ord(c) > 127 for c in rn)
            
            for c_idx in c_list:
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
                        
                batch_feats.append([
                    n_sort, n_set, a_set, a_sort, n_jacc, a_jacc, 
                    num_overlap, num_jacc, is_non_ascii, domain_match, 
                    abs(len(n) - len(tg_n)), abs(len(a) - len(tg_a))
                ])
                batch_cand_map.append((idx, tgt_ids[c_idx]))
                
        # Batch Predict with LightGBM
        if batch_feats:
            feats_arr = np.array(batch_feats, dtype=np.float32)
            probs = booster.predict(feats_arr)
            for i, p_val in enumerate(probs):
                entity_idx, tid = batch_cand_map[i]
                entity_cands_map[entity_idx].append((tid, p_val))
                
        # Format rows for this batch
        for idx in range(batch_start, batch_end):
            sid = s1_ids[idx]
            cands = entity_cands_map.get(idx, [])
            if not cands:
                match_rows.append(f"{sid}\t\n")
                cand_rows.append(f"{sid}\t\n")
            else:
                cands.sort(key=lambda x: x[1], reverse=True)
                top_cands = [c[0] for c in cands[:MAX_CANDIDATES]]
                matched_cands = [c[0] for c in cands if c[1] >= SCORE_THRESHOLD]
                match_rows.append(f"{sid}\t{','.join(matched_cands)}\n")
                cand_rows.append(f"{sid}\t{','.join(top_cands)}\n")
                
        if len(match_rows) >= 50000:
            f_match.writelines(match_rows)
            f_cand.writelines(cand_rows)
            match_rows.clear()
            cand_rows.clear()
            print(f"  Processed {batch_end}/{len(s1_ids)} entities ({time.time()-t4:.1f}s)...", flush=True)
            
    if match_rows:
        f_match.writelines(match_rows)
        f_cand.writelines(cand_rows)
        match_rows.clear()
        cand_rows.clear()
        
    print(f"Finished {country} in {time.time()-t4:.2f}s ({len(s1_ids)/(time.time()-t4):.1f} entities/sec).", flush=True)
    del tgt_ids, tgt_names, tgt_addrs, tgt_nums_raw, tgt_raw_names, idx_w2_sorted, idx_w1, idx_w1_num, idx_aw_num
    del s1_ids, s1_names, s1_addrs, s1_nums_raw, s1_raw_names
    gc.collect()


def main():
    print("=== STARTING LIGHTGBM PRODUCTION INFERENCE ===", flush=True)
    t_start = time.time()
    
    os.makedirs("output", exist_ok=True)
    
    print(f"Loading LightGBM model from {MODEL_PATH}...", flush=True)
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    with open(MATCHING_OUT, "w", encoding="utf-8") as f_match, \
         open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for country in ["France", "US", "India"]:
            run_country_inference(country, booster, f_match, f_cand)
            
    print(f"\nAll countries processed in {time.time()-t_start:.2f}s!", flush=True)
    print("Files written:")
    print(f"  {MATCHING_OUT} ({os.path.getsize(MATCHING_OUT):,} bytes)")
    print(f"  {CANDIDATE_OUT} ({os.path.getsize(CANDIDATE_OUT):,} bytes)")
    
    print("\nValidating generated submission files with check_submission.py...", flush=True)
    import check_submission
    is_valid = check_submission.check_submission_files(
        matching_path=MATCHING_OUT,
        candidate_path=CANDIDATE_OUT,
        test_dir=TEST_DIR,
        check_ids=False
    )
    if is_valid:
        print("\n=== LIGHTGBM SUBMISSION GENERATED AND FULLY VALIDATED! ===", flush=True)
    else:
        print("\nERROR: Submission validation FAILED!", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
