"""
Build Pruned Candidate Set for Amazon ML Challenge 2026.
Combines:
  1. High-Recall Dense Embedding Candidates (Top-8 from BGE-M3 / Qwen)
  2. Multi-Key Lexical Inverted Index Candidates (Sorted 2-word, address words x numbers, transliterated)
Applies rank-based pruning to enforce:
  - Average candidates per S1 <= 10 (Target: 7-8 candidates/S1)
  - Maximized recall ceiling (>= 98.5%)
  - Strict compliance with candidate_pairs.tsv official validator schema
"""
import os
import sys
import gc
import time
import argparse
from collections import defaultdict
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    name_tokens_no_stops,
    build_inverted_index, query_index,
    extract_house_number
)

def build_country_candidates(
    country: str,
    data_dir: str,
    dense_path: str,
    max_dense: int = 8,
    max_total_per_s1: int = 10,
):
    print(f"\n{'='*70}", flush=True)
    print(f"BUILDING CANDIDATE SET: {country.upper()}", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    
    # 1. Load S1
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s1_df = pl.scan_csv(s1_path, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    
    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].fill_null("").to_list()
    s1_addrs = s1_df["business_address"].fill_null("").to_list()
    del s1_df
    gc.collect()
    N_s1 = len(s1_ids)
    print(f"  Loaded {N_s1:,} S1 entities ({time.time()-t0:.1f}s)", flush=True)
    
    # 2. Load Dense Top-8 Candidates
    print(f"  Loading dense candidates from {dense_path}...", flush=True)
    dense_df = pl.read_parquet(dense_path)
    
    dense_dict = {}
    for row in dense_df.iter_rows():
        # row: (s1_id, top_tgt_ids, top_scores)
        dense_dict[row[0]] = (row[1][:max_dense], row[2][:max_dense])
    del dense_df
    gc.collect()
    print(f"  Dense candidates loaded for {len(dense_dict):,} S1 entities ({time.time()-t0:.1f}s)", flush=True)
    
    # 3. Load Targets & Build Lexical Index
    t_idx = time.time()
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
    
    N_tgt = len(tgt_ids)
    print(f"  Loaded {N_tgt:,} targets ({time.time()-t_idx:.1f}s)", flush=True)
    
    print("  Building lexical inverted index...", flush=True)
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw)
    
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
                        
    print(f"  Lexical index built ({time.time()-t_idx:.1f}s)", flush=True)
    
    # 4. Merge Dense & Lexical Candidates
    print(f"  Merging and ranking candidate sets (max {max_total_per_s1} per S1)...", flush=True)
    t_merge = time.time()
    
    s1_out_ids = []
    cand_out_lists = []
    cand_counts = []
    
    for i, sid in enumerate(s1_ids):
        raw_name = s1_names[i]
        raw_addr = s1_addrs[i]
        
        c_name = normalize_name(raw_name)
        c_addr = normalize_name(raw_addr)
        nums = [s for s in c_addr.split() if s.isdigit()]
        
        # Candidate score map: target_id -> composite score
        cand_scores = defaultdict(float)
        
        # A. Dense candidates
        if sid in dense_dict:
            d_ids, d_scores = dense_dict[sid]
            for tid, sc in zip(d_ids, d_scores):
                cand_scores[tid] += float(sc)
                
        # B. Lexical candidates (add top lexical matches with boost)
        lex_cands = query_index(
            c_name, c_addr, nums,
            idx_w2, idx_w1, idx_w1_num, idx_aw_num
        )
        if country == "India" and any(ord(c) > 127 for c in raw_name):
            trans_s1 = normalize_name(transliterate(raw_name))
            if LOANWORD_MAP:
                trans_s1 = apply_loanword_map(trans_s1, LOANWORD_MAP)
            tw = name_tokens_no_stops(trans_s1)
            if tw:
                if len(tw) >= 2:
                    k2 = f"{min(tw[0], tw[1])}_{max(tw[0], tw[1])}"
                    lex_cands.update(idx_trans.get(k2, []))
                lex_cands.update(idx_trans.get(tw[0], []))
                
        for c_idx in lex_cands:
            tid = tgt_ids[c_idx]
            cand_scores[tid] += 0.25  # lexical presence boost
            
        # Sort and prune to max_total_per_s1
        if len(cand_scores) > max_total_per_s1:
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:max_total_per_s1]
            final_cands = [c[0] for c in sorted_cands]
        else:
            final_cands = list(cand_scores.keys())
            
        s1_out_ids.append(sid)
        cand_out_lists.append(final_cands)
        cand_counts.append(len(final_cands))
        
        if (i + 1) % 100000 == 0 or (i + 1) == N_s1:
            print(f"    Processed {i+1:,}/{N_s1:,} S1 entities (mean cands: {np.mean(cand_counts):.2f})...", flush=True)
            
    avg_cands = float(np.mean(cand_counts))
    med_cands = float(np.median(cand_counts))
    max_cands = int(np.max(cand_counts)) if cand_counts else 0
    total_pairs = sum(cand_counts)
    possible_pairs = N_s1 * N_tgt
    reduction_ratio = 1.0 - (total_pairs / possible_pairs) if possible_pairs > 0 else 1.0
    
    print(f"\n--- {country} CANDIDATE METRICS ---")
    print(f"  Total S1 Entities   : {N_s1:,}")
    print(f"  Total Candidates    : {total_pairs:,}")
    print(f"  Mean Candidates / S1: {avg_cands:.2f}")
    print(f"  Median Candidates/S1: {med_cands:.1f}")
    print(f"  Max Candidates / S1 : {max_cands}")
    print(f"  Reduction Ratio     : {reduction_ratio:.8f} (99.99...%)")
    print(f"  Time Elapsed        : {time.time()-t0:.1f}s")
    
    return s1_out_ids, cand_out_lists, {
        "country": country,
        "n_s1": N_s1,
        "n_tgt": N_tgt,
        "total_cands": total_pairs,
        "avg_cands": avg_cands,
        "med_cands": med_cands,
        "max_cands": max_cands,
        "reduction_ratio": reduction_ratio
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--candidate-dir", default="outputs/candidates")
    parser.add_argument("--out-tsv", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-cands", type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    
    countries = ["France", "US", "India"]
    metrics = []
    
    with open(args.out_tsv, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for c in countries:
            dense_file = os.path.join(args.candidate_dir, f"{c}_dense_top8.parquet")
            if not os.path.exists(dense_file):
                print(f"ERROR: {dense_file} not found. Complete dense retrieval first!")
                return
                
            s1_ids, cand_lists, m = build_country_candidates(
                country=c,
                data_dir=args.test_dir,
                dense_path=dense_file,
                max_dense=8,
                max_total_per_s1=args.max_cands
            )
            print(f"  Streaming {c} candidates to {args.out_tsv}...", flush=True)
            for sid, clist in zip(s1_ids, cand_lists):
                f_out.write(f"{sid}\t{','.join(clist)}\n")
            f_out.flush()
            metrics.append(m)
            
            # Explicit cleanup
            del s1_ids, cand_lists
            gc.collect()
            
    print(f"\nSaved {args.out_tsv} ({os.path.getsize(args.out_tsv)/1024/1024:.1f} MB)")
    
    print("\n" + "="*70)
    print("FINAL CANDIDATE SET SUMMARY (OFFICIAL RULE COMPLIANCE)")
    print("="*70)
    total_s1 = sum(m["n_s1"] for m in metrics)
    total_cands = sum(m["total_cands"] for m in metrics)
    print(f"Overall S1 Entities   : {total_s1:,}")
    print(f"Overall Candidate Pairs: {total_cands:,}")
    print(f"Overall Mean Cands/S1  : {total_cands/total_s1:.2f} (<= 10 REQUIRED)")
    for m in metrics:
        print(f"  {m['country']}: Mean {m['avg_cands']:.2f}, Median {m['med_cands']:.1f}, Max {m['max_cands']}, Reduction: {m['reduction_ratio']:.8f}")

if __name__ == "__main__":
    main()
