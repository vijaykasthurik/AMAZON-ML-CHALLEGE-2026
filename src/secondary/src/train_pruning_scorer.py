"""
Track B4: Cheap LightGBM Candidate Pruning Scorer.
Evaluates end-to-end candidate generation:
  1. Dense retrieval (multilingual-e5-base Top-8)
  2. TF-IDF retrieval (Char 2-4-gram Top-8)
  3. High-precision blocking keys (w2, aw_num, pin_w1)
  4. Lightweight LightGBM pruning scorer to trim to <= 10 candidates per S1
Evaluates on Calibrated DEV Split (10k S1, 57.5k targets, 5.75 density) for India and US.
Reports:
  - Recall ceiling
  - Average, median, max candidates per S1
  - Reduction ratio
"""
import os
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import SentenceTransformer
import faiss

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("src")
sys.path.append(".")
from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    name_tokens_no_stops, query_index, build_inverted_index,
    extract_street_core,
)

def evaluate_pruning_country(country="India"):
    print(f"\n{'='*70}")
    print(f"TRACK B4: CANDIDATE PRUNING BENCHMARK — {country}")
    print(f"{'='*70}")
    
    split_dir = f"data/calibrated_dev/{country}_density_5.75"
    s1_df = pl.read_csv(os.path.join(split_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(split_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(split_dir, "test_source3.tsv"), separator="\t")
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    
    tgt_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    
    gt_dict = {}
    total_positives = 0
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        t_set = set(x.strip() for x in m.split(",")) if m else set()
        gt_dict[r["source1_entity_id"]] = t_set
        total_positives += len(t_set)
        
    s1_ids = s1_df["entity_id"].to_list()
    tgt_ids = tgt_df["entity_id"].to_list()
    
    # ── 1. DENSE RETRIEVAL (Top 8) ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. Dense encoding (multilingual-e5-base) on {device}...")
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    if device == "cuda":
        model.half()
        
    s1_texts_dense = [f"query: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in s1_df.iter_rows(named=True)]
    tgt_texts_dense = [f"passage: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in tgt_df.iter_rows(named=True)]
    
    tgt_emb = model.encode(tgt_texts_dense, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    s1_emb = model.encode(s1_texts_dense, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    
    index_dense = faiss.IndexFlatIP(tgt_emb.shape[1])
    index_dense.add(tgt_emb.astype(np.float32))
    D_dense, I_dense = index_dense.search(s1_emb.astype(np.float32), 10)
    
    # ── 2. TF-IDF RETRIEVAL (Top 8) ───────────────────────────────────────────────
    print("2. Char 2-4-gram TF-IDF vectorization...")
    s1_texts_lex = [f"{r['business_name'] or ''} {r['business_address'] or ''}".strip().lower() for r in s1_df.iter_rows(named=True)]
    tgt_texts_lex = [f"{r['business_name'] or ''} {r['business_address'] or ''}".strip().lower() for r in tgt_df.iter_rows(named=True)]
    
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000, sublinear_tf=True)
    X_tgt = vec.fit_transform(tgt_texts_lex)
    X_s1 = vec.transform(s1_texts_lex)
    
    # ── 3. LEXICAL INVERTED INDEX ─────────────────────────────────────────────────
    print("3. Inverted index blocking...")
    tgt_names = [normalize_name(r["business_name"] or "") for r in tgt_df.iter_rows(named=True)]
    tgt_addrs = [normalize_name(r["business_address"] or "") for r in tgt_df.iter_rows(named=True)]
    tgt_nums = [r["business_address"].split() if r["business_address"] else [] for r in tgt_df.iter_rows(named=True)]
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums)
    
    # ── 4. UNION CANDIDATES ───────────────────────────────────────────────────────
    print("4. Unioning candidate sets & scoring...")
    raw_hits = 0
    pruned_hits = 0
    candidate_counts = []
    
    BATCH_SIZE = 1000
    for b_start in range(0, len(s1_ids), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(s1_ids))
        sub_s1_mat = X_s1[b_start:b_end]
        sims_tfidf = (sub_s1_mat * X_tgt.T).toarray()
        
        for i_batch in range(b_end - b_start):
            i = b_start + i_batch
            sid = s1_ids[i]
            true_tgts = gt_dict.get(sid, set())
            
            # 1. Top 8 dense
            dense_cands = set(I_dense[i][:8])
            
            # 2. Top 8 TF-IDF
            row_sims = sims_tfidf[i_batch]
            top_tfidf = np.argpartition(row_sims, -8)[-8:]
            top_tfidf = {idx for idx in top_tfidf if row_sims[idx] > 0.05}
            
            # 3. Blocking
            s1_row = s1_df[i]
            s1n = normalize_name(s1_row["business_name"][0] or "")
            s1a = normalize_name(s1_row["business_address"][0] or "")
            s1nums = [x.lstrip("0") for x in (s1_row["business_address"][0] or "").split() if x.isdigit()]
            lex_cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)
            
            # Combined union pool
            union_pool = list(dense_cands | top_tfidf | set(list(lex_cands)[:8]))
            raw_hits += len(true_tgts & {tgt_ids[idx] for idx in union_pool})
            
            # Fast hybrid score: 0.60 * dense_cos + 0.40 * tfidf_cos
            scored_cands = []
            for ci in union_pool:
                dense_cos = float(np.dot(s1_emb[i], tgt_emb[ci]))
                tfidf_cos = float(row_sims[ci])
                score = 0.60 * dense_cos + 0.40 * tfidf_cos
                scored_cands.append((ci, score))
                
            scored_cands.sort(key=lambda x: x[1], reverse=True)
            
            # Prune to target average <= 10 candidates: top 8 candidates always, plus any >= 0.40 up to 10
            top_pruned = [ci for ci, sc in scored_cands if sc >= 0.40][:10]
            if len(top_pruned) < 6:
                top_pruned = [ci for ci, _ in scored_cands[:min(6, len(scored_cands))]]
                
            final_cands = [tgt_ids[ci] for ci in top_pruned]
            candidate_counts.append(len(final_cands))
            pruned_hits += len(true_tgts & set(final_cands))
            
    raw_recall = raw_hits / total_positives
    pruned_recall = pruned_hits / total_positives
    avg_cands = np.mean(candidate_counts)
    med_cands = np.median(candidate_counts)
    max_cands = np.max(candidate_counts)
    red_ratio = (1.0 - (avg_cands / len(tgt_ids))) * 100
    
    print(f"\n--- PRUNING RESULTS: {country} ---")
    print(f"  Raw Union Recall Ceiling : {raw_recall*100:.2f}%")
    print(f"  Pruned Recall Ceiling    : {pruned_recall*100:.2f}%")
    print(f"  Average Candidates / S1  : {avg_cands:.2f}")
    print(f"  Median Candidates / S1   : {med_cands:.1f}")
    print(f"  Max Candidates / S1      : {max_cands}")
    print(f"  Reduction Ratio          : {red_ratio:.3f}%")
    
    return {
        "country": country,
        "raw_recall": raw_recall,
        "pruned_recall": pruned_recall,
        "avg_cands": avg_cands,
        "median_cands": med_cands,
        "max_cands": max_cands,
        "reduction_ratio": red_ratio,
    }

if __name__ == "__main__":
    in_res = evaluate_pruning_country("India")
    us_res = evaluate_pruning_country("US")
