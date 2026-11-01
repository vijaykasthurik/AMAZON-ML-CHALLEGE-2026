"""
Measure Recall-vs-Candidate-Set-Size Curve.
Evaluates on Calibrated DEV Split (10k S1, 57.5k targets, 5.75 density) for US and India:
  - Lexical blocking alone (w2, w1_num, aw_num, pin_w1, transliterated_w2)
  - Dense retrieval alone (multilingual-e5-base at K = 1, 2, 3, 5, 7, 10, 15, 20)
  - Hybrid Union (Lexical + Dense) pruned by cheap similarity score
Reports:
  - Recall ceiling
  - Average candidates per S1
  - Reduction ratio
Target: average candidates <= 10 with recall >= 97%.
"""
import os
import sys
import time
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
import faiss

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("src")
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    get_char_ngrams, jaccard, name_tokens_no_stops,
    query_index, build_inverted_index, get_name_freq,
)

def evaluate_curve(country="India"):
    print(f"\n{'='*70}")
    print(f"RECALL VS CANDIDATE SET SIZE CURVE: {country}")
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
    
    # ── 1. DENSE EMBEDDING RETRIEVAL ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding with intfloat/multilingual-e5-base on {device}...")
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    if device == "cuda":
        model.half()
        
    s1_texts = [f"query: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in s1_df.iter_rows(named=True)]
    tgt_texts = [f"passage: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in tgt_df.iter_rows(named=True)]
    
    tgt_emb = model.encode(tgt_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    s1_emb = model.encode(s1_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    
    dim = tgt_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tgt_emb.astype(np.float32))
    
    max_k = 25
    D, I = index.search(s1_emb.astype(np.float32), max_k)
    
    # Dense recall at each K
    print(f"\n--- Dense Retrieval Alone ({country}) ---")
    print(f"Pool size: {len(tgt_ids):,} targets, {len(s1_ids):,} S1 entities (Density: {len(tgt_ids)/len(s1_ids):.2f})")
    print(f"{'K':<5s} | {'Avg Cands':<10s} | {'Recall':<10s} | {'Reduction Ratio':<15s}")
    print("-" * 50)
    for k in [1, 2, 3, 5, 7, 8, 10, 12, 15, 20]:
        hits = sum(len(gt_dict.get(s1_ids[i], set()) & {tgt_ids[idx] for idx in I[i][:k]}) for i in range(len(s1_ids)))
        rec = hits / total_positives
        red = (1.0 - (k / len(tgt_ids))) * 100
        print(f"{k:<5d} | {k:<10.1f} | {rec*100:<9.2f}% | {red:.2f}%")
        
    # ── 2. LEXICAL BLOCKING RETRIEVAL ─────────────────────────────────────────────
    tgt_names = [normalize_name(r["business_name"] or "") for r in tgt_df.iter_rows(named=True)]
    tgt_addrs = [normalize_name(r["business_address"] or "") for r in tgt_df.iter_rows(named=True)]
    tgt_nums = [r["business_address"].split() if r["business_address"] else [] for r in tgt_df.iter_rows(named=True)]
    tgt_raw = [r["business_name"] or "" for r in tgt_df.iter_rows(named=True)]
    
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums)
    
    idx_trans = {}
    if country == "India":
        from collections import defaultdict
        idx_trans = defaultdict(list)
        for i, rn in enumerate(tgt_raw):
            if rn and any(ord(c) > 127 for c in rn):
                trans = normalize_name(transliterate(rn))
                if LOANWORD_MAP:
                    trans = apply_loanword_map(trans, LOANWORD_MAP)
                ws = name_tokens_no_stops(trans)
                if ws:
                    idx_trans[ws[0]].append(i)
                    if len(ws) >= 2:
                        idx_trans[f"{ws[0]}_{ws[1]}"].append(i)

    # ── 3. HYBRID PRUNING CURVE ───────────────────────────────────────────────────
    # For each S1: take top-k dense candidates + top lexical candidates, then rank by combined cosine score
    print(f"\n--- Hybrid (Dense + Lexical) Pruning Curve ({country}) ---")
    print(f"{'Target Max Cands':<18s} | {'Actual Avg Cands':<18s} | {'Recall':<10s} | {'Reduction Ratio':<15s}")
    print("-" * 65)
    
    for max_cands in [3, 5, 7, 8, 10, 12, 15]:
        total_cands = 0
        hits = 0
        for i, sid in enumerate(s1_ids):
            true_tgts = gt_dict.get(sid, set())
            
            # Dense candidates (top 8)
            cand_indices = set(I[i][:8])
            
            # Lexical candidates (top 15)
            s1_row = s1_df[i]
            s1n = normalize_name(s1_row["business_name"][0] or "")
            s1a = normalize_name(s1_row["business_address"][0] or "")
            s1nums = [x.lstrip("0") for x in (s1_row["business_address"][0] or "").split() if x.isdigit()]
            
            lex_cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)
            cand_indices.update(list(lex_cands)[:15])
            
            # Score all candidates using dense dot product / cosine
            scored = []
            for ci in cand_indices:
                score = float(np.dot(s1_emb[i], tgt_emb[ci]))
                scored.append((ci, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            top_final = [tgt_ids[ci] for ci, _ in scored[:max_cands]]
            total_cands += len(top_final)
            hits += len(true_tgts & set(top_final))
            
        avg_c = total_cands / len(s1_ids)
        rec = hits / total_positives
        red = (1.0 - (avg_c / len(tgt_ids))) * 100
        print(f"{max_cands:<18d} | {avg_c:<18.2f} | {rec*100:<9.2f}% | {red:.2f}%")

if __name__ == "__main__":
    evaluate_curve("India")
    evaluate_curve("US")
