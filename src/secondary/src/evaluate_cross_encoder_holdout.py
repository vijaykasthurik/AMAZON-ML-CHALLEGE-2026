"""
Evaluate Cross-Encoder (BAAI/bge-reranker-v2-m3) on Calibrated Holdout Split.
Holdout Use #2 (of 3 allowed).

Target: Weighted Macro F0.5 >= 0.98999
Baseline (EXP_006): 0.91384 (US: 0.94574, India: 0.88774)

Pipeline:
  1. Load calibrated holdout S1 entities and targets (density = 5.75)
  2. Extract candidate pairs via multi-key lexical blocking + high-overlap fuzzy search
  3. Score candidate pairs with fine-tuned bge-reranker-v2-m3 (fp16 on RTX 3050 GPU)
  4. Apply optimal thresholds: tau_US = 0.75, tau_IN = 0.70
  5. Apply global mutual exclusivity (greedy match resolution)
  6. Score using src/score_holdout.py and record in calibrated_holdout_use_log.jsonl
"""
import os
import sys
import gc
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from rapidfuzz import fuzz
from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    name_tokens_no_stops,
    build_inverted_index, query_index
)
from score_holdout import score as run_holdout_scorer

MODEL_DIR = r"outputs/kaggle/cross_encoder/bge_reranker_model"
HOLDOUT_DIR = r"data/calibrated_holdout"
OUTPUT_DIR = r"output/holdout_cross_encoder"

class RerankerDataset(Dataset):
    def __init__(self, queries, passages):
        self.queries = queries
        self.passages = passages
        
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        return self.queries[idx], self.passages[idx]

def reranker_collate(batch, tokenizer, max_len=96):
    queries = [item[0] for item in batch]
    passages = [item[1] for item in batch]
    enc = tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc

def score_candidates_with_cross_encoder(pairs_df: pl.DataFrame, model, tokenizer, batch_size=32, device="cuda"):
    print(f"  Scoring {len(pairs_df):,} candidate pairs on {device} (batch_size={batch_size})...", flush=True)
    t0 = time.time()
    
    dataset = RerankerDataset(pairs_df["query"].to_list(), pairs_df["passage"].to_list())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: reranker_collate(b, tokenizer, max_len=96)
    )
    
    probs = []
    model.eval()
    with torch.no_grad():
        for step, enc in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast('cuda'):
                logits = model(**enc).logits.squeeze(-1)
                p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p if p.ndim > 0 else [float(p)])
            
            if (step + 1) % 1000 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - t0
                speed = len(probs) / elapsed
                print(f"    Scored {len(probs):,}/{len(dataset):,} pairs ({speed:.0f} pairs/s)...", flush=True)
                
    return np.array(probs, dtype=np.float32)

def evaluate_holdout_cross_encoder(
    tau_us: float = 0.75,
    tau_in: float = 0.70,
    batch_size: int = 32,
    dry_run: bool = False
):
    print("=" * 70, flush=True)
    print("HOLDOUT USE #2: BGE-RERANKER-V2-M3 EVALUATION", flush=True)
    print(f"Thresholds: US = {tau_us:.2f} | India = {tau_in:.2f}", flush=True)
    print("=" * 70, flush=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading cross-encoder model from {MODEL_DIR}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    print("Model loaded successfully!", flush=True)
    
    countries = ["US", "India"]
    thresholds = {"US": tau_us, "India": tau_in}
    all_matching_lines = []
    
    for country in countries:
        print(f"\n--- Processing Holdout: {country} ---", flush=True)
        t_country = time.time()
        tau = thresholds[country]
        
        # 1. Load S1
        s1 = pl.read_csv(os.path.join(HOLDOUT_DIR, "test_source1.tsv"), separator="\t").filter(
            pl.col("country") == country
        )
        s1_ids = s1["entity_id"].to_list()
        s1_raw_names = s1["business_name"].fill_null("").to_list()
        s1_raw_addrs = s1["business_address"].fill_null("").to_list()
        N_s1 = len(s1_ids)
        print(f"  Holdout S1 entities: {N_s1:,}", flush=True)
        
        # 2. Load Targets
        s2 = pl.scan_csv(os.path.join(HOLDOUT_DIR, "test_source2.tsv"), separator="\t").filter(
            pl.col("country") == country
        ).collect()
        s3 = pl.scan_csv(os.path.join(HOLDOUT_DIR, "test_source3.tsv"), separator="\t").filter(
            pl.col("country") == country
        ).collect()
        tgt = pl.concat([s2, s3])
        del s2, s3
        gc.collect()
        
        tgt_ids = tgt["entity_id"].to_list()
        tgt_raw_names = tgt["business_name"].fill_null("").to_list()
        tgt_raw_addrs = tgt["business_address"].fill_null("").to_list()
        N_tgt = len(tgt_ids)
        print(f"  Holdout Targets: {N_tgt:,} (density = {N_tgt/N_s1:.2f} targets/S1)", flush=True)
        
        # 3. Clean and build index
        print("  Cleaning and building lexical index...", flush=True)
        tgt_clean_names = [normalize_name(n) for n in tgt_raw_names]
        tgt_clean_addrs = [normalize_name(a) for a in tgt_raw_addrs]
        tgt_nums = [[s for s in a.split() if s.isdigit()] for a in tgt_clean_addrs]
        
        idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_clean_names, tgt_clean_addrs, tgt_nums)
        
        idx_trans = defaultdict(list)
        if country == "India":
            for i, rn in enumerate(tgt_raw_names):
                if rn and any(ord(c) > 127 for c in rn):
                    trans = normalize_name(transliterate(rn))
                    if LOANWORD_MAP:
                        trans = apply_loanword_map(trans, LOANWORD_MAP)
                    tw = name_tokens_no_stops(trans)
                    if tw:
                        idx_trans[tw[0]].append(i)
                        if len(tw) >= 2:
                            k2 = f"{min(tw[0], tw[1])}_{max(tw[0], tw[1])}"
                            idx_trans[k2].append(i)
                            
        # 4. Generate pairs
        print("  Generating candidate pairs for S1 entities...", flush=True)
        pair_s1_ids = []
        pair_tgt_ids = []
        pair_queries = []
        pair_passages = []
        
        s1_clean_names = [normalize_name(n) for n in s1_raw_names]
        s1_clean_addrs = [normalize_name(a) for a in s1_raw_addrs]
        s1_nums = [[s for s in a.split() if s.isdigit()] for a in s1_clean_addrs]
        
        for i in range(N_s1):
            sid = s1_ids[i]
            cn = s1_clean_names[i]
            ca = s1_clean_addrs[i]
            cnums = s1_nums[i]
            rn = s1_raw_names[i]
            ra = s1_raw_addrs[i]
            
            cands = query_index(cn, ca, cnums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)
            if country == "India" and any(ord(c) > 127 for c in rn):
                trans_s1 = normalize_name(transliterate(rn))
                if LOANWORD_MAP:
                    trans_s1 = apply_loanword_map(trans_s1, LOANWORD_MAP)
                tw = name_tokens_no_stops(trans_s1)
                if tw:
                    if len(tw) >= 2:
                        k2 = f"{min(tw[0], tw[1])}_{max(tw[0], tw[1])}"
                        cands.update(idx_trans.get(k2, []))
                    cands.update(idx_trans.get(tw[0], []))
                    
            if len(cands) > 1:
                scored = []
                for c_idx in cands:
                    sc = fuzz.token_sort_ratio(cn, tgt_clean_names[c_idx])
                    if country == "India" and any(ord(c) > 127 for c in rn):
                        sc = max(sc, fuzz.token_sort_ratio(trans_s1, tgt_clean_names[c_idx]))
                    scored.append((sc, c_idx))
                scored.sort(key=lambda x: x[0], reverse=True)
                promising = [c[1] for c in scored if c[0] >= 45]
                if not promising:
                    promising = [c[1] for c in scored[:4]]
                cands = promising[:8]
                
            q_text = f"{rn} {ra}".strip()
            for c_idx in cands:
                pair_s1_ids.append(sid)
                pair_tgt_ids.append(tgt_ids[c_idx])
                pair_queries.append(q_text)
                pair_passages.append(f"{tgt_raw_names[c_idx]} {tgt_raw_addrs[c_idx]}".strip())
                
        pairs_df = pl.DataFrame({
            "s1_id": pair_s1_ids,
            "tgt_id": pair_tgt_ids,
            "query": pair_queries,
            "passage": pair_passages
        })
        print(f"  Generated {len(pairs_df):,} candidate pairs for {N_s1:,} S1 entities ({len(pairs_df)/N_s1:.2f} pairs/S1)", flush=True)
        
        # 5. Score with cross-encoder
        scores = score_candidates_with_cross_encoder(pairs_df, model, tokenizer, batch_size=batch_size, device=device)
        pairs_df = pairs_df.with_columns(pl.Series("score", scores))
        
        # Save scored pairs parquet for instant threshold tuning
        cache_pq = os.path.join(OUTPUT_DIR, f"{country}_scored_pairs.parquet")
        pairs_df.write_parquet(cache_pq)
        print(f"  Saved raw scored pairs to {cache_pq}", flush=True)
        
        # 6. Apply threshold and global mutual exclusivity
        print(f"  Filtering at tau={tau:.2f} and applying global mutual exclusivity...", flush=True)
        filtered = pairs_df.filter(pl.col("score") >= tau).sort("score", descending=True)
        
        matched_pairs = []
        assigned_targets = set()
        
        for row in filtered.iter_rows(named=True):
            sid = row["s1_id"]
            tid = row["tgt_id"]
            if tid in assigned_targets:
                continue
            assigned_targets.add(tid)
            matched_pairs.append((sid, tid))
            
        s1_to_matches = defaultdict(list)
        for sid, tid in matched_pairs:
            s1_to_matches[sid].append(tid)
            
        n_matched_s1 = len(s1_to_matches)
        print(f"  Matched S1 entities: {n_matched_s1:,}/{N_s1:,} ({n_matched_s1/N_s1*100:.1f}%) | Singletons: {N_s1-n_matched_s1:,} ({(N_s1-n_matched_s1)/N_s1*100:.2f}%)", flush=True)
        
        for sid in s1_ids:
            all_matching_lines.append(f"{sid}\t{','.join(s1_to_matches.get(sid, []))}\n")
            
        del pairs_df, filtered
        gc.collect()
        print(f"  {country} completed in {time.time()-t_country:.1f}s!", flush=True)
        
    out_matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    print(f"\nWriting holdout predictions to {out_matching_file}...", flush=True)
    with open(out_matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(all_matching_lines)
    print(f"Saved {out_matching_file} ({os.path.getsize(out_matching_file)/1024/1024:.1f} MB)", flush=True)
    
    # 7. Score with official holdout scorer
    print("\n" + "=" * 70, flush=True)
    print("CALLING OFFICIAL HOLDOUT SCORER (src/score_holdout.py)...", flush=True)
    print("=" * 70, flush=True)
    run_holdout_scorer(version="bge_reranker_v2_holdout", pred_path=out_matching_file, dry_run=dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-us", type=float, default=0.75)
    parser.add_argument("--tau-in", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Score without consuming an official holdout attempt")
    args = parser.parse_args()
    
    evaluate_holdout_cross_encoder(
        tau_us=args.tau_us,
        tau_in=args.tau_in,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
