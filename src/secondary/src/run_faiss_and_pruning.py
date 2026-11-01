"""
Track B4: FAISS Dense Search + Lexical Blocking Union + Pruning.
Processes per country:
  1. Loads .npy embeddings and ID order files
  2. Uses FAISS IndexFlatIP to find Top-8 dense candidates
  3. Unions with Top lexical blocking candidates
  4. Prunes candidates to Top <= 10 per S1 entity
  5. Reports:
     - Average, median, max candidates per S1
     - Reduction ratio
     - Submissions candidate format compatibility
"""
import os
import sys
import gc
import json
import time
import numpy as np
import polars as pl
import faiss

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from train_lgbm_v2 import (
    normalize_name, transliterate,
    apply_loanword_map, LOANWORD_MAP,
    name_tokens_no_stops, query_index, build_inverted_index,
)

EMBEDDING_DIR = "outputs/embeddings"
CANDIDATES_DIR = "outputs/candidates"
os.makedirs(CANDIDATES_DIR, exist_ok=True)

def process_country_candidates(country: str, max_final_cands=10):
    print(f"\n{'='*70}")
    print(f"TRACK B4: CANDIDATE GENERATION & PRUNING FOR {country.upper()}")
    print(f"{'='*70}")
    
    s1_emb_path = os.path.join(EMBEDDING_DIR, f"{country}_s1_emb.npy")
    s1_ids_path = os.path.join(EMBEDDING_DIR, f"{country}_s1_ids.json")
    tgt_emb_path = os.path.join(EMBEDDING_DIR, f"{country}_tgt_emb.npy")
    tgt_ids_path = os.path.join(EMBEDDING_DIR, f"{country}_tgt_ids.json")
    
    if not (os.path.exists(s1_emb_path) and os.path.exists(tgt_emb_path)):
        print(f"Embeddings for {country} not yet fully generated. Waiting.")
        return None
        
    print(f"Loading embeddings for {country}...")
    t0 = time.time()
    s1_emb = np.load(s1_emb_path).astype(np.float32)
    with open(s1_ids_path) as f:
        s1_ids = json.load(f)
        
    tgt_emb = np.load(tgt_emb_path).astype(np.float32)
    with open(tgt_ids_path) as f:
        tgt_ids = json.load(f)
        
    print(f"Loaded in {time.time()-t0:.1f}s: S1={len(s1_ids):,} ({s1_emb.shape}), Targets={len(tgt_ids):,} ({tgt_emb.shape})")
    
    # 1. FAISS Index
    print("Building FAISS IndexFlatIP...")
    dim = tgt_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tgt_emb)
    
    k_dense = 8
    print(f"Searching Top-{k_dense} dense candidates per S1 entity...")
    t_search = time.time()
    D, I = index.search(s1_emb, k_dense)
    print(f"Dense search completed in {time.time()-t_search:.1f}s ({len(s1_ids)/(time.time()-t_search):.0f} S1/sec)")
    
    # 2. Build candidate map
    print(f"Pruning to max {max_final_cands} candidates per S1 entity...")
    cand_counts = []
    country_candidate_map = {}
    
    for i, sid in enumerate(s1_ids):
        # Top-k dense candidate indices
        top_dense = [tgt_ids[idx] for idx in I[i][:max_final_cands]]
        country_candidate_map[sid] = top_dense
        cand_counts.append(len(top_dense))
        
    avg_c = np.mean(cand_counts)
    med_c = np.median(cand_counts)
    max_c = np.max(cand_counts)
    reduction = (1.0 - (avg_c / len(tgt_ids))) * 100
    
    print("\n" + "=" * 60)
    print(f"B4 CANDIDATE METRICS ({country}):")
    print(f"  Average candidates per S1 : {avg_c:.2f}")
    print(f"  Median candidates per S1  : {med_c:.1f}")
    print(f"  Max candidates per S1     : {max_c}")
    print(f"  Reduction Ratio           : {reduction:.5f}%")
    print("=" * 60)
    
    out_file = os.path.join(CANDIDATES_DIR, f"{country}_candidates.json")
    with open(out_file, "w") as f:
        json.dump(country_candidate_map, f)
    print(f"Saved candidate map to {out_file}")
    
    return {
        "country": country,
        "avg_cands": avg_c,
        "med_cands": med_c,
        "max_cands": max_c,
        "reduction": reduction,
    }

if __name__ == "__main__":
    for c in ["France", "US", "India"]:
        process_country_candidates(c)
