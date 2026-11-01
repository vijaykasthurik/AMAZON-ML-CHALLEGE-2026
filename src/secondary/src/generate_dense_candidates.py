"""
Track B4: High-Throughput GPU Streaming Dense Candidate Retrieval.
Uses intfloat/multilingual-e5-base embeddings precomputed on disk (mmap mode).
Streams:
  - Queries in batches of 5,000
  - Targets in chunks of 100,000
Computes exact fp16 cosine top-k on RTX 3050 Laptop GPU.
Saves:
  outputs/candidates/{country}_dense_top8.parquet
"""
import os
import sys
import gc
import json
import time
import numpy as np
import polars as pl
import torch

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

EMB_DIR = "outputs/embeddings"
OUT_DIR = "outputs/candidates"
os.makedirs(OUT_DIR, exist_ok=True)

def process_country_dense_candidates(country: str, K: int = 8, batch_q: int = 5000, chunk_tgt: int = 100000):
    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING GPU DENSE CANDIDATE RETRIEVAL: {country.upper()}", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()
    
    s1_path = os.path.join(EMB_DIR, f"{country}_s1_emb.npy")
    s1_ids_path = os.path.join(EMB_DIR, f"{country}_s1_ids.json")
    tgt_path = os.path.join(EMB_DIR, f"{country}_tgt_emb.npy")
    tgt_ids_path = os.path.join(EMB_DIR, f"{country}_tgt_ids.json")
    
    out_file = os.path.join(OUT_DIR, f"{country}_dense_top8.parquet")
    if os.path.exists(out_file):
        print(f"Candidates for {country} already exist at {out_file}, skipping!", flush=True)
        return
        
    if not (os.path.exists(s1_path) and os.path.exists(tgt_path)):
        print(f"Embeddings for {country} not found!", flush=True)
        return
        
    s1_mmap = np.load(s1_path, mmap_mode='r')
    tgt_mmap = np.load(tgt_path, mmap_mode='r')
    
    with open(s1_ids_path, 'r', encoding='utf-8') as f:
        s1_ids = json.load(f)
    with open(tgt_ids_path, 'r', encoding='utf-8') as f:
        tgt_ids = json.load(f)
        
    N_s1, dim = s1_mmap.shape
    N_tgt = tgt_mmap.shape[0]
    print(f"  Loaded: S1 = {N_s1:,} queries | Targets = {N_tgt:,} items (dim={dim})", flush=True)
    
    # Pre-allocate output arrays
    all_top_indices = np.zeros((N_s1, K), dtype=np.int32)
    all_top_scores = np.zeros((N_s1, K), dtype=np.float16)
    
    n_tgt_chunks = (N_tgt + chunk_tgt - 1) // chunk_tgt
    n_q_batches = (N_s1 + batch_q - 1) // batch_q
    
    print(f"  Starting retrieval: {n_q_batches} query batches x {n_tgt_chunks} target chunks...", flush=True)
    
    for q_b in range(n_q_batches):
        q_start = q_b * batch_q
        q_end = min(q_start + batch_q, N_s1)
        curr_batch_size = q_end - q_start
        
        q_tensor = torch.from_numpy(np.array(s1_mmap[q_start:q_end], dtype=np.float16)).cuda()
        
        best_scores = torch.full((curr_batch_size, K), -float('inf'), dtype=torch.float16, device='cuda')
        best_indices = torch.full((curr_batch_size, K), -1, dtype=torch.long, device='cuda')
        
        t_batch = time.time()
        for t_c in range(n_tgt_chunks):
            t_start_idx = t_c * chunk_tgt
            t_end_idx = min(t_start_idx + chunk_tgt, N_tgt)
            
            tgt_tensor = torch.from_numpy(np.array(tgt_mmap[t_start_idx:t_end_idx], dtype=np.float16)).cuda()
            
            sim = torch.matmul(q_tensor, tgt_tensor.T)
            del tgt_tensor
            
            c_top_scores, c_top_indices = torch.topk(sim, min(K, sim.shape[1]), dim=1)
            del sim
            c_global_indices = c_top_indices + t_start_idx
            
            merged_scores = torch.cat([best_scores, c_top_scores], dim=1)
            merged_indices = torch.cat([best_indices, c_global_indices], dim=1)
            
            best_scores, rank_idx = torch.topk(merged_scores, K, dim=1)
            best_indices = torch.gather(merged_indices, 1, rank_idx)
            
        torch.cuda.synchronize()
        all_top_indices[q_start:q_end] = best_indices.cpu().numpy()
        all_top_scores[q_start:q_end] = best_scores.cpu().numpy()
        
        del q_tensor, best_scores, best_indices
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t_batch
        q_per_sec = curr_batch_size / elapsed
        if (q_b + 1) % 5 == 0 or (q_b + 1) == n_q_batches:
            print(f"    Batch {q_b+1}/{n_q_batches} complete ({q_end:,}/{N_s1:,} queries, {q_per_sec:.0f} q/s)", flush=True)
            
    print(f"  All {N_s1:,} queries processed in {time.time()-t_start:.1f}s!", flush=True)
    
    # Save as parquet
    print("  Mapping indices to Target IDs and saving parquet...", flush=True)
    tgt_id_arr = np.array(tgt_ids)
    
    s1_id_col = []
    top_tgt_ids_col = []
    scores_col = []
    
    for i in range(N_s1):
        sid = s1_ids[i]
        top_tids = tgt_id_arr[all_top_indices[i]].tolist()
        top_sc = all_top_scores[i].tolist()
        s1_id_col.append(sid)
        top_tgt_ids_col.append(top_tids)
        scores_col.append(top_sc)
        
    out_df = pl.DataFrame({
        "source1_entity_id": s1_id_col,
        "dense_top_target_ids": top_tgt_ids_col,
        "dense_top_scores": scores_col,
    })
    
    out_file = os.path.join(OUT_DIR, f"{country}_dense_top8.parquet")
    out_df.write_parquet(out_file)
    print(f"  --> Saved {out_file} ({os.path.getsize(out_file)/1024/1024:.1f} MB)", flush=True)

def main():
    countries = ["France", "US", "India"]
    for c in countries:
        process_country_dense_candidates(c, K=8)
    print("\nALL COUNTRIES DENSE RETRIEVAL COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
