"""
Local Benchmark of Qwen/Qwen3-Embedding-0.6B on RTX 3050 Laptop GPU.
Evaluates on India Calibrated DEV Split (10k S1, 57.5k targets).
Measures Recall@10, @20, @50, Speed, and VRAM.
"""
import os
import sys
import time
import json
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 70)
    print("BENCHMARKING Qwen/Qwen3-Embedding-0.6B on RTX 3050 GPU")
    print("=" * 70)
    
    split_dir = "data/calibrated_dev/India_density_5.75"
    s1_df = pl.read_csv(os.path.join(split_dir, "test_source1.tsv"), separator="\t")
    tgt_df = pl.concat([
        pl.read_csv(os.path.join(split_dir, "test_source2.tsv"), separator="\t"),
        pl.read_csv(os.path.join(split_dir, "test_source3.tsv"), separator="\t")
    ]).unique(subset=["entity_id"])
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    
    gt_dict = {}
    total_positives = 0
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        t_set = set(x.strip() for x in m.split(",")) if m else set()
        gt_dict[r["source1_entity_id"]] = t_set
        total_positives += len(t_set)
        
    s1_texts = [f"{r['business_name'] or ''} | {r['business_address'] or ''} | India".strip() for r in s1_df.iter_rows(named=True)]
    tgt_texts = [f"{r['business_name'] or ''} | {r['business_address'] or ''} | India".strip() for r in tgt_df.iter_rows(named=True)]
    tgt_ids = tgt_df["entity_id"].to_list()
    s1_ids = s1_df["entity_id"].to_list()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device, trust_remote_code=True)
    if device == "cuda":
        model.half()
        
    print(f"Model loaded in {time.time()-t0:.1f}s.")
    
    # Encode targets
    t_tgt = time.time()
    tgt_emb = model.encode(tgt_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    tgt_speed = len(tgt_texts) / (time.time() - t_tgt)
    print(f"Targets encoded: {len(tgt_texts):,} in {time.time()-t_tgt:.1f}s ({tgt_speed:.0f} rec/s)")
    
    # Encode S1
    t_s1 = time.time()
    s1_emb = model.encode(s1_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    s1_speed = len(s1_texts) / (time.time() - t_s1)
    print(f"S1 encoded: {len(s1_texts):,} in {time.time()-t_s1:.1f}s ({s1_speed:.0f} rec/s)")
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Peak VRAM: {peak_vram:.0f} MB")
    
    # Cosine top-k search on GPU
    with torch.no_grad():
        tgt_tensor = torch.tensor(tgt_emb, device=device, dtype=torch.float16)
        s1_tensor = torch.tensor(s1_emb, device=device, dtype=torch.float16)
        sims = torch.mm(s1_tensor, tgt_tensor.T)
        topk_vals, topk_indices = torch.topk(sims, k=50, dim=1)
        I = topk_indices.cpu().numpy()
        del tgt_tensor, s1_tensor, sims, topk_vals, topk_indices
        torch.cuda.empty_cache()
        
    recalls = {10: 0, 20: 0, 50: 0}
    for i, sid in enumerate(s1_ids):
        true_tgts = gt_dict.get(sid, set())
        if not true_tgts:
            continue
        cands = [tgt_ids[idx] for idx in I[i]]
        for k in recalls:
            recalls[k] += len(true_tgts & set(cands[:k]))
            
    r10 = recalls[10] / total_positives
    r20 = recalls[20] / total_positives
    r50 = recalls[50] / total_positives
    
    print("\n" + "=" * 70)
    print(f"Qwen/Qwen3-Embedding-0.6B RESULTS (India):")
    print(f"  Recall@10 : {r10*100:.2f}%")
    print(f"  Recall@20 : {r20*100:.2f}%")
    print(f"  Recall@50 : {r50*100:.2f}%")
    print(f"  Speed     : {tgt_speed:.0f} rec/s")
    print(f"  Peak VRAM : {peak_vram:.0f} MB")
    print("=" * 70)
    
    res = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "country": "India",
        "recall@10": r10,
        "recall@20": r20,
        "recall@50": r50,
        "speed": tgt_speed,
        "peak_vram_mb": peak_vram
    }
    with open("outputs/qwen_benchmark_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
