"""
Local Laptop GPU (RTX 3050) Full Test Embedding Generation.
Model: intfloat/multilingual-e5-base (MIT, 278M params).
Encodes all test records for France, US, India.
Features:
  - fp16 precision on CUDA (RTX 3050)
  - Chunked saving (every 50,000 records) for 100% resumability
  - Saves .npy embeddings and .json ID order files per country
"""
import os
import sys
import gc
import json
import time
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

OUTPUT_DIR = "outputs/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_in_chunks(model, texts, output_npy_path, batch_size=256, chunk_size=50000):
    total = len(texts)
    dim = 768
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    
    if os.path.exists(output_npy_path):
        try:
            existing = np.load(output_npy_path, mmap_mode="r")
            if len(existing) == total:
                print(f"  [Resume] {output_npy_path} already complete ({total:,} records). Skipping.")
                return
        except Exception:
            pass
            
    part_path = output_npy_path + ".mmap.npy"
    prog_path = output_npy_path + ".prog"
    
    # Open or create memmap array
    if os.path.exists(part_path):
        print(f"  [Memmap] Opening existing memmap: {part_path}")
        mmap_arr = np.lib.format.open_memmap(part_path, mode="r+")
    else:
        print(f"  [Memmap] Creating new memmap on disk: {part_path} (shape: {total}x{dim}, float16)")
        mmap_arr = np.lib.format.open_memmap(part_path, mode="w+", dtype=np.float16, shape=(total, dim))
        
    last_prog = 0
    if os.path.exists(prog_path):
        try:
            with open(prog_path, "r") as pf:
                last_prog = int(pf.read().strip())
        except Exception:
            last_prog = 0
            
    t0 = time.time()
    for start_idx in range(0, total, chunk_size):
        end_idx = min(start_idx + chunk_size, total)
        chunk_file = f"{output_npy_path}.chunk_{start_idx}_{end_idx}.npy"
        
        # Check if already recorded in progress
        if end_idx <= last_prog:
            continue
            
        # Check if legacy chunk file exists on disk
        if os.path.exists(chunk_file):
            print(f"  [Resume] Loading legacy chunk {start_idx:,}..{end_idx:,} from {chunk_file} into memmap")
            c_data = np.load(chunk_file)
            mmap_arr[start_idx:end_idx] = c_data
            del c_data
            try:
                os.remove(chunk_file)
            except Exception:
                pass
            with open(prog_path, "w") as pf:
                pf.write(str(end_idx))
            continue
            
        # Encode chunk
        chunk_texts = texts[start_idx:end_idx]
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float16)
        
        mmap_arr[start_idx:end_idx] = chunk_emb
        del chunk_emb
        mmap_arr.flush()
        
        with open(prog_path, "w") as pf:
            pf.write(str(end_idx))
            
        speed = len(chunk_texts) / (time.time() - t0 + 1e-6)
        print(f"  Encoded {start_idx:,}..{end_idx:,}/{total:,} ({speed:.0f} rec/s)")
        t0 = time.time()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    mmap_arr.flush()
    del mmap_arr
    gc.collect()
    
    if os.path.exists(output_npy_path):
        os.remove(output_npy_path)
    os.rename(part_path, output_npy_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)
    print(f"  --> Saved full embeddings: {output_npy_path} ({os.path.getsize(output_npy_path) / 1024 / 1024:.1f} MB)")

def main():
    print("=" * 70)
    print("STARTING LOCAL GPU FULL TEST EMBEDDINGS (intfloat/multilingual-e5-base)")
    print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    print("=" * 70)
    
    model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
    model.half()
    
    test_dir = "student_resource/dataset/test"
    print("Loading test data...")
    s1_df = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")
    tgt_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    print(f"Loaded: S1 = {len(s1_df):,} | Targets = {len(tgt_df):,}")
    
    # Order: France first (fast test), then US, then India
    countries = ["France", "US", "India"]
    
    for country in countries:
        print(f"\n{'='*60}")
        print(f"PROCESSING {country.upper()}")
        print(f"{'='*60}")
        
        # 1. Source 1
        c_s1 = s1_df.filter(pl.col("country") == country)
        s1_ids = c_s1["entity_id"].to_list()
        s1_texts = [f"query: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in c_s1.iter_rows(named=True)]
        
        s1_out = os.path.join(OUTPUT_DIR, f"{country}_s1_emb.npy")
        s1_ids_out = os.path.join(OUTPUT_DIR, f"{country}_s1_ids.json")
        with open(s1_ids_out, "w") as f:
            json.dump(s1_ids, f)
            
        print(f"Encoding {len(s1_texts):,} Source 1 records for {country}...")
        encode_in_chunks(model, s1_texts, s1_out)
        
        # 2. Targets (S2 + S3)
        c_tgt = tgt_df.filter(pl.col("country") == country)
        tgt_ids = c_tgt["entity_id"].to_list()
        tgt_texts = [f"passage: {r['business_name'] or ''} | {r['business_address'] or ''} | {country}".strip() for r in c_tgt.iter_rows(named=True)]
        
        tgt_out = os.path.join(OUTPUT_DIR, f"{country}_tgt_emb.npy")
        tgt_ids_out = os.path.join(OUTPUT_DIR, f"{country}_tgt_ids.json")
        with open(tgt_ids_out, "w") as f:
            json.dump(tgt_ids, f)
            
        print(f"Encoding {len(tgt_texts):,} Target records for {country}...")
        encode_in_chunks(model, tgt_texts, tgt_out)
        
    print("\n" + "=" * 70)
    print("ALL TEST EMBEDDINGS SUCCESSFULLY GENERATED ON LAPTOP GPU!")
    print("=" * 70)

if __name__ == "__main__":
    main()
