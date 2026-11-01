"""
Prepares the minimal PRIVATE dataset for Kaggle.
Columns per table:
  - entity_id
  - country
  - clean_text: "{name} | {address} | {country}" (normalized, Indic transliterated for cross-script)
Format: snappy/zstd compressed Parquet.
Generates dataset-metadata.json with isPrivate=True.
"""
import os
import sys
import gc
import json
import time
import polars as pl
import text_unidecode

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("src")
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_lgbm_v2 import normalize_name, apply_loanword_map, LOANWORD_MAP

OUT_DIR = "kaggle_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def transliterate_indic(text: str) -> str:
    if text and any(ord(c) > 127 for c in text):
        trans = text_unidecode.unidecode(text).lower()
        if LOANWORD_MAP:
            trans = apply_loanword_map(trans, LOANWORD_MAP)
        return trans
    return text

def process_tsv(path: str, out_name: str):
    print(f"Processing {path} -> {out_name}...")
    t0 = time.time()
    
    # Read TSV
    df = pl.scan_csv(path, separator="\t").select([
        pl.col("entity_id"),
        pl.col("country"),
        pl.col("business_name").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("clean_name"),
        pl.col("business_address").fill_null("").map_elements(normalize_name, return_dtype=pl.Utf8).alias("clean_addr"),
        pl.col("business_name").fill_null("").map_elements(transliterate_indic, return_dtype=pl.Utf8).alias("trans_name"),
    ]).collect()
    
    # Build unified text column
    # For Indic records, append transliterated name if different
    df = df.with_columns(
        pl.when((pl.col("trans_name") != "") & (pl.col("trans_name") != pl.col("clean_name")))
          .then(pl.concat_str([pl.col("clean_name"), pl.lit(" ("), pl.col("trans_name"), pl.lit(") | "), pl.col("clean_addr"), pl.lit(" | "), pl.col("country")]))
          .otherwise(pl.concat_str([pl.col("clean_name"), pl.lit(" | "), pl.col("clean_addr"), pl.lit(" | "), pl.col("country")]))
          .alias("text")
    ).select(["entity_id", "country", "text"])
    
    out_path = os.path.join(OUT_DIR, out_name)
    df.write_parquet(out_path, compression="zstd")
    print(f"  Saved {len(df):,} rows to {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s)")
    del df
    gc.collect()

def main():
    print("=" * 70)
    print("PREPARING MINIMAL KAGGLE DATASET")
    print("=" * 70)
    
    # Process test set first (needed for Track A embeddings)
    process_tsv("student_resource/dataset/test/test_source1.tsv", "test_source1.parquet")
    process_tsv("student_resource/dataset/test/test_source2.tsv", "test_source2.parquet")
    process_tsv("student_resource/dataset/test/test_source3.tsv", "test_source3.parquet")
    
    # Process calibrated DEV split for fast benchmarking on Kaggle
    for country in ["US", "India"]:
        split_dir = f"data/calibrated_dev/{country}_density_5.75"
        if os.path.exists(split_dir):
            process_tsv(os.path.join(split_dir, "test_source1.tsv"), f"dev_{country}_s1.parquet")
            # Combine S2 and S3 for targets
            s2 = pl.read_csv(os.path.join(split_dir, "test_source2.tsv"), separator="\t")
            s3 = pl.read_csv(os.path.join(split_dir, "test_source3.tsv"), separator="\t")
            tgt = pl.concat([s2, s3]).unique(subset=["entity_id"])
            tgt_tmp = os.path.join(split_dir, "tgt_combined.tsv")
            tgt.write_csv(tgt_tmp, separator="\t")
            process_tsv(tgt_tmp, f"dev_{country}_targets.parquet")
            if os.path.exists(tgt_tmp):
                os.remove(tgt_tmp)
            # Copy ground truth
            gt = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
            gt.write_parquet(os.path.join(OUT_DIR, f"dev_{country}_gt.parquet"))
            
    # Metadata for Kaggle API
    meta = {
        "title": "Amazon ML Challenge 2026 Entity Resolution Minimal",
        "id": "dinnypaulnavis/amazon-ml-2026-minimal",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True
    }
    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nMetadata written: {os.path.join(OUT_DIR, 'dataset-metadata.json')} (isPrivate=True)")
    total_size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"Total dataset package size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
