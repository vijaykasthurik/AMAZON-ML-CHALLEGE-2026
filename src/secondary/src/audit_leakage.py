"""
Phase 1: Leakage Audit and Data Structure Analysis.
Checks ID correlations, row order correlations, train-test overlaps, and duplicate patterns.
"""

import sys
import os
import polars as pl
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("=== STARTING LEAKAGE AUDIT ===")

# 1. Check ID structure and row order correlation
gt_sample = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=200000)
gt_sample = gt_sample.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
gt_pairs = gt_sample.with_columns(
    pl.col("matched_entity_ids").str.split(",")
).explode("matched_entity_ids").rename({"matched_entity_ids": "target_id"})

# Extract numerical parts of IDs
gt_pairs = gt_pairs.with_columns([
    pl.col("source1_entity_id").str.slice(3).cast(pl.Int64, strict=False).alias("s1_num"),
    pl.col("target_id").str.slice(3).cast(pl.Int64, strict=False).alias("target_num"),
    pl.col("target_id").str.slice(0, 2).alias("target_src"),
])

# Correlation between numerical IDs
corr_s2 = gt_pairs.filter(pl.col("target_src") == "S2").select(["s1_num", "target_num"]).to_pandas().corr().iloc[0, 1]
corr_s3 = gt_pairs.filter(pl.col("target_src") == "S3").select(["s1_num", "target_num"]).to_pandas().corr().iloc[0, 1]

print(f"Numerical ID correlation (S1 vs S2): {corr_s2:.4f}")
print(f"Numerical ID correlation (S1 vs S3): {corr_s3:.4f}")

# Check difference |s1_num - target_num|
diff_s2 = (gt_pairs.filter(pl.col("target_src") == "S2")["s1_num"] - gt_pairs.filter(pl.col("target_src") == "S2")["target_num"]).abs()
print(f"ID absolute diff mean: {diff_s2.mean():.1f}, min: {diff_s2.min()}, max: {diff_s2.max()}")

# 2. Train-Test Overlap check (Do any entity IDs in test exist in train?)
print("\nChecking train/test entity ID overlap...")
tr_s1_ids = set(pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
te_s1_ids = set(pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
s1_overlap = tr_s1_ids.intersection(te_s1_ids)
print(f"Overlap between Train S1 and Test S1 IDs: {len(s1_overlap)}")

tr_s2_ids = set(pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
te_s2_ids = set(pl.read_csv("student_resource/dataset/test/test_source2.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
s2_overlap = tr_s2_ids.intersection(te_s2_ids)
print(f"Overlap between Train S2 and Test S2 IDs: {len(s2_overlap)}")

tr_s3_ids = set(pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
te_s3_ids = set(pl.read_csv("student_resource/dataset/test/test_source3.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
s3_overlap = tr_s3_ids.intersection(te_s3_ids)
print(f"Overlap between Train S3 and Test S3 IDs: {len(s3_overlap)}")

# 3. Exact matching statistics on training sample
print("\nAnalyzing exact normalized name / address matching rates...")
# Sample 50,000 S1 and find their ground truth targets
sample_s1_ids = set(gt_sample["source1_entity_id"][:50000].to_list())
s1_sub = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(sample_s1_ids))
gt_sub = gt_pairs.filter(pl.col("source1_entity_id").is_in(sample_s1_ids))

sample_target_ids = set(gt_sub["target_id"].to_list())
s2_sub = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(sample_target_ids))
s3_sub = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(sample_target_ids))
s23_sub = pl.concat([s2_sub, s3_sub])

pairs_df = gt_sub.select(["source1_entity_id", "target_id"]).join(
    s1_sub.select(["entity_id", "business_name", "business_address"]),
    left_on="source1_entity_id", right_on="entity_id"
).rename({"business_name": "s1_name", "business_address": "s1_addr"}).join(
    s23_sub.select(["entity_id", "business_name", "business_address"]),
    left_on="target_id", right_on="entity_id"
).rename({"business_name": "tgt_name", "business_address": "tgt_addr"})

# Check exact matches
pairs_df = pairs_df.with_columns([
    (pl.col("s1_name").str.to_lowercase().str.strip_chars() == pl.col("tgt_name").str.to_lowercase().str.strip_chars()).alias("exact_name"),
    (pl.col("s1_addr").str.to_lowercase().str.strip_chars() == pl.col("tgt_addr").str.to_lowercase().str.strip_chars()).alias("exact_addr"),
])

print(f"Sample positive pairs analyzed: {pairs_df.shape[0]}")
print(f"Exact Lowercase Name Match: {pairs_df['exact_name'].mean()*100:.2f}%")
print(f"Exact Lowercase Address Match: {pairs_df['exact_addr'].mean()*100:.2f}%")
print(f"Exact Name OR Address Match: {(pairs_df['exact_name'] | pairs_df['exact_addr']).mean()*100:.2f}%")
print(f"Exact Name AND Address Match: {(pairs_df['exact_name'] & pairs_df['exact_addr']).mean()*100:.2f}%")

print("\n=== LEAKAGE AUDIT COMPLETE ===")
