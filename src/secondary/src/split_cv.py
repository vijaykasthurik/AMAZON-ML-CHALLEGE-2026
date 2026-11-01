"""
Phase 1: Validation Split Generator.
Creates a stratified 5-fold CV mapping and a fast representative validation split.
Stratification is done jointly on country and match count bucket (singletons, 1-2 matches, 3+ matches).
"""

import sys
import os
import polars as pl
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("=== CREATING VALIDATION SPLITS ===")

os.makedirs("data/splits", exist_ok=True)

# Load S1 train and ground truth
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t").select(["entity_id", "country"])
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")

# Compute match count per S1 entity
mids = gt["matched_entity_ids"].fill_null("").to_list()
counts = np.array([len(x.split(",")) if x else 0 for x in mids], dtype=np.int32)

# Create match count bucket: 0 (singleton), 1, 2, 3, 4, 5+
bucket = np.clip(counts, 0, 5)

# Build pandas dataframe for stratification
df_strat = pd.DataFrame({
    "source1_entity_id": s1["entity_id"].to_list(),
    "country": s1["country"].to_list(),
    "match_count": counts,
    "bucket": bucket,
})

df_strat["strat_key"] = df_strat["country"].astype(str) + "_" + df_strat["bucket"].astype(str)

print("Stratification key distribution:")
print(df_strat["strat_key"].value_counts())

# Generate reproducible 5 folds
np.random.seed(42)
df_strat["fold"] = -1
for key, group in df_strat.groupby("strat_key"):
    n = len(group)
    idx = group.index.values
    perm = np.random.permutation(n)
    folds = perm % 5
    df_strat.loc[idx, "fold"] = folds

print("\nFold distribution:")
print(pd.crosstab(df_strat["fold"], df_strat["country"]))

# Save 5-fold CV mapping
cv_path = "data/splits/train_cv_5fold.parquet"
df_strat[["source1_entity_id", "country", "match_count", "fold"]].to_parquet(cv_path, index=False)
print(f"Saved full 5-fold CV mapping to {cv_path}")

# Create mini validation set (Fold 0 sample of 20,000 S1 records, stratified)
mini_size = 20000
mini_indices = []
for key, group in df_strat[df_strat["fold"] == 0].groupby("strat_key"):
    k_size = int(np.round(mini_size * len(group) / len(df_strat[df_strat["fold"] == 0])))
    k_size = max(1, min(k_size, len(group)))
    mini_indices.extend(group.sample(n=k_size, random_state=42).index.values)

df_mini = df_strat.loc[mini_indices].reset_index(drop=True)
mini_path = "data/splits/val_mini_20k.parquet"
df_mini[["source1_entity_id", "country", "match_count"]].to_parquet(mini_path, index=False)
print(f"Saved fast validation split (20k rows) to {mini_path}")
print(f"  Mini validation countries:\n{df_mini['country'].value_counts()}")
print(f"  Mini validation singletons: {(df_mini['match_count'] == 0).sum()} ({(df_mini['match_count'] == 0).mean()*100:.2f}%)")

print("=== VALIDATION SPLITS READY ===")
