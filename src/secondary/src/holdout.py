"""
Create and lock the 10% holdout split.

Rules:
- Grouped by source1_entity_id (no S1 entity leaks across train/holdout).
- Stratified by country × match_count_bucket (same as 5-fold CV).
- Holdout fraction: exactly 10%.
- SEED = 2026 (fixed, never changed).
- Holdout IDs written to data/splits/holdout_ids.parquet and must never be
  used for training, feature design, threshold tuning, or dictionary mining.
- 5-fold CV is computed on the remaining 90% only.
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

HOLDOUT_SEED = 2026
HOLDOUT_FRAC = 0.10
N_FOLDS = 5

os.makedirs("data/splits", exist_ok=True)

HOLDOUT_IDS_PATH = "data/splits/holdout_ids.parquet"
DEV_CV_PATH = "data/splits/dev_cv_5fold.parquet"

# If already locked, do not overwrite — just validate and exit.
if os.path.exists(HOLDOUT_IDS_PATH):
    df = pl.read_parquet(HOLDOUT_IDS_PATH)
    print(f"[holdout.py] Holdout already locked: {len(df)} entities in {HOLDOUT_IDS_PATH}")
    print(f"  Country distribution:")
    if "country" in df.columns:
        for r in df.group_by("country").agg(pl.count()).sort("country").iter_rows(named=True):
            print(f"    {r['country']}: {r['count']}")
    sys.exit(0)

print("[holdout.py] Creating locked holdout split...")

# Load S1 and ground truth
s1 = pl.read_csv(
    "student_resource/dataset/train/train_source1.tsv", separator="\t"
).select(["entity_id", "country"])
gt = pl.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv", separator="\t"
).select(["source1_entity_id", "matched_entity_ids"])

# Build match count per S1
gt_dict = {}
for r in gt.iter_rows(named=True):
    m = r["matched_entity_ids"]
    gt_dict[r["source1_entity_id"]] = len(m.split(",")) if m else 0

s1_list = s1["entity_id"].to_list()
country_list = s1["country"].to_list()
match_counts = np.array([gt_dict.get(sid, 0) for sid in s1_list], dtype=np.int32)
bucket = np.clip(match_counts, 0, 5)

df_strat = pd.DataFrame({
    "source1_entity_id": s1_list,
    "country": country_list,
    "match_count": match_counts,
    "bucket": bucket,
})
df_strat["strat_key"] = df_strat["country"] + "_" + df_strat["bucket"].astype(str)
df_strat["split"] = "dev"  # will be overwritten for holdout

# For each stratification key, take 10% as holdout
rng = np.random.default_rng(HOLDOUT_SEED)
holdout_idx = []
for key, group in df_strat.groupby("strat_key"):
    n = len(group)
    n_holdout = max(1, int(round(n * HOLDOUT_FRAC)))
    chosen = rng.choice(group.index.values, size=n_holdout, replace=False)
    holdout_idx.extend(chosen.tolist())

df_strat.loc[holdout_idx, "split"] = "holdout"

holdout_df = df_strat[df_strat["split"] == "holdout"].copy()
dev_df = df_strat[df_strat["split"] == "dev"].copy()

print(f"Total S1 entities: {len(df_strat)}")
print(f"  Holdout: {len(holdout_df)} ({len(holdout_df)/len(df_strat)*100:.1f}%)")
print(f"  Dev    : {len(dev_df)} ({len(dev_df)/len(df_strat)*100:.1f}%)")
print(f"Country distribution:")
for c, cnt in df_strat[df_strat['split']=='holdout'].groupby('country').size().items():
    print(f"  holdout/{c}: {cnt}")
for c, cnt in df_strat[df_strat['split']=='dev'].groupby('country').size().items():
    print(f"  dev/{c}: {cnt}")

# Save holdout
pl.from_pandas(holdout_df[["source1_entity_id", "country", "match_count"]]).write_parquet(HOLDOUT_IDS_PATH)
print(f"\nHoldout IDs saved to {HOLDOUT_IDS_PATH}")

# Create 5-fold CV on dev only
dev_df = dev_df.copy()
dev_df["fold"] = -1
rng2 = np.random.default_rng(42)  # separate seed for CV
for key, group in dev_df.groupby("strat_key"):
    n = len(group)
    idx = group.index.values
    perm = rng2.permutation(n)
    dev_df.loc[idx, "fold"] = perm % N_FOLDS

print(f"\nDev 5-fold distribution:")
print(pd.crosstab(dev_df["fold"], dev_df["country"]))

pl.from_pandas(
    dev_df[["source1_entity_id", "country", "match_count", "fold"]]
).write_parquet(DEV_CV_PATH)
print(f"\nDev 5-fold CV mapping saved to {DEV_CV_PATH}")
print("[holdout.py] Done.")
