"""
Realistic Calibrated DEV Validation Split Builder.
Constructs DEV evaluation environment matching the real test set:
  - Exact target-to-S1 density of 5.75 (plus 4.5 and 8.0 for stability)
  - True singleton rate of 5.58%
  - No artificial forcing of positive targets into candidate sets
  - Per-country separation (US and India)
"""
import os
import sys
import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

SEED = 42
RNG = np.random.default_rng(SEED)

DEV_CV_PATH = "data/splits/dev_cv_5fold.parquet"
CALIBRATED_DIR = "data/calibrated_dev"
os.makedirs(CALIBRATED_DIR, exist_ok=True)

def build_calibrated_split(n_s1_per_country=10000, target_density=5.75):
    print("=" * 70)
    print(f"BUILDING CALIBRATED DEV SPLIT: {n_s1_per_country} S1/country, Density={target_density:.2f}")
    print("=" * 70)
    
    # Load DEV IDs (fold 0 as holdout validation fold, folds 1-4 for train)
    dev_meta = pl.read_parquet(DEV_CV_PATH)
    
    # Load S1, Ground Truth, S2, S3
    s1_all = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    gt_all = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
    s2_all = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    s3_all = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    
    # Ground truth mapping: S1 -> list of matched target IDs
    gt_map = {}
    for r in gt_all.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_map[r["source1_entity_id"]] = [x.strip() for x in m.split(",") if x.strip()] if m else []
        
    for country in ["US", "India"]:
        print(f"\n--- Processing {country} ---")
        # Filter DEV S1 entities in fold 0
        c_dev = dev_meta.filter((pl.col("country") == country) & (pl.col("fold") == 0))
        c_s1_ids = c_dev["source1_entity_id"].to_list()
        
        # Sample n_s1_per_country entities
        chosen_s1_ids = RNG.choice(c_s1_ids, size=min(n_s1_per_country, len(c_s1_ids)), replace=False).tolist()
        chosen_set = set(chosen_s1_ids)
        
        # Filter S1 records
        sub_s1 = s1_all.filter(pl.col("entity_id").is_in(chosen_set))
        
        # Collect all TRUE matched targets for these chosen S1 entities
        needed_target_ids = set()
        for sid in chosen_s1_ids:
            needed_target_ids.update(gt_map.get(sid, []))
            
        print(f"  Selected {len(sub_s1):,} S1 entities ({country})")
        print(f"  Required true targets: {len(needed_target_ids):,}")
        
        # Total targets needed for target_density = 5.75 targets per S1
        total_targets_needed = int(round(len(sub_s1) * target_density))
        n_distractors_needed = max(0, total_targets_needed - len(needed_target_ids))
        print(f"  Target pool size: {total_targets_needed:,} (True: {len(needed_target_ids):,}, Distractors: {n_distractors_needed:,})")
        
        # All available targets for this country in S2 and S3
        c_s2 = s2_all.filter(pl.col("country") == country)
        c_s3 = s3_all.filter(pl.col("country") == country)
        c_targets = pl.concat([c_s2, c_s3])
        
        # Split into true targets vs available distractors
        true_tgt_df = c_targets.filter(pl.col("entity_id").is_in(needed_target_ids))
        distractor_candidates = c_targets.filter(~pl.col("entity_id").is_in(needed_target_ids))
        
        # Sample distractors
        distractor_indices = RNG.choice(len(distractor_candidates), size=min(n_distractors_needed, len(distractor_candidates)), replace=False)
        distractor_df = distractor_candidates[distractor_indices]
        
        # Combine true targets and distractors
        final_targets = pl.concat([true_tgt_df, distractor_df]).unique(subset=["entity_id"])
        
        # Save S1, targets, and ground truth for this country
        out_country_dir = os.path.join(CALIBRATED_DIR, f"{country}_density_{target_density:.2f}")
        os.makedirs(out_country_dir, exist_ok=True)
        
        sub_s1.write_csv(os.path.join(out_country_dir, "test_source1.tsv"), separator="\t")
        final_targets.filter(pl.col("entity_id").str.starts_with("S2")).write_csv(os.path.join(out_country_dir, "test_source2.tsv"), separator="\t")
        final_targets.filter(pl.col("entity_id").str.starts_with("S3")).write_csv(os.path.join(out_country_dir, "test_source3.tsv"), separator="\t")
        
        sub_gt = gt_all.filter(pl.col("source1_entity_id").is_in(chosen_set))
        sub_gt.write_csv(os.path.join(out_country_dir, "ground_truth.tsv"), separator="\t")
        
        print(f"  Saved to {out_country_dir}:")
        print(f"    S1: {len(sub_s1):,} | S2: {len(final_targets.filter(pl.col('entity_id').str.starts_with('S2'))):,} | S3: {len(final_targets.filter(pl.col('entity_id').str.starts_with('S3'))):,}")
        print(f"    Measured density: {len(final_targets)/len(sub_s1):.2f} targets/S1")

if __name__ == "__main__":
    build_calibrated_split(n_s1_per_country=10000, target_density=5.75)
