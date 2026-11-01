"""
Rebuild Calibrated Holdout Split (Density = 5.75 targets per S1).

Matches the real test set characteristics:
  - Exact held-out S1 IDs from data/splits/holdout_ids.parquet (never seen during training)
  - Target-to-S1 density of exactly 5.75 targets per S1
  - Natural singleton rate (~5.58%)
  - Seed = 2026 (fixed)
  - Saves to data/calibrated_holdout/
"""
import os
import sys
import json
import hashlib
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

SEED = 2026
RNG = np.random.default_rng(SEED)

HOLDOUT_IDS_PATH = "data/splits/holdout_ids.parquet"
OUTPUT_DIR = "data/calibrated_holdout"
MANIFEST_PATH = "data/splits/calibrated_holdout_manifest.json"

def main():
    print("=" * 70)
    print("REBUILDING CALIBRATED HOLDOUT (Density = 5.75 targets/S1)")
    print(f"Seed: {SEED} | Holdout IDs: {HOLDOUT_IDS_PATH}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    
    # 1. Load locked holdout S1 IDs
    holdout_meta = pl.read_parquet(HOLDOUT_IDS_PATH)
    holdout_s1_set = set(holdout_meta["source1_entity_id"].to_list())
    print(f"Loaded {len(holdout_s1_set):,} locked holdout S1 IDs.")
    
    # 2. Load Train Data
    print("Loading train source files...")
    s1_all = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    gt_all = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
    s2_all = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    s3_all = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    
    # Filter S1 to holdout IDs
    holdout_s1 = s1_all.filter(pl.col("entity_id").is_in(holdout_s1_set))
    holdout_gt = gt_all.filter(pl.col("source1_entity_id").is_in(holdout_s1_set))
    print(f"Filtered S1: {len(holdout_s1):,} records.")
    
    # Build GT map for holdout S1 entities
    gt_map = {}
    all_needed_targets = set()
    for r in holdout_gt.iter_rows(named=True):
        m = r["matched_entity_ids"]
        if m:
            tgts = [x.strip() for x in m.split(",") if x.strip()]
            gt_map[r["source1_entity_id"]] = tgts
            all_needed_targets.update(tgts)
        else:
            gt_map[r["source1_entity_id"]] = []
            
    print(f"Total true matched targets required: {len(all_needed_targets):,}")
    
    # 3. Process by country (US, India)
    all_final_s2 = []
    all_final_s3 = []
    
    target_density = 5.75
    manifest = {
        "seed": SEED,
        "target_density": target_density,
        "n_holdout_s1": len(holdout_s1),
        "countries": {}
    }
    
    for country in ["US", "India"]:
        print(f"\n--- Constructing Calibrated Holdout for {country} ---")
        c_s1 = holdout_s1.filter(pl.col("country") == country)
        c_s1_ids = set(c_s1["entity_id"].to_list())
        
        # True targets for this country's holdout S1 entities
        c_needed_targets = set()
        for sid in c_s1_ids:
            c_needed_targets.update(gt_map.get(sid, []))
            
        total_targets_needed = int(round(len(c_s1) * target_density))
        n_distractors_needed = max(0, total_targets_needed - len(c_needed_targets))
        
        print(f"  Holdout S1 entities : {len(c_s1):,}")
        print(f"  True target entities: {len(c_needed_targets):,}")
        print(f"  Total target pool   : {total_targets_needed:,} (Distractors needed: {n_distractors_needed:,})")
        
        # Country target candidates
        c_s2 = s2_all.filter(pl.col("country") == country)
        c_s3 = s3_all.filter(pl.col("country") == country)
        c_targets = pl.concat([c_s2, c_s3]).unique(subset=["entity_id"])
        
        # True targets df
        true_tgt_df = c_targets.filter(pl.col("entity_id").is_in(c_needed_targets))
        # Available distractors (cannot match ANY holdout S1)
        distractor_candidates = c_targets.filter(~pl.col("entity_id").is_in(c_needed_targets))
        
        # Deterministic sampling with fixed RNG
        distractor_indices = RNG.choice(
            len(distractor_candidates),
            size=min(n_distractors_needed, len(distractor_candidates)),
            replace=False
        )
        distractor_df = distractor_candidates[distractor_indices]
        
        c_final_targets = pl.concat([true_tgt_df, distractor_df]).unique(subset=["entity_id"])
        measured_density = len(c_final_targets) / len(c_s1)
        print(f"  Measured target density: {measured_density:.4f} targets/S1")
        
        c_final_s2 = c_final_targets.filter(pl.col("entity_id").str.starts_with("S2"))
        c_final_s3 = c_final_targets.filter(pl.col("entity_id").str.starts_with("S3"))
        
        all_final_s2.append(c_final_s2)
        all_final_s3.append(c_final_s3)
        
        manifest["countries"][country] = {
            "n_s1": len(c_s1),
            "n_true_targets": len(c_needed_targets),
            "n_distractors": len(distractor_df),
            "n_total_targets": len(c_final_targets),
            "density": measured_density,
            "n_s2": len(c_final_s2),
            "n_s3": len(c_final_s3)
        }
        
    final_s2_df = pl.concat(all_final_s2)
    final_s3_df = pl.concat(all_final_s3)
    
    print("\nSaving calibrated holdout files...")
    holdout_s1.write_csv(os.path.join(OUTPUT_DIR, "test_source1.tsv"), separator="\t")
    final_s2_df.write_csv(os.path.join(OUTPUT_DIR, "test_source2.tsv"), separator="\t")
    final_s3_df.write_csv(os.path.join(OUTPUT_DIR, "test_source3.tsv"), separator="\t")
    holdout_gt.write_csv(os.path.join(OUTPUT_DIR, "ground_truth.tsv"), separator="\t")
    
    # Compute sha256 checksums
    checksums = {}
    for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv", "ground_truth.tsv"]:
        p = os.path.join(OUTPUT_DIR, fname)
        with open(p, "rb") as f:
            checksums[fname] = hashlib.sha256(f.read()).hexdigest()
    manifest["checksums"] = checksums
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nCalibrated holdout successfully built and locked!")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(f"Files saved to: {OUTPUT_DIR}")
    print(f"  S1: {len(holdout_s1):,} rows")
    print(f"  S2: {len(final_s2_df):,} rows")
    print(f"  S3: {len(final_s3_df):,} rows")
    print(f"  GT: {len(holdout_gt):,} rows")

if __name__ == "__main__":
    main()
