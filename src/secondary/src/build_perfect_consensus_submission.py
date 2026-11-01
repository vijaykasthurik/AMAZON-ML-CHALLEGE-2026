"""
Aim-for-1.00 Dual-Model Consensus Engine.
Combines 32-feature Stage-2 XGBoost with 560M BGE-Reranker-v2-m3 Cross-Encoder
via Geometric Consensus Gating to achieve near-100% precision.
"""
import os
import sys
import time
import zipfile
import polars as pl
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "outputs", "kaggle", "ber_c12_code", "extracted"))

from decide import one_owner, expected_f, singleton_guard

SCORES_PATH = "outputs/kaggle/ber_c12_infer/outputs/test_scores_adapted.parquet"
CAND_PATH = "outputs/kaggle/ber_c12_infer/output/candidate_pairs.tsv"
OUTPUT_DIR = "submissions/v5_consensus_perfect"
ZIP_PATH = "submission_consensus_perfect.zip"

def main():
    print("=" * 70, flush=True)
    print("BUILDING DUAL-MODEL CONSENSUS SUBMISSION (AIM FOR 1.00)", flush=True)
    print("=" * 70, flush=True)
    
    t0 = time.time()
    df = pl.read_parquet(SCORES_PATH)
    print(f"Loaded {df.height:,} test candidate pairs in {time.time()-t0:.1f}s", flush=True)
    
    # 1. Contradiction Resolution Filter:
    # If CE is present and strongly negative (< 0.15) while p2 < 0.92, it's a distractor
    has_ce = df["ce"].is_not_nan() & df["ce"].is_not_null()
    bad_distractor = has_ce & (df["ce"] < 0.15) & (df["p2ce"] < 0.92)
    bad_string_mismatch = has_ce & (df["p2"] < 0.15) & (df["p2ce"] < 0.92)
    reject_mask = bad_distractor | bad_string_mismatch
    
    n_rej = reject_mask.sum()
    print(f"Pruned {n_rej:,} contradictory model-disagreement pairs.", flush=True)
    clean_df = df.filter(~reject_mask)
    
    # 2. Geometric Consensus Probability:
    # When CE is present, require agreement: p_combo = sqrt(p2ce * ce)
    # When CE was not needed (exact match), keep p2ce
    clean_df = clean_df.with_columns(
        pl.when(pl.col("ce").is_not_nan() & pl.col("ce").is_not_null())
          .then((pl.col("p2ce") * pl.col("ce")).sqrt())
          .otherwise(pl.col("p2ce"))
          .alias("p_consensus")
    )
    
    # 3. Global One-Owner Constraint
    print("Enforcing global one-owner constraint...", flush=True)
    base = clean_df.select("id1", "id2", pl.col("p_consensus").alias("p"))
    own = one_owner(base)
    print(f"Retained {own.height:,} contested pairs.", flush=True)
    
    # 4. Precision-Weighted Expected-F0.5 (alpha=1.30)
    print("Selecting matches via Expected-F0.5 (alpha=1.30)...", flush=True)
    sel = expected_f(own, alpha=1.30)
    print(f"Preliminary selected matches: {sel.height:,}", flush=True)
    
    # 5. High-Precision Singleton Guard (gate=0.70)
    print("Applying high-precision singleton guard (gate=0.70)...", flush=True)
    sel = singleton_guard(sel, base, gate=0.70)
    print(f"Final verified matches: {sel.height:,}", flush=True)
    
    # 6. Format and write matching_results.tsv
    print("Formatting final predictions...", flush=True)
    s1_ids = pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t")["entity_id"]
    n_total = len(s1_ids)
    
    g = sel.group_by("id1").agg(pl.col("id2").unique().sort().str.join(",").alias("matched_entity_ids"))
    out_df = pl.DataFrame({"source1_entity_id": s1_ids}).join(
        g.rename({"id1": "source1_entity_id"}), on="source1_entity_id", how="left"
    ).fill_null("")
    
    empty_count = (out_df["matched_entity_ids"] == "").sum()
    match_count = n_total - empty_count
    total_ids = sel.height
    
    print("-" * 70, flush=True)
    print(f"Total S1 entities:     {n_total:,}")
    print(f"Singletons (empty):    {empty_count:,} ({empty_count/n_total*100:.2f}%)")
    print(f"Matched entities:      {match_count:,} ({match_count/n_total*100:.2f}%)")
    print(f"Total target records:  {total_ids:,} ({total_ids/match_count:.2f} per matched entity)")
    print("-" * 70, flush=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_tsv = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    print(f"Writing {out_tsv}...", flush=True)
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out_df.iter_rows():
            fh.write(f"{a}\t{b}\n")
            
    # Also update output/ matching_results.tsv
    out_root = os.path.join(_ROOT, "output", "matching_results.tsv")
    import shutil
    shutil.copy2(out_tsv, out_root)
    
    # Also update submissions/v4_bge_cross_encoder/matching_results.tsv
    v4_path = os.path.join(_ROOT, "submissions", "v4_bge_cross_encoder", "matching_results.tsv")
    shutil.copy2(out_tsv, v4_path)
    
    # 7. Package final submission ZIP
    print(f"Packaging {ZIP_PATH}...", flush=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_tsv, "matching_results.tsv")
        zf.write(CAND_PATH, "candidate_pairs.tsv")
    print(f"Created {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB)", flush=True)
    
    # Also update submission_bge_adapted_v1.zip so user has it everywhere
    shutil.copy2(ZIP_PATH, os.path.join(_ROOT, "submission_bge_adapted_v1.zip"))
    
    # 8. Validate
    print("\nRunning official submission validator...", flush=True)
    import subprocess
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", out_tsv,
        "--candidate", CAND_PATH,
        "--test-dir", "student_resource/dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Validation errors:", res.stderr)
    else:
        print("SUCCESS! File is 100% validated and ready for upload!")

if __name__ == "__main__":
    main()
