"""
Ultra-High Precision Decision Layer for Business Entity Resolution.
Target: Macro F0.5 >= 0.999999

Pipeline:
  1. Load test_scores_adapted.parquet (8.79M candidate pairs)
  2. Cross-Encoder Consistency Verification:
     Reject pairs where Cross-Encoder score is strongly negative (ce < 0.15),
     eliminating false positive distractors.
  3. Strict Global One-Owner Constraint (Each target S2/S3 belongs to at most 1 S1)
  4. Precision-Weighted Expected-F0.5 Subset Selection
  5. Calibrated Singleton Guard Gate (Protects 5.6% singleton entities from false merges)
  6. Writes matching_results.tsv and candidate_pairs.tsv
  7. Validates format against official challenge validator.
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
OUTPUT_DIR = "submissions/v4_bge_cross_encoder"
ZIP_PATH = "submission_bge_adapted_v1.zip"

def main(alpha: float = 1.25, gate: float = 0.65, ce_min: float = 0.10):
    print("=" * 70, flush=True)
    print("ULTRA-HIGH PRECISION DECISION LAYER OPTIMIZATION", flush=True)
    print(f"Parameters: alpha={alpha:.2f} | gate={gate:.2f} | ce_min={ce_min:.2f}", flush=True)
    print("=" * 70, flush=True)
    
    t0 = time.time()
    print("Loading test scores...", flush=True)
    df = pl.read_parquet(SCORES_PATH)
    print(f"Loaded {df.height:,} candidate pairs in {time.time()-t0:.1f}s", flush=True)
    
    # 1. Cross-Encoder Consistency Filter
    # Where ce is not null/nan, enforce ce >= ce_min to prevent false positive merges
    print(f"Applying cross-encoder consistency filter (ce >= {ce_min})...", flush=True)
    has_ce = df["ce"].is_not_nan() & df["ce"].is_not_null()
    bad_ce = has_ce & (df["ce"] < ce_min) & (df["p2ce"] < 0.90)
    n_rejected = bad_ce.sum()
    print(f"  Pruned {n_rejected:,} contradictory false-positive candidate pairs.", flush=True)
    df_clean = df.filter(~bad_ce)
    
    # 2. Strict One-Owner Constraint
    print("Enforcing strict global one-owner constraint...", flush=True)
    base = df_clean.select("id1", "id2", pl.col("p2ce").alias("p"))
    own = one_owner(base)
    print(f"  Retained {own.height:,} owner-contested candidate pairs.", flush=True)
    
    # 3. Expected F0.5 Selection
    print(f"Selecting matches via Expected-F0.5 (alpha={alpha:.2f})...", flush=True)
    sel = expected_f(own, alpha=alpha)
    print(f"  Preliminary selected matches: {sel.height:,}", flush=True)
    
    # 4. Calibrated Singleton Guard
    print(f"Applying singleton guard gate (gate={gate:.2f})...", flush=True)
    sel = singleton_guard(sel, base, gate=gate)
    print(f"  Final verified matches: {sel.height:,}", flush=True)
    
    # 5. Format Submission
    print("Formatting predictions into matching_results.tsv...", flush=True)
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
    print(f"Writing {out_tsv} line by line...", flush=True)
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out_df.iter_rows():
            fh.write(f"{a}\t{b}\n")
    
    # Also update output/ matching_results.tsv
    out_root = os.path.join(_ROOT, "output", "matching_results.tsv")
    import shutil
    shutil.copy2(out_tsv, out_root)
    
    # 6. Package final submission ZIP
    print(f"Packaging {ZIP_PATH}...", flush=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_tsv, "matching_results.tsv")
        zf.write(CAND_PATH, "candidate_pairs.tsv")
    print(f"Created {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB)", flush=True)
    
    # 7. Validate
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
    main(alpha=1.25, gate=0.65, ce_min=0.10)
