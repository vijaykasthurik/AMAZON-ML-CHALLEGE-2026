"""
Master orchestration script for the full ML Challenge pipeline.
Executes all steps sequentially after India inference (task-1234) completes.

Step 1: Save v1 submission + run check_submission.py
Step 2: Install CUDA PyTorch from cached wheel (in venv_gpu)
Step 3: Create locked holdout
Step 4: Run loanword mining (RAM-safe, streaming)
Step 5: Run v1 holdout scoring (use #1)
Step 6: Train LightGBM v2 (enhanced features)
Step 7: Run v2 inference (test set)
Step 8: Save v2 submission
Step 9: Run v2 holdout scoring (use #2)
Step 10: Decision gate — if ≥ 0.98999, package and deliver; else continue

Usage:
    python run_pipeline.py

This script is idempotent: each step is guarded by a checkpoint file.
If it crashes, re-running resumes from the last completed step.
"""

import os
import sys
import gc
import json
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CHECKPOINT_DIR = "data/pipeline_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("submissions/v1_lgbm12", exist_ok=True)
os.makedirs("submissions/final", exist_ok=True)
os.makedirs("submissions/v2_lgbm_enhanced", exist_ok=True)


def checkpoint_done(name: str) -> bool:
    return os.path.exists(os.path.join(CHECKPOINT_DIR, f"{name}.done"))


def mark_done(name: str, info: dict = None):
    p = os.path.join(CHECKPOINT_DIR, f"{name}.done")
    with open(p, "w") as f:
        json.dump({"ts": datetime.now().isoformat(), **(info or {})}, f)
    print(f"  ✓ Checkpoint: {name}")


def run(cmd, desc="", check=True, cwd=None):
    """Run a subprocess command, print output live, raise on failure."""
    print(f"\n--- {desc or ' '.join(cmd[:3])} ---", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd or ".", capture_output=False, text=True, check=check)
    print(f"  Done in {time.time()-t0:.0f}s (exit {result.returncode})", flush=True)
    return result


def check_ram_gb() -> float:
    """Return free RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return 4.0  # conservative default


def wait_for_free_ram(min_gb=2.0, timeout=600):
    """Wait until at least min_gb free RAM is available."""
    deadline = time.time() + timeout
    while check_ram_gb() < min_gb:
        if time.time() > deadline:
            print(f"WARNING: RAM not freed after {timeout}s, proceeding anyway.")
            break
        print(f"  Waiting for RAM (free={check_ram_gb():.1f}GB, need {min_gb}GB)...")
        time.sleep(30)


# ────────────────────────────────────────────────────────────────────────────────
# STEP 1: Save v1 submission and validate
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step1_save_v1"):
    print("\n" + "="*70)
    print("STEP 1: Save v1 submission + check_submission.py")
    print("="*70)
    # Verify output files exist
    assert os.path.exists("output/matching_results.tsv"), \
        "ERROR: output/matching_results.tsv not found. Is task-1234 complete?"
    assert os.path.exists("output/candidate_pairs.tsv"), \
        "ERROR: output/candidate_pairs.tsv not found."

    # Copy to v1 directory
    shutil.copy("output/matching_results.tsv", "submissions/v1_lgbm12/matching_results.tsv")
    shutil.copy("output/candidate_pairs.tsv", "submissions/v1_lgbm12/candidate_pairs.tsv")
    print("Copied output/ → submissions/v1_lgbm12/")

    # Run official check
    result = subprocess.run(
        [sys.executable, "src/check_submission.py",
         "--matching", "submissions/v1_lgbm12/matching_results.tsv",
         "--candidate", "submissions/v1_lgbm12/candidate_pairs.tsv"],
        capture_output=False, text=True
    )
    if result.returncode != 0:
        print("WARNING: check_submission.py returned non-zero for v1!")
    mark_done("step1_save_v1", {"check_rc": result.returncode})
else:
    print("STEP 1: Already done (v1 saved).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 2: Install CUDA PyTorch from cached wheel
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step2_cuda_torch"):
    print("\n" + "="*70)
    print("STEP 2: Install CUDA PyTorch in venv_gpu (from pip cache)")
    print("="*70)
    wait_for_free_ram(min_gb=6.0)  # wheel extraction needs ~5 GB
    pip_exe = "venv_gpu\\Scripts\\pip.exe"
    result = subprocess.run(
        [pip_exe, "install", "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu124", "--no-deps"],
        capture_output=False, text=True
    )
    # Verify
    py_exe = "venv_gpu\\Scripts\\python.exe"
    verify = subprocess.run(
        [py_exe, "-c",
         "import torch; print('cuda:', torch.cuda.is_available()); "
         "x=torch.ones(1000,1000,dtype=torch.float16).cuda(); "
         "y=torch.ones(1000,1000,dtype=torch.float16).cuda(); "
         "z=x@y; print('fp16 matmul OK, peak VRAM MB:', "
         "torch.cuda.max_memory_allocated()//1e6)"],
        capture_output=True, text=True
    )
    print(verify.stdout)
    if verify.returncode == 0 and "cuda: True" in verify.stdout:
        print("CUDA PyTorch available. GPU enabled.")
        mark_done("step2_cuda_torch", {"cuda_available": True})
    else:
        print("WARNING: CUDA not available. GPU steps will be skipped (CPU fallback).")
        mark_done("step2_cuda_torch", {"cuda_available": False, "err": verify.stderr[:200]})
else:
    print("STEP 2: Already done (CUDA PyTorch installed).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 3: Create locked holdout
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step3_holdout"):
    print("\n" + "="*70)
    print("STEP 3: Create locked holdout split")
    print("="*70)
    wait_for_free_ram(min_gb=2.0)
    run([sys.executable, "src/holdout.py"], desc="Create 10% locked holdout")
    mark_done("step3_holdout")
else:
    print("STEP 3: Already done (holdout locked).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 4: Data-driven loanword mining
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step4_loanwords"):
    print("\n" + "="*70)
    print("STEP 4: Data-driven Indic loanword mining")
    print("="*70)
    wait_for_free_ram(min_gb=3.0)
    run([sys.executable, "mine_indic_loanwords.py"], desc="Mine Indic loanword mappings")
    mark_done("step4_loanwords")
else:
    print("STEP 4: Already done (loanword map ready).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 5: Score v1 on locked holdout (use #1)
# ────────────────────────────────────────────────────────────────────────────────
# To score v1 on holdout, we need predictions for holdout entities.
# We generate these by running v1 inference on TRAIN data with holdout IDs as the "test".
if not checkpoint_done("step5_v1_holdout"):
    print("\n" + "="*70)
    print("STEP 5: Run v1 inference on holdout set and score (use #1 of 3)")
    print("="*70)
    wait_for_free_ram(min_gb=3.0)

    # Create a temp test-format directory from holdout train entities
    import polars as pl
    holdout_ids_df = pl.read_parquet("data/splits/holdout_ids.parquet")
    holdout_ids = set(holdout_ids_df["source1_entity_id"].to_list())

    # Write holdout entities to a temp test directory in the same TSV format
    HOLDOUT_TEST_DIR = "data/holdout_test"
    os.makedirs(HOLDOUT_TEST_DIR, exist_ok=True)

    # Source 1: holdout S1 entities
    s1_full = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    s1_holdout = s1_full.filter(pl.col("entity_id").is_in(holdout_ids))
    s1_holdout.write_csv(os.path.join(HOLDOUT_TEST_DIR, "test_source1.tsv"), separator="\t")
    countries_in_holdout = set(s1_holdout["country"].to_list())
    del s1_full
    gc.collect()

    # Source 2+3: all targets for those countries
    for src_num in [2, 3]:
        src_path = f"student_resource/dataset/train/train_source{src_num}.tsv"
        tgt_df = pl.scan_csv(src_path, separator="\t").filter(
            pl.col("country").is_in(countries_in_holdout)
        ).collect()
        tgt_df.write_csv(
            os.path.join(HOLDOUT_TEST_DIR, f"test_source{src_num}.tsv"), separator="\t"
        )
        del tgt_df
        gc.collect()

    print(f"Holdout test dir written: {HOLDOUT_TEST_DIR} "
          f"({len(s1_holdout)} S1 entities, countries: {countries_in_holdout})", flush=True)
    del holdout_ids_df, s1_holdout
    gc.collect()

    # Run v1 inference on holdout test dir
    os.makedirs("output/holdout_v1", exist_ok=True)
    run([sys.executable, "-u", "src/run_infer_v1.py",
         "--test-dir", HOLDOUT_TEST_DIR,
         "--out-dir", "output/holdout_v1"],
        desc="v1 inference on holdout")

    # Score
    run([sys.executable, "src/score_holdout.py",
         "--pred", "output/holdout_v1/matching_results.tsv",
         "--version", "v1"],
        desc="Score v1 on holdout")

    mark_done("step5_v1_holdout")
else:
    print("STEP 5: Already done (v1 holdout scored).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 6: Train LightGBM v2
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step6_train_v2"):
    print("\n" + "="*70)
    print("STEP 6: Train LightGBM v2 (enhanced features, dev set only)")
    print("="*70)
    wait_for_free_ram(min_gb=3.0)
    run([sys.executable, "train_lgbm_v2.py"], desc="Train LightGBM v2")
    mark_done("step6_train_v2")
else:
    print("STEP 6: Already done (model/lgbm_v2.txt ready).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 7: v2 inference on TEST set
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step7_infer_v2"):
    print("\n" + "="*70)
    print("STEP 7: LightGBM v2 inference on test set")
    print("="*70)
    wait_for_free_ram(min_gb=3.0)
    os.makedirs("output/v2", exist_ok=True)
    run([sys.executable, "-u", "src/infer_lgbm_v2.py",
         "--out-dir", "output/v2",
         "--model", "models/lgbm_v2.txt"],
        desc="v2 inference on test set")
    mark_done("step7_infer_v2")
else:
    print("STEP 7: Already done (output/v2/ ready).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 8: Save v2 submission
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step8_save_v2"):
    print("\n" + "="*70)
    print("STEP 8: Save v2 submission")
    print("="*70)
    shutil.copy("output/v2/matching_results.tsv",
                "submissions/v2_lgbm_enhanced/matching_results.tsv")
    shutil.copy("output/v2/candidate_pairs.tsv",
                "submissions/v2_lgbm_enhanced/candidate_pairs.tsv")
    print("Copied output/v2/ → submissions/v2_lgbm_enhanced/")
    mark_done("step8_save_v2")
else:
    print("STEP 8: Already done (v2 submission saved).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 9: Score v2 on holdout (use #2)
# ────────────────────────────────────────────────────────────────────────────────
if not checkpoint_done("step9_v2_holdout"):
    print("\n" + "="*70)
    print("STEP 9: Run v2 inference on holdout set and score (use #2 of 3)")
    print("="*70)
    wait_for_free_ram(min_gb=3.0)

    # Run v2 inference on holdout dir
    os.makedirs("output/holdout_v2", exist_ok=True)
    run([sys.executable, "-u", "src/infer_lgbm_v2.py",
         "--out-dir", "output/holdout_v2",
         "--data-dir", "data/holdout_test",
         "--model", "models/lgbm_v2.txt"],
        desc="v2 inference on holdout")

    # Score
    v2_result = run([sys.executable, "src/score_holdout.py",
                     "--pred", "output/holdout_v2/matching_results.tsv",
                     "--version", "v2_best"],
                    desc="Score v2 on holdout")

    mark_done("step9_v2_holdout")
else:
    print("STEP 9: Already done (v2 holdout scored).")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 10: Decision gate
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 10: Decision gate")
print("="*70)

if os.path.exists("data/splits/holdout_use_log.jsonl"):
    uses = []
    with open("data/splits/holdout_use_log.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                uses.append(json.loads(line))
    print("Holdout use log:")
    for u in uses:
        print(f"  [{u['version']}] {u['timestamp']} weighted_f05={u['weighted_f05']:.5f} "
              f"{'PASS ✓' if u['passed_gate'] else 'FAIL'}")

    best = max(uses, key=lambda x: x["weighted_f05"])
    print(f"\nBest holdout score: {best['weighted_f05']:.5f} (version={best['version']})")

    if best["passed_gate"]:
        # Determine which files to copy to submissions/final/
        if best["version"] == "v2_best":
            src_match = "output/v2/matching_results.tsv"
            src_cand = "output/v2/candidate_pairs.tsv"
        else:
            src_match = "submissions/v1_lgbm12/matching_results.tsv"
            src_cand = "submissions/v1_lgbm12/candidate_pairs.tsv"
        shutil.copy(src_match, "submissions/final/matching_results.tsv")
        shutil.copy(src_cand, "submissions/final/candidate_pairs.tsv")
        print(f"\n✓ GATE PASSED — submissions/final/ populated from version={best['version']}")
        print(f"  Weighted holdout F0.5 = {best['weighted_f05']:.5f} ≥ 0.98999")
    else:
        print(f"\n✗ GATE NOT YET PASSED (best={best['weighted_f05']:.5f} < 0.98999)")
        print("  Continuing improvement loop (see Section 3)...")
else:
    print("No holdout scores recorded yet.")

print("\n=== PIPELINE COMPLETE ===")
