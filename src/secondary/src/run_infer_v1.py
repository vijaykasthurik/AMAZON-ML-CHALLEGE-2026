"""
Runs the production LightGBM v1 inference on any arbitrary test directory.
Used by run_pipeline.py to score the locked holdout.

Usage:
    python src/run_infer_v1.py --test-dir data/holdout_test \
                               --out-dir output/holdout_v1
"""

import sys
import os
import gc
import time
import argparse
from collections import defaultdict

import numpy as np
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(__file__))
sys.path.append("src")

# Re-use helpers from infer_lightgbm
from infer_lightgbm import (
    LEGAL_STOPS, GENERIC_ADDR, get_char_ngrams, jaccard,
    run_country_inference,
)
import infer_lightgbm as _infer_mod

MODEL_PATH = "models/lgbm_reranker.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True, help="Test data directory")
    parser.add_argument("--out-dir", default="output/holdout_v1", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    matching_out = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.out_dir, "candidate_pairs.tsv")

    # Override module-level constants at runtime
    _infer_mod.TEST_DIR = args.test_dir
    _infer_mod.MATCHING_OUT = matching_out
    _infer_mod.CANDIDATE_OUT = candidate_out
    _infer_mod.MODEL_PATH = MODEL_PATH

    print(f"=== V1 Inference: {args.test_dir} → {args.out_dir} ===", flush=True)
    booster = lgb.Booster(model_file=MODEL_PATH)

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        s1_tmp = pl.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), separator="\t")
        countries = s1_tmp["country"].unique().to_list()
        del s1_tmp
        gc.collect()

        for country in countries:
            run_country_inference(country, booster, f_match, f_cand)

    print(f"\nV1 inference complete.", flush=True)
    print(f"  {matching_out}: {os.path.getsize(matching_out):,} bytes")
    print(f"  {candidate_out}: {os.path.getsize(candidate_out):,} bytes")


if __name__ == "__main__":
    main()
