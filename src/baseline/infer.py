"""Score the test candidate pairs and write the two submission files.

Steps
  1. load the normalised test sources and the test candidate table
     (cache/cands_test.parquet from candidates.py), keeping the same top-K_MODEL
     pairs the models were trained on;
  2. stage 1: pair features chunk by chunk, p1 = mean of the two fold models;
  3. stage 2 (if it won on validation): graph-context features from p1 over
     the whole test candidate graph, p2 from the stage-2 model;
  4. decode with the decoder chosen on validation and write
     output/matching_results.tsv and output/candidate_pairs.tsv. The candidate
     file is exactly the pair set scored in step 2.

Usage: python infer.py [decoder] [stage]
       (defaults: the choice stored in models/decoder.json)
"""

from __future__ import annotations

import _env  # noqa: F401  (must come before numpy)

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from decode import expected_f_decode, threshold_decode, write_outputs  # noqa: E402
from features import query_chunks  # noqa: E402
from stage2 import country_codes, s2_predict  # noqa: E402
from train import CHUNK, S1_PASS, S2_PASS, features_of, prepare  # noqa: E402


def decode(pairs: pl.DataFrame, name: str) -> pl.DataFrame:
    """Apply a decoder given by its name ("thr_<t>" or "ef_<miss>") to column p."""
    kind, val = name.split("_")
    if kind == "thr":
        return threshold_decode(pairs, float(val))
    return expected_f_decode(pairs, miss=float(val))


def main(decoder: str | None = None, stage: int | None = None, reuse_p1: bool = False):
    """Predict every test pair and write the submission files.

    ``reuse_p1`` loads the stage-1 checkpoint (cache/test_p1.parquet) written by
    a previous run with the same stage-1 models instead of recomputing it.
    """
    t0 = time.time()
    with open(os.path.join(config.MODEL_DIR, "decoder.json")) as f:
        meta = json.load(f)
    with open(os.path.join(config.MODEL_DIR, "stage1.json")) as f:
        feats1 = json.load(f)["features"]
    stage = stage or meta["stage"]
    decoder = decoder or meta["best"]
    m1 = [lgb.Booster(model_file=os.path.join(config.MODEL_DIR, f"lgb_s1_{k}.txt")) for k in "AB"]
    s1, other, cands, c_stats, rec_q, rec_c, freq_c, q_is_s3 = prepare("test")
    ctx = (c_stats, rec_q, rec_c, freq_c, q_is_s3)
    print(f"test: {s1.height} S1, {other.height} S2+S3, {cands.height} pairs "
          f"({time.time() - t0:.0f}s)", flush=True)

    ckpt = os.path.join(config.CACHE_DIR, "test_p1.parquet")
    if reuse_p1 and os.path.exists(ckpt):
        out = pl.read_parquet(ckpt)
        assert out.height == cands.height, "checkpoint does not match the candidate table"
        print(f"reusing stage-1 checkpoint {ckpt}", flush=True)
    else:
        p1 = np.empty(cands.height, np.float32)
        pass_idx = [feats1.index(f) for f in S1_PASS]
        passed = np.empty((cands.height, len(S1_PASS)), np.float32)
        pos = 0
        for part in query_chunks(cands, CHUNK):
            x, names = features_of(part, ctx)
            assert names == feats1, "feature mismatch with the trained model"
            passed[pos:pos + part.height] = x[:, pass_idx]
            p1[pos:pos + part.height] = np.mean(
                [m.predict(x, num_threads=config.N_JOBS) for m in m1], axis=0)
            pos += part.height
            print(f"  stage 1 scored {pos}/{cands.height} ({time.time() - t0:.0f}s)", flush=True)
        out = cands.select("q", "c", *S2_PASS).with_columns(
            pl.Series("p1", p1), pl.Series("q_is_s3", q_is_s3[cands["q"].to_numpy()]),
            *[pl.Series(f, passed[:, k]) for k, f in enumerate(S1_PASS)])
        del passed
        out.write_parquet(ckpt)  # checkpoint: a stage-2 failure does not cost stage 1 again

    # stage-1 inputs are no longer needed; free them before the graph features
    del ctx, rec_q, rec_c, freq_c, c_stats
    cands = cands.select("q", "c")
    if stage == 2:
        m2 = lgb.Booster(model_file=os.path.join(config.MODEL_DIR, "lgb_s2.txt"))
        p2 = s2_predict(out, country_codes("test", out["q"].to_numpy()), m2, meta["features2"])
        out = out.with_columns(pl.Series("p", p2))
    else:
        out = out.with_columns(pl.col("p1").alias("p"))
    out.select("q", "c", "p1", "p").write_parquet(os.path.join(config.CACHE_DIR, "test_pred.parquet"))

    pred = decode(out.select("q", "c", "p"), decoder)
    write_outputs(None, pred, cands.select("q", "c"), s1, other,
                  os.path.join(config.OUTPUT_DIR, "matching_results.tsv"),
                  os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"))
    print(f"stage {stage}, decoder {decoder}: {pred.height} matched pairs for "
          f"{pred['c'].n_unique()} of {s1.height} S1 entities ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if len(args) > 0 else None, int(args[1]) if len(args) > 1 else None,
         reuse_p1="--reuse-p1" in sys.argv)
