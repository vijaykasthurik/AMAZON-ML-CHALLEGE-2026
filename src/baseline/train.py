"""Stage 1: pairwise LightGBM with out-of-fold predictions on every train pair.

Validation design (mirrors how the test set is scored):
  * 4% of the training Source 1 entities (by a hash of their id) form the
    validation entity set V. Every query with a candidate in V, together with
    *all* of its candidates, is a validation query; validation queries are
    never used to fit any model.
  * The remaining queries are split in two folds (hash of the query id). Model
    A is fit on a sample of fold A and predicts fold B, model B the reverse, so
    every train pair - validation pairs included - gets an out-of-fold
    probability p1. These p1 values feed the stage-2 model (stage2.py).
  * Macro F0.5 is computed on V with the exact metric (decode.macro_f05).

Outputs: models/lgb_s1_A.txt, models/lgb_s1_B.txt, models/stage1.json,
cache/p1_train.parquet (q, c, y, p1 for all train pairs + fold / val flags).

Usage: python train.py [train_query_fraction]
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
from candidates import RANKS, true_pairs  # noqa: E402
from decode import expected_f_decode, macro_f05, threshold_decode  # noqa: E402
from features import (add_scores, c_context, frequency_table, pair_features,  # noqa: E402
                      q_context, query_chunks, record_table, set_token_freq)
from preprocess import norm_path  # noqa: E402

K_MODEL = 2          # a pair enters the model if it is in the top-2 of any retrieval list
CHUNK = 1_000_000    # pairs per feature chunk
VAL_MOD = 25         # 1 / VAL_MOD of the S1 entities are validation entities
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=config.N_JOBS, seed=config.SEED, verbose=-1)
# retrieval columns carried into the stage-2 table
S2_PASS = ["sn", "sa", "kn", "ka", "comb", "kcomb", "rk_c", "rk_kc"]
# strongest stage-1 pair features, also handed to stage 2 so it can weigh the
# graph context against direct pair evidence
S1_PASS = ["wa_num_q_min", "wa_num_c_min", "wa_n_q_min", "wa_n_c_min", "wa_a_q_unm",
           "n_full", "a_tset", "legal_diff", "q_addr_empty", "f_c_name", "hn_sd", "num_sd"]
LEAN_COLS = ["entity_id", "business_name", "country", "nn", "nc", "nk", "nl", "na", "ac", "anum"]


def load_lean(split: str):
    """Normalised S1 and S2+S3 tables with only the columns the model needs."""
    s1 = pl.read_parquet(norm_path(split, 1), columns=LEAN_COLS)
    other = pl.concat([pl.read_parquet(norm_path(split, s), columns=LEAN_COLS) for s in (2, 3)])
    return s1, other


def prepare(split: str):
    """Load a split: model candidates (with S1-side context) and record tables."""
    s1, other = load_lean(split)
    set_token_freq(s1["nc"], s1["country"])  # word frequencies of this split's S1 names (label free)
    cands = pl.read_parquet(os.path.join(config.CACHE_DIR, f"cands_{split}.parquet"))
    cands = cands.filter(pl.any_horizontal([pl.col(r) < K_MODEL for r in RANKS]))
    cands, c_stats = c_context(add_scores(cands))
    cands = cands.sort("q")
    rec_c, rec_q = record_table(s1), record_table(other)
    freq_c = frequency_table(s1)
    q_is_s3 = other["entity_id"].str.starts_with("S3-").to_numpy()
    s1 = s1.select("entity_id", "country")
    other = other.select("entity_id", "country")
    return s1, other, cands, c_stats, rec_q, rec_c, freq_c, q_is_s3


def features_of(part: pl.DataFrame, ctx) -> tuple:
    """Feature matrix (float32) and column names for one query-complete chunk."""
    c_stats, rec_q, rec_c, freq_c, q_is_s3 = ctx
    part2 = q_context(part, c_stats)
    # rows must stay aligned with ``part`` (labels / folds are read from it)
    same = ((part2["q"].to_numpy() == part["q"].to_numpy()).all()
            and (part2["c"].to_numpy() == part["c"].to_numpy()).all())
    assert same, "row order changed"
    f = pair_features(part2, rec_q, rec_c, freq_c, q_is_s3)
    return f.to_numpy().astype(np.float32), f.columns


def build_features(pairs: pl.DataFrame, ctx):
    """Features for a (q-sorted, query-complete) pair table, chunk by chunk."""
    X, names = [], None
    for part in query_chunks(pairs, CHUNK):
        x, names = features_of(part, ctx)
        X.append(x)
    return np.concatenate(X), names


def fold_of(q: pl.Expr) -> pl.Expr:
    """Fold (0 = A, 1 = B) of a query row number."""
    return (q.hash(seed=11) % 2).cast(pl.Int8)


def evaluate(vp: pl.DataFrame, vtruth: pl.DataFrame, val_c: np.ndarray) -> dict:
    """Macro F0.5 of several decoders on the validation entities (column ``p``)."""
    res = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        res[f"thr_{t}"] = macro_f05(threshold_decode(vp, t), vtruth, val_c)
    for miss in (0.0, 0.1, 0.2, 0.3, 0.5):
        res[f"ef_{miss}"] = macro_f05(expected_f_decode(vp, miss=miss), vtruth, val_c)
    return res


def main(frac: float = 0.3, reuse: bool = False):
    """Fit the two stage-1 models and write out-of-fold p1 for all train pairs.

    With ``reuse`` the fold models of the previous run are loaded instead of
    refitted (used when only the saved pass-through columns change).
    """
    t0 = time.time()
    s1, other, cands, c_stats, rec_q, rec_c, freq_c, q_is_s3 = prepare("train")
    ctx = (c_stats, rec_q, rec_c, freq_c, q_is_s3)
    truth = true_pairs(s1, other)
    is_val = (s1["entity_id"].hash(seed=7) % VAL_MOD == 0).to_numpy()
    val_c = np.flatnonzero(is_val).astype(np.int32)
    val_q = cands.filter(pl.col("c").is_in(pl.Series(val_c).implode()))["q"].unique()
    cands = (cands.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q", "c"], how="left")
             .with_columns(pl.col("y").fill_null(0), fold_of(pl.col("q")).alias("fold"),
                           pl.col("q").is_in(val_q.implode()).alias("is_val"))
             .sort("q"))
    print(f"loaded {cands.height} pairs, positives {cands['y'].sum()} / {truth.height}, "
          f"val entities {len(val_c)}, val queries {len(val_q)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- fit one model per fold on a sample of its non-validation queries
    models = {}
    names = None
    if reuse:  # load the fold models saved by a previous run (same features)
        with open(os.path.join(config.MODEL_DIR, "stage1.json")) as f:
            names = json.load(f)["features"]
        for fold in (0, 1):
            models[fold] = lgb.Booster(model_file=os.path.join(config.MODEL_DIR, f"lgb_s1_{'AB'[fold]}.txt"))
        print("reusing saved stage-1 fold models", flush=True)
    for fold in (() if reuse else (0, 1)):
        pool_q = cands.filter((pl.col("fold") == fold) & ~pl.col("is_val"))["q"].unique()
        fit_q = pool_q.sample(fraction=frac, seed=config.SEED + fold)
        es_q = (cands.filter((pl.col("fold") == 1 - fold) & ~pl.col("is_val"))["q"].unique()
                .sample(fraction=0.03, seed=config.SEED))
        fit = cands.filter(pl.col("q").is_in(fit_q.implode()))
        es = cands.filter(pl.col("q").is_in(es_q.implode()))
        Xf, names = build_features(fit, ctx)
        Xe, _ = build_features(es, ctx)
        print(f"fold {'AB'[fold]}: fit {Xf.shape}, early-stop {Xe.shape} ({time.time() - t0:.0f}s)",
              flush=True)
        dtr = lgb.Dataset(Xf, fit["y"].to_numpy(), feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(Xe, es["y"].to_numpy(), reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        m.save_model(os.path.join(config.MODEL_DIR, f"lgb_s1_{'AB'[fold]}.txt"))
        models[fold] = m
        del Xf, Xe, dtr, dva, fit, es
        print(f"  best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)

    # ---- out-of-fold p1 for every train pair (+ pass-through features for stage 2)
    p1 = np.empty(cands.height, np.float32)
    pass_idx = [names.index(f) for f in S1_PASS]
    passed = np.empty((cands.height, len(S1_PASS)), np.float32)
    pos = 0
    for part in query_chunks(cands, CHUNK):
        x, _ = features_of(part, ctx)
        passed[pos:pos + part.height] = x[:, pass_idx]
        fold = part["fold"].to_numpy()
        pr = np.empty(len(fold), np.float32)
        for f in (0, 1):
            m = fold == f
            if m.any():
                pr[m] = models[1 - f].predict(x[m], num_threads=config.N_JOBS)
        p1[pos:pos + part.height] = pr
        pos += part.height
    out = cands.select("q", "c", "y", "fold", "is_val", *S2_PASS).with_columns(
        pl.Series("p1", p1), pl.Series("q_is_s3", q_is_s3[cands["q"].to_numpy()]),
        *[pl.Series(f, passed[:, k]) for k, f in enumerate(S1_PASS)])
    del passed
    out.write_parquet(os.path.join(config.CACHE_DIR, "p1_train.parquet"))
    print(f"OOF p1 written for {out.height} pairs ({time.time() - t0:.0f}s)", flush=True)

    vtruth = truth.filter(pl.col("c").is_in(pl.Series(val_c).implode()))
    vp = out.filter(pl.col("is_val")).select("q", "c", "y", pl.col("p1").alias("p"))
    res = evaluate(vp, vtruth, val_c)
    for k, v in res.items():
        print(f"  stage1 {k}: macro F0.5 = {v:.5f}")
    best = max(res, key=res.get)
    ceiling = macro_f05(vp.filter(pl.col("y") == 1).select("q", "c"), vtruth, val_c)
    print(f"stage1 best decoder {best}: {res[best]:.5f}   (oracle on candidates: {ceiling:.5f})")
    imp = sorted(zip(names, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, round(g / 1e6, 1)) for f, g in imp[:25]])
    vp.write_parquet(os.path.join(config.CACHE_DIR, "val_pred.parquet"))
    np.save(os.path.join(config.CACHE_DIR, "val_entities.npy"), val_c)
    with open(os.path.join(config.MODEL_DIR, "stage1.json"), "w") as f:
        json.dump({"best": best, "scores": res, "features": names, "k_model": K_MODEL,
                   "ceiling": ceiling}, f, indent=1)
    with open(os.path.join(config.MODEL_DIR, "decoder.json"), "w") as f:
        json.dump({"stage": 1, "best": best, "scores": res}, f, indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(float(args[0]) if args else 0.3, reuse="--reuse" in sys.argv)
