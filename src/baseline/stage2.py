"""Stage 2: re-score every pair using the stage-1 probabilities of its neighbours.

A pair is not judged in isolation. Useful evidence sits in the rest of the
candidate graph:
  * query side  - is this the query's best S1 candidate, and by how much does
    it beat the runner-up? (a query belongs to at most one S1 entity)
  * entity side - does the S1 entity already have other confident matches, how
    many, from which source, and where does this query rank among them?
    (entities have 0-11 matches; one with no other support is more likely a
    singleton, one with many confident matches can afford a weaker one)

These features are computed from the out-of-fold stage-1 probabilities p1, so
the stage-2 model never sees a probability produced by a model that was fit on
the same query. Stage 2 is fit on non-validation queries and scored on the
validation entities exactly like stage 1.

Outputs: models/lgb_s2.txt, models/decoder.json (updated), cache/val_pred2.parquet

Usage: python stage2.py
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
from candidates import load_split, true_pairs  # noqa: E402
from decode import macro_f05  # noqa: E402
from train import S1_PASS, evaluate  # noqa: E402

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=config.N_JOBS, seed=config.SEED, verbose=-1)


def s2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Graph-context features from p1 (df needs q, c, p1, q_is_s3 and retrieval scores)."""
    conf = (pl.col("p1") > 0.5).cast(pl.Float32)
    df = df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("q_p1_rank"),
        pl.col("p1").rank("ordinal", descending=True).over("c").cast(pl.Float32).alias("c_p1_rank"),
        pl.col("p1").sum().over("q").alias("q_p1_sum"),
        pl.col("p1").sum().over("c").alias("c_p1_sum"),
        conf.sum().over("c").alias("c_nconf"),
        (conf * pl.col("q_is_s3")).sum().over("c").alias("c_nconf_s3"),
        pl.len().over("c").cast(pl.Float32).alias("c_n"),
    )
    # best / second-best p1 of each query, and whether each query picks this entity
    df = df.with_columns(
        pl.col("p1").max().over("q").alias("q_p1_max"),
        pl.when(pl.col("q_p1_rank") == 2).then(pl.col("p1")).otherwise(None)
        .max().over("q").fill_null(0.0).alias("q_p1_2nd"),
    ).with_columns(
        (pl.col("q_p1_rank") == 1).cast(pl.Float32).alias("is_q_best"),
    ).with_columns(
        # how many other queries choose this entity as their best, confidently
        ((pl.col("is_q_best") * conf).sum().over("c") - pl.col("is_q_best") * conf).alias("c_sib"),
        ((pl.col("is_q_best") * pl.col("p1")).sum().over("c")
         - pl.col("is_q_best") * pl.col("p1")).alias("c_sib_p"),
        ((pl.col("is_q_best") * pl.col("p1")).max().over("c")).alias("c_best_p"),
    )
    own_s3 = pl.col("q_is_s3").cast(pl.Float32)
    return df.select(
        "p1",
        (pl.col("p1") - pl.when(pl.col("q_p1_rank") == 1).then(pl.col("q_p1_2nd"))
         .otherwise(pl.col("q_p1_max"))).alias("q_margin_p1"),
        "q_p1_rank", "q_p1_sum", "q_p1_max", "is_q_best",
        "c_p1_rank", (pl.col("c_p1_sum") - pl.col("p1")).alias("c_p1_other"),
        (pl.col("c_nconf") - conf).alias("c_nconf_other"),
        (pl.col("c_nconf_s3") - conf * own_s3).alias("c_nconf_s3_other"),
        ((pl.col("c_nconf") - pl.col("c_nconf_s3")) - conf * (1 - own_s3)).alias("c_nconf_s2_other"),
        "c_n", "c_sib", "c_sib_p", "c_best_p",
        "q_is_s3", "sn", "sa", "kn", "ka", "comb", "kcomb", "rk_c", "rk_kc",
        *[f for f in S1_PASS if f in df.columns],
    ).cast(pl.Float32)


def country_codes(split: str, q: np.ndarray) -> np.ndarray:
    """Small integer country code of each query row (q and its candidates share it)."""
    _, other = load_split(split, columns=["country"])
    _, inv = np.unique(other["country"].to_numpy().astype(str), return_inverse=True)
    return inv[q].astype(np.int8)


def s2_by_country(df: pl.DataFrame, code: np.ndarray, keep: np.ndarray | None = None):
    """Stage-2 features computed country by country (exact: windows never cross countries).

    Returns (row positions in ``df``, float32 feature matrix, feature names) for
    the rows where ``keep`` is True (all rows when keep is None).
    """
    rows_out, X_out, names = [], [], None
    pos = np.arange(df.height)
    for g in np.unique(code):
        m = code == g
        sub = df.filter(pl.Series(m))
        X = s2_features(sub)
        names = X.columns
        r = pos[m]
        if keep is not None:
            k = keep[m]
            X = X.filter(pl.Series(k))
            r = r[k]
        rows_out.append(r)
        X_out.append(X.to_numpy())
        del sub, X
    return np.concatenate(rows_out), np.concatenate(X_out), names


def s2_predict(df: pl.DataFrame, code: np.ndarray, model, names_expected: list) -> np.ndarray:
    """Stage-2 probabilities for every row of ``df``, one country at a time (bounded memory)."""
    p = np.empty(df.height, np.float32)
    pos = np.arange(df.height)
    for g in np.unique(code):
        m = code == g
        X = s2_features(df.filter(pl.Series(m)))
        assert X.columns == names_expected, "stage-2 feature mismatch"
        p[pos[m]] = model.predict(X.to_numpy(), num_threads=config.N_JOBS)
        del X
    return p


def main(frac: float = 0.3):
    """Fit the stage-2 model on out-of-fold p1 and evaluate on validation entities."""
    t0 = time.time()
    df = pl.read_parquet(os.path.join(config.CACHE_DIR, "p1_train.parquet"))
    q = df["q"].to_numpy()
    is_val = df["is_val"].to_numpy()
    fit_q = pl.Series(np.unique(q[~is_val])).sample(fraction=frac, seed=config.SEED).to_numpy()
    fit = np.isin(q, fit_q)
    rest_q = np.setdiff1d(np.unique(q[~is_val]), fit_q)
    es_q = np.random.default_rng(0).choice(rest_q, size=max(1, len(rest_q) // 20), replace=False)
    es = np.isin(q, es_q)
    rows, Xs, names = s2_by_country(df, country_codes("train", q), keep=fit | es | is_val)
    y = df["y"].to_numpy()[rows]
    fit_r, es_r, val_r = fit[rows], es[rows], is_val[rows]
    print(f"stage-2 features for {len(rows)} rows x {len(names)} ({time.time() - t0:.0f}s)", flush=True)
    dtr = lgb.Dataset(Xs[fit_r], y[fit_r], feature_name=names)
    dva = lgb.Dataset(Xs[es_r], y[es_r], reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    m.save_model(os.path.join(config.MODEL_DIR, "lgb_s2.txt"))
    p2 = m.predict(Xs[val_r], num_threads=config.N_JOBS).astype(np.float32)
    vp = (df[rows[val_r]].select("q", "c", "y", "p1").with_columns(pl.Series("p", p2)))
    vp.write_parquet(os.path.join(config.CACHE_DIR, "val_pred2.parquet"))
    print(f"stage 2 fitted, best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)

    val_c = np.load(os.path.join(config.CACHE_DIR, "val_entities.npy"))
    s1, other = load_split("train", columns=["entity_id"])
    vtruth = true_pairs(s1, other).filter(pl.col("c").is_in(pl.Series(val_c).implode()))
    res = evaluate(vp.select("q", "c", "p"), vtruth, val_c)
    for k, v in res.items():
        print(f"  stage2 {k}: macro F0.5 = {v:.5f}")
    best = max(res, key=res.get)
    print(f"stage2 best decoder {best}: {res[best]:.5f}")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, round(g / 1e6, 1)) for f, g in imp[:15]])
    with open(os.path.join(config.MODEL_DIR, "stage1.json")) as f:
        s1best = json.load(f)["scores"]
    use2 = res[best] > max(s1best.values())
    dec = {"stage": 2 if use2 else 1, "best": best if use2 else max(s1best, key=s1best.get),
           "scores": res, "scores_stage1": s1best, "features2": names}
    with open(os.path.join(config.MODEL_DIR, "decoder.json"), "w") as f:
        json.dump(dec, f, indent=1)
    print(f"decoder choice: stage {dec['stage']} / {dec['best']}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.3)
