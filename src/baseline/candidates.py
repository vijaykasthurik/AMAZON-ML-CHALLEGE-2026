"""Run blocking for a split and save the candidate pair table.

Output: ``cache/cands_{split}.parquet`` with one row per (query, S1) pair:
  q      row number in the concatenated Source 2 + Source 3 table
  c      row number in the Source 1 table
  sn, sa name / address TF-IDF cosine (index grams only)
  rk_n, rk_a, rk_c  rank of the pair in the name / address / combined list

For the training split it also prints the recall ceiling of the candidate set
(fraction of true pairs retrieved) for several list depths, which is how the
list depth K was chosen.

Usage:  python candidates.py train|test
"""

from __future__ import annotations

import _env  # noqa: F401  (must come before numpy)

import os
import sys
import time

from multiprocessing import Pool

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from blocking import CountryIndex  # noqa: E402
from preprocess import norm_path  # noqa: E402


BLOCK_COLS = ["entity_id", "country", "nk", "nc", "na", "ac"]


def load_split(split: str, columns: list | None = BLOCK_COLS):
    """Load normalised S1 and the concatenated S2+S3 table for a split.

    Only ``columns`` are read (the blocking stage needs six), which keeps the
    ~12M-record tables within a 16 GB machine.
    """
    s1 = pl.read_parquet(norm_path(split, 1), columns=columns)
    other = pl.concat([pl.read_parquet(norm_path(split, 2), columns=columns),
                       pl.read_parquet(norm_path(split, 3), columns=columns)])
    return s1, other


def true_pairs(s1: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    """Training ground truth as (q, c) row-number pairs."""
    gt = pl.read_csv(config.gt_path(), separator="\t", quote_char=None, infer_schema=False)
    p = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").drop_nulls()
         .rename({"source1_entity_id": "s1id", "matched_entity_ids": "eid"}))
    s1i = s1.select(pl.col("entity_id").alias("s1id"), pl.int_range(pl.len()).cast(pl.Int32).alias("c"))
    oi = other.select(pl.col("entity_id").alias("eid"), pl.int_range(pl.len()).cast(pl.Int32).alias("q"))
    return p.join(s1i, on="s1id").join(oi, on="eid").select("q", "c")


# Blocking settings, chosen with a recall study on 200k sampled train queries per
# country (see docs): grams pruned above 0.2% document frequency, composite
# keys above 1500 records; each of the six lists keeps its exact top 3.
BLOCK = dict(max_df_frac=0.002, max_df_key=1500, P_name=12, P_addr=16, K_probe=10, K_final=3)
RANKS = ["rk_n", "rk_a", "rk_c", "rk_kn", "rk_ka", "rk_kc"]


def generate(split: str, cfg: dict = BLOCK) -> pl.DataFrame:
    """Blocking for every country of ``split``; returns and caches the candidate frame.

    Each query first collects the top ``K_probe`` Source 1 records by name, by
    address and by the combined score (using only its rarest grams); all of them
    are rescored with exact cosines and a pair is kept when it is within the
    exact top ``K_final`` of at least one of the six lists (gram / key x name /
    address / combined).
    """
    s1, other = load_split(split)
    s1 = s1.with_row_index("row")
    other = other.with_row_index("row")
    parts = []
    pool = Pool(min(8, config.N_JOBS))
    for country in s1["country"].unique().sort().to_list():
        t0 = time.time()
        s1c = s1.filter(pl.col("country") == country)
        qc = other.filter(pl.col("country") == country)
        print(f"[{split}] {country}: {s1c.height} S1, {qc.height} queries", flush=True)
        index = CountryIndex(s1c, max_df_frac=cfg["max_df_frac"],
                             max_df_key=cfg["max_df_key"], pool=pool)
        cf = index.search(qc, qc["row"].to_numpy(), s1c["row"].to_numpy(), K=cfg["K_probe"],
                          P_name=cfg["P_name"], P_addr=cfg["P_addr"])
        k = cfg["K_final"]
        cf = cf.filter(pl.any_horizontal([pl.col(r) < k for r in RANKS]))
        parts.append(cf)
        print(f"  {cf.height} pairs, {cf.height / max(qc.height, 1):.2f} per query "
              f"({time.time() - t0:.0f}s)", flush=True)
        del index
    pool.close()
    cands = pl.concat(parts)
    cands.write_parquet(os.path.join(config.CACHE_DIR, f"cands_{split}.parquet"))
    return cands


def recall_report(cands: pl.DataFrame, truth: pl.DataFrame, K: int):
    """Print recall of true pairs and pairs-per-query for list depths k <= K."""
    lab = truth.with_columns(pl.lit(1).alias("y"))
    j = cands.join(lab, on=["q", "c"], how="left").with_columns(pl.col("y").fill_null(0))
    n_true, n_q = truth.height, cands["q"].n_unique()
    for k in range(1, K + 1):
        sub = j.filter(pl.any_horizontal([pl.col(r) < k for r in RANKS]))
        print(f"  k={k}: recall={sub['y'].sum() / n_true:.4f}  pairs={sub.height} "
              f"({sub.height / n_q:.2f} per query)")
    for r in RANKS:
        print(f"  only {r}<{K}: recall={j.filter(pl.col(r) < K)['y'].sum() / n_true:.4f}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t0 = time.time()
    cands = generate(split)
    print(f"total {cands.height} pairs in {time.time() - t0:.0f}s")
    if split == "train":
        s1, other = load_split("train")
        recall_report(cands, true_pairs(s1, other), BLOCK["K_final"])
