"""Blocking audit on train: pair recall, recall by forward rank, and the oracle macro F0.5
(the score a perfect matcher would get restricted to our candidate set)."""
import os
import polars as pl
from config import WORK
from data import load_norm
from features import load_truth
import progress

cand = pl.read_parquet(os.path.join(WORK, "train_cand.parquet"))
s1 = load_norm("train", sources=(1,), columns=["id", "country"])
truth = load_truth().join(s1.select(pl.col("id").alias("id1")), on="id1", how="semi")
cand = cand.with_columns(pl.col("s_blk").rank("ordinal", descending=True).over("id1").alias("rk"),
                         pl.col("s_blk").rank("ordinal", descending=True).over("id2").alias("rk2"))
jj = truth.join(cand, on=["id1", "id2"], how="left")
for k in (5, 10, 15, 20):
    for r in (1, 2, 3):
        keep = (cand["rk"] <= k) | (cand["rk2"] <= r)
        print(f"K_KEEP={k} R_KEEP={r}: recall", float(((jj["rk"] <= k) | (jj["rk2"] <= r)).mean()),
              "pairs", int(keep.sum()))
j = truth.join(cand, on=["id1", "id2"], how="left").join(s1, left_on="id1", right_on="id")
print("candidate pairs", cand.height, "per S1", cand.height / s1.height)
print("pair recall overall", j["s_blk"].is_not_null().mean())
print(j.group_by("country").agg(pl.col("s_blk").is_not_null().mean().alias("recall"), pl.len()))
for k in (5, 10, 15, 20, 30, 40):
    print(f"recall@rank<={k}", (j["rk"] <= k).mean(), " pairs kept", (cand["rk"] <= k).sum())
print("found only by reverse pass", (j["in_fwd"].not_() & j["in_rev"]).sum())
# oracle macro F0.5: predict exactly truth ∩ candidates
per = j.group_by("id1").agg(pl.len().alias("t"), pl.col("s_blk").is_not_null().sum().alias("h"))
allids = s1.select(pl.col("id").alias("id1")).join(per, on="id1", how="left").fill_null(0)
f = allids.with_columns(
    pl.when(pl.col("t") == 0).then(1.0)
      .when(pl.col("h") == 0).then(0.0)
      .otherwise(1.25 * (pl.col("h") / pl.col("t")) / (0.25 + pl.col("h") / pl.col("t"))).alias("f"))
print("oracle macro F0.5", f["f"].mean())
progress.metric("Blocking recall (train)", round(float(j["s_blk"].is_not_null().mean()), 5))
progress.metric("Oracle F0.5 ceiling", round(float(f["f"].mean()), 5))
miss = j.filter(pl.col("s_blk").is_null()); miss = miss.sample(min(30, miss.height), seed=0)
n = load_norm("train", columns=["id", "nf", "at", "num"])
miss = miss.join(n, left_on="id1", right_on="id").join(n, left_on="id2", right_on="id", suffix="_b")
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(260)
print(miss.select("country", "nf", "nf_b", "at", "at_b", "num", "num_b"))
