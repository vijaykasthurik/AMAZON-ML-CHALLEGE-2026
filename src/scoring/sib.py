"""Sibling-consistency features for the collective (stage 2) model.

All true matches of one Source 1 entity are variants of the same business, so a candidate that
closely resembles the entity's OTHER confident candidates is likely a match too - even when its
own evidence against the Source 1 record is weak (e.g. a name-only record with an empty address).

For every pair (id1, id2) we compare id2 with the "siblings": the other candidates of id1 whose
stage-1 probability p1 > SIB_P. Features: max / mean name similarity, max address similarity,
max number overlap, and the number of confident siblings.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from data import load_norm

SIB_P = 0.5
PARTS = 4


def add_siblings(full: pl.DataFrame, split: str) -> pl.DataFrame:
    """Return `full` (needs id1, id2, p1) with sib_* columns added."""
    tx = load_norm(split, columns=["id", "nc", "at", "num"]).with_columns(
        pl.col("nc").fill_null(""), pl.col("at").fill_null(""), pl.col("num").fill_null(""))
    base = full.select("id1", "id2", "p1")
    conf = base.filter(pl.col("p1") > SIB_P).select("id1", pl.col("id2").alias("sib"))
    out = []
    for j in range(PARTS):
        part = base.filter(pl.col("id1").hash(seed=5) % PARTS == j).select("id1", "id2")
        pr = part.join(conf, on="id1").filter(pl.col("id2") != pl.col("sib"))
        pr = (pr.join(tx.rename({"id": "id2"}), on="id2", how="left")
                .join(tx.rename({"id": "sib", "nc": "nc_s", "at": "at_s", "num": "num_s"}), on="sib", how="left"))
        if pr.height == 0:
            continue
        nm = process.cpdist(pr["nc"].to_list(), pr["nc_s"].to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.float32) / 100
        ad = process.cpdist(pr["at"].to_list(), pr["at_s"].to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.float32) / 100
        pr = pr.with_columns(pl.Series("s_nm", nm), pl.Series("s_ad", ad),
                             (pl.col("num").str.split(" ").list.set_intersection(pl.col("num_s").str.split(" "))
                              .list.eval(pl.element().filter(pl.element() != "")).list.len() > 0)
                             .cast(pl.Float32).alias("s_num"))
        out.append(pr.group_by("id1", "id2").agg(
            pl.col("s_nm").max().alias("sib_nm_max"), pl.col("s_nm").mean().alias("sib_nm_mean"),
            pl.col("s_ad").max().alias("sib_ad_max"), pl.col("s_num").max().alias("sib_num"),
            pl.len().cast(pl.Float32).alias("sib_n")))
    agg = pl.concat(out) if out else pl.DataFrame(schema={"id1": pl.String, "id2": pl.String})
    full = full.join(agg, on=["id1", "id2"], how="left")
    return full.with_columns(pl.col("sib_nm_max", "sib_nm_mean", "sib_ad_max", "sib_num").fill_null(-1.0),
                             pl.col("sib_n").fill_null(0.0))
