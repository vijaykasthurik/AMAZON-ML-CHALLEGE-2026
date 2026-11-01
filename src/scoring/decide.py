"""Decision layer shared by validation and inference.

1. One-owner constraint: a Source 2/3 record can belong to at most one Source 1
   entity (true for 100% of training links), so each target keeps only its best owner.
2. Expected-F0.5 subset selection per Source 1 entity: with calibrated probabilities
   sorted descending, choose the prefix size k (including k = 0, "singleton") that
   maximises the expected F0.5 approximation
        E_k = 1.25 * sum_{i<=k} p_i / (0.25 * sum_all p_i + k),   E_0 = prod(1 - p_i)
   An optional hard floor `thr` removes pairs below a minimum probability.
"""
import polars as pl


def one_owner(df: pl.DataFrame, p: str = "p") -> pl.DataFrame:
    """Keep, for every target id2, only the row(s) of its highest-probability Source 1."""
    return df.filter(pl.col(p) == pl.col(p).max().over("id2"))


def expected_f(df: pl.DataFrame, p: str = "p", alpha: float = 1.0, thr: float = 0.0) -> pl.DataFrame:
    """Return selected (id1, id2) rows by expected-F0.5 maximisation per id1."""
    d = df.select("id1", "id2", (pl.col(p).clip(1e-6, 1 - 1e-6) ** alpha).alias("q"))
    d = d.sort(["id1", "q"], descending=[False, True])
    d = d.with_columns(
        pl.col("q").cum_sum().over("id1").alias("cq"),
        pl.int_range(1, pl.len() + 1).over("id1").alias("k"),
        pl.col("q").sum().over("id1").alias("sq"),
        (1 - pl.col("q")).log().sum().over("id1").exp().alias("e0"),
    )
    d = d.with_columns((1.25 * pl.col("cq") / (0.25 * pl.col("sq") + pl.col("k"))).alias("ek"))
    best = d.group_by("id1").agg(pl.col("ek").max().alias("emax"), pl.col("e0").first(),
                                 pl.col("k").filter(pl.col("ek") == pl.col("ek").max()).first().alias("kbest"))
    d = d.join(best, on="id1")
    sel = d.filter((pl.col("emax") > pl.col("e0")) & (pl.col("k") <= pl.col("kbest")) & (pl.col("q") >= thr))
    return sel.select("id1", "id2")


def threshold(df: pl.DataFrame, p: str = "p", thr: float = 0.5) -> pl.DataFrame:
    """Plain probability threshold."""
    return df.filter(pl.col(p) >= thr).select("id1", "id2")


def singleton_guard(sel: pl.DataFrame, scored: pl.DataFrame, p: str = "p", gate: float = 0.0) -> pl.DataFrame:
    """Drop an entity's whole prediction (declare it a singleton) when its best candidate
    probability is below `gate`. A false merge on a true singleton scores 0 instead of 1, so
    abstaining when the top evidence is weak protects those entities."""
    if gate <= 0:
        return sel
    keep = scored.group_by("id1").agg(pl.col(p).max().alias("pmax")).filter(pl.col("pmax") >= gate).select("id1")
    return sel.join(keep, on="id1", how="semi")


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, all_ids: pl.Series) -> float:
    """Official macro F0.5 over `all_ids` given predicted and true (id1, id2) pair tables."""
    tp = pred.join(truth, on=["id1", "id2"], how="semi").group_by("id1").agg(pl.len().alias("tp"))
    npred = pred.group_by("id1").agg(pl.len().alias("np"))
    ntrue = truth.group_by("id1").agg(pl.len().alias("nt"))
    t = (pl.DataFrame({"id1": all_ids}).join(tp, on="id1", how="left").join(npred, on="id1", how="left")
           .join(ntrue, on="id1", how="left").fill_null(0))
    t = t.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
          .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))).alias("f"))
    return float(t["f"].mean())
