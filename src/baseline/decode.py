"""Turn pair probabilities into per-entity match lists, and score them.

Decoding uses two facts from the data:

1. Assignment constraint - in the training ground truth every Source 2/3
   record is matched to at most one Source 1 entity. So each query keeps only
   its highest-probability Source 1 candidate.

2. The metric is a per-entity F0.5 averaged over entities. For one Source 1
   entity with assigned candidates of probability p1 >= p2 >= ... we predict
   the top-m, choosing m to maximise the expected F0.5:

       E[F(m)] ~= 1.25 * sum_{i<=m} p_i / (0.25 * E|T| + m)      (m >= 1)
       E[F(0)]  = P(no true match) ~= prod_i (1 - p_i) * (1 - p_miss)

   where E|T| = sum_i p_i + miss (expected true matches, including ones the
   candidate set did not recover). This naturally predicts "no match" for
   entities whose best candidates are uncertain - they are probably singletons.
   A plain probability threshold is also provided; both are tuned on held-out
   training entities.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl


def best_per_query(pairs: pl.DataFrame, pcol: str = "p") -> pl.DataFrame:
    """Keep, for every query q, only its highest-probability candidate c."""
    return pairs.sort(pcol, descending=True).unique(subset=["q"], keep="first")


def threshold_decode(pairs: pl.DataFrame, t: float, pcol: str = "p") -> pl.DataFrame:
    """Assignment + global threshold. Returns (q, c) predicted pairs."""
    b = best_per_query(pairs, pcol)
    return b.filter(pl.col(pcol) >= t).select("q", "c")


@nb.njit(cache=True)
def _efmax(order_c, p, starts, miss, t_min):
    """For each entity segment (sorted by c, p desc) pick m maximising E[F0.5]."""
    keep = np.zeros(len(p), np.bool_)
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        tot = 0.0
        pnone = 1.0
        for i in range(s, e):
            tot += p[i]
            pnone *= (1.0 - p[i])
        et = tot + miss
        best_m = 0
        best = pnone * (1.0 - min(miss, 1.0))
        cum = 0.0
        for m in range(1, e - s + 1):
            if p[s + m - 1] < t_min:
                break
            cum += p[s + m - 1]
            val = 1.25 * cum / (0.25 * et + m)
            if val > best:
                best = val
                best_m = m
        for i in range(s, s + best_m):
            keep[i] = True
    return keep


def expected_f_decode(pairs: pl.DataFrame, miss: float = 0.1, t_min: float = 0.0,
                      pcol: str = "p") -> pl.DataFrame:
    """Assignment + per-entity expected-F0.5 subset selection."""
    b = best_per_query(pairs, pcol).sort(["c", pcol], descending=[False, True])
    c = b["c"].to_numpy()
    p = b[pcol].to_numpy().astype(np.float64)
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True]).astype(np.int64)
    keep = _efmax(c, p, starts, miss, t_min)
    return b.filter(pl.Series(keep)).select("q", "c")


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, entities: np.ndarray) -> float:
    """Macro F0.5 over ``entities`` (S1 row numbers) from (q, c) pair tables."""
    ent = pl.DataFrame({"c": entities.astype(np.int32)})
    tp = pred.join(truth, on=["q", "c"], how="inner").group_by("c").len().rename({"len": "tp"})
    npred = pred.group_by("c").len().rename({"len": "np"})
    ntrue = truth.group_by("c").len().rename({"len": "nt"})
    t = (ent.join(npred, on="c", how="left").join(ntrue, on="c", how="left")
         .join(tp, on="c", how="left").fill_null(0))
    f = t.select(
        pl.when(pl.col("nt") == 0)
        .then((pl.col("np") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")))
        .alias("f"))
    return float(f["f"].mean())


def write_outputs(s1_ids: list, pred: pl.DataFrame, cands: pl.DataFrame,
                  s1_tab: pl.DataFrame, other_tab: pl.DataFrame,
                  match_path: str, cand_path: str):
    """Write matching_results.tsv and candidate_pairs.tsv (tab separated)."""
    oid = other_tab["entity_id"]

    def lists(pairs: pl.DataFrame) -> pl.DataFrame:
        return (pairs.select(pl.col("c"), oid[pairs["q"].to_numpy()].alias("eid"))
                .group_by("c").agg(pl.col("eid").unique().sort().str.join(",")))

    base = pl.DataFrame({"c": np.arange(s1_tab.height, dtype=np.int32),
                         "source1_entity_id": s1_tab["entity_id"]})
    for pairs, path, col in ((pred, match_path, "matched_entity_ids"),
                             (cands, cand_path, "candidate_entity_ids")):
        out = (base.join(lists(pairs.select("q", "c")), on="c", how="left")
               .sort("c").select("source1_entity_id",
                                 pl.col("eid").fill_null("").alias(col)))
        out.write_csv(path, separator="\t", quote_style="never")
