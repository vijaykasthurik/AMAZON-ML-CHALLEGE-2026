"""Pair features for every (Source 1, Source 2/3) candidate pair.

Groups of features
  * fuzzy string similarity (rapidfuzz, C++ multithreaded) on names and addresses
  * IDF-weighted cosine of name / address-word / address-number token bags
  * structural agreement: numbers, state, legal form, script, website-style names
  * context features: how this pair's blocking score compares with the other
    candidates of the same Source 1 record AND of the same Source 2/3 record
Label y (train only) comes from the ground truth.
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from block import hashed_matrix
from config import CACHE, WORK
from data import load_norm
from normalize import LEGAL_CANON, phon
import progress

CH = 2_000_000
W = -1


def _cp(a, b, scorer):
    """Aligned pairwise similarity for two equal-length string lists (0..1 floats)."""
    return (process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32) / 100.0) \
        if scorer in (fuzz.ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio,
                      fuzz.WRatio, fuzz.partial_token_set_ratio) \
        else process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32)


def bag_matrix(tok_lists, n_rows):
    """IDF-weighted, L2-normalized CSR for a list-of-token-lists column (no df cap)."""
    ex = pl.DataFrame({"r": np.arange(n_rows, dtype=np.uint32), "t": tok_lists}).explode("t")
    ex = ex.filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
    dfq = ex.group_by("t").agg(pl.len().alias("df"))
    dfq = dfq.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("c"),
                           (np.log(n_rows + 1) - pl.col("df").log()).cast(pl.Float32).alias("w"))
    ex = ex.join(dfq, on="t")
    m = sp.csr_matrix((ex["w"].to_numpy(), (ex["r"].to_numpy(), ex["c"].to_numpy())),
                      shape=(n_rows, max(dfq.height, 1)), dtype=np.float32)
    nr = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    nr[nr == 0] = 1
    return (sp.diags((1 / nr).astype(np.float32)) @ m).tocsr()


def rowdot(M, ia, ib):
    """Cosine between rows ia[k] and ib[k] of M, chunked."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), 1_000_000):
        A, B = M[ia[s:s + 1_000_000]], M[ib[s:s + 1_000_000]]
        out[s:s + 1_000_000] = np.asarray(A.multiply(B).sum(1)).ravel()
    return out


def legal_set(nf):
    """Legal-form tokens present in a normalized full name."""
    return [t for t in nf.split() if t in LEGAL_CANON]


def prepare_country(rec, country):
    """Country record table with row index, helper columns and the four IDF bag matrices."""
    rec = rec.filter(pl.col("country") == country).with_row_index("ri")
    n = rec.height
    glued = [phon(s.replace(" ", "")) if s else "" for s in rec["nc"].to_list()]
    rec = rec.with_columns(pl.Series("glued", glued), pl.col("nc").str.replace_all(" ", "").alias("ncj"),
                           pl.col("at").str.split(" ").list.first().fill_null("").alias("fw"))
    # name-uniqueness: how many Source 1 / target records share this (phonetic, glued) name,
    # alone and together with the leading address word
    is1 = (pl.col("src") == 1).cast(pl.Int32)
    rec = rec.with_columns(
        is1.sum().over("glued").cast(pl.Float32).alias("cg1"),
        (1 - is1).sum().over("glued").cast(pl.Float32).alias("cgt"),
        is1.sum().over(["glued", "fw"]).cast(pl.Float32).alias("cgf1"),
    )
    mats = {k: hashed_matrix(rec[col].fill_null("").to_list(), df_cap=None, min_df=1)
            for k, col in (("cos_name", "ncp"), ("cos_addr", "at"), ("cos_num", "num"), ("cos_full", "nf"))}
    return rec, mats


def country_features(rec, mats, cand, truth):
    """Compute the feature table for a chunk of candidate pairs of one country."""
    idx = rec.select("id", "ri")
    c = (cand.join(idx, left_on="id1", right_on="id").rename({"ri": "ia"})
             .join(idx, left_on="id2", right_on="id").rename({"ri": "ib"}))
    ia, ib = c["ia"].to_numpy(), c["ib"].to_numpy()
    f = {k: rowdot(M, ia, ib) for k, M in mats.items()}

    cols = {k: rec[k] for k in ("nf", "nc", "ncp", "at", "num", "glued", "ncj", "state")}
    fuzzy = {k: [] for k in ("r_nf", "tsr_nc", "tso_nc", "pr_nc", "jw_nc", "tsr_ncp", "r_glued",
                             "pr_ncj", "r_at", "tsr_at", "pr_at", "lev_num", "tsr_na_ab", "tsr_na_ba")}
    for s in range(0, len(ia), CH):
        a, b = pl.Series(ia[s:s + CH]), pl.Series(ib[s:s + CH])
        g = {k: (v.gather(a).to_list(), v.gather(b).to_list()) for k, v in cols.items()}
        fuzzy["r_nf"].append(_cp(*g["nf"], fuzz.ratio))
        fuzzy["tsr_nc"].append(_cp(*g["nc"], fuzz.token_set_ratio))
        fuzzy["tso_nc"].append(_cp(*g["nc"], fuzz.token_sort_ratio))
        fuzzy["pr_nc"].append(_cp(*g["nc"], fuzz.partial_ratio))
        fuzzy["jw_nc"].append(_cp(*g["nc"], JaroWinkler.normalized_similarity))
        fuzzy["tsr_ncp"].append(_cp(*g["ncp"], fuzz.token_set_ratio))
        fuzzy["r_glued"].append(_cp(*g["glued"], fuzz.ratio))
        fuzzy["pr_ncj"].append(_cp(*g["ncj"], fuzz.partial_ratio))
        fuzzy["r_at"].append(_cp(*g["at"], fuzz.ratio))
        fuzzy["tsr_at"].append(_cp(*g["at"], fuzz.token_set_ratio))
        fuzzy["pr_at"].append(_cp(*g["at"], fuzz.partial_ratio))
        fuzzy["lev_num"].append(_cp(*g["num"], Levenshtein.normalized_similarity))
        # name of one side found inside the address of the other (landmark / DBA style)
        fuzzy["tsr_na_ab"].append(_cp(g["nc"][0], g["at"][1], fuzz.partial_token_set_ratio))
        fuzzy["tsr_na_ba"].append(_cp(g["nc"][1], g["at"][0], fuzz.partial_token_set_ratio))
    for k, v in fuzzy.items():
        f[k] = np.concatenate(v) if v else np.zeros(0, np.float32)

    # set-based structural features with polars list ops
    side = rec.select("ri", pl.col("num").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("nl"),
                      pl.col("ncp").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("pl_"),
                      pl.col("nf").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(list(LEGAL_CANON)))).alias("lg"),
                      "state", "web", "nat", "anull", "src", "cg1", "cgt", "cgf1", "ncp", "glued",
                      pl.col("nc").str.len_chars().alias("len_nc"), pl.col("at").str.len_chars().alias("len_at"),
                      pl.col("nc").str.count_matches(" ").alias("ntok"))
    L = pl.DataFrame({"ia": ia, "ib": ib})
    L = (L.join(side, left_on="ia", right_on="ri", how="left")
          .join(side, left_on="ib", right_on="ri", how="left", suffix="_b"))
    L = L.with_columns(
        pl.col("nl").list.set_intersection("nl_b").list.len().alias("num_inter"),
        pl.col("nl").list.set_union("nl_b").list.len().alias("num_union"),
        pl.col("nl").list.len().alias("num_na"), pl.col("nl_b").list.len().alias("num_nb"),
        pl.col("pl_").list.set_intersection("pl__b").list.len().alias("ph_inter"),
        pl.col("pl_").list.set_union("pl__b").list.len().alias("ph_union"),
        pl.col("lg").list.set_intersection("lg_b").list.len().alias("lg_inter"),
        pl.col("lg").list.len().alias("lg_na"), pl.col("lg_b").list.len().alias("lg_nb"),
        ((pl.col("state") != "") & (pl.col("state_b") != "")).alias("st_both"),
        (pl.col("state") == pl.col("state_b")).alias("st_eq"),
    )
    L = L.with_columns(
        (pl.col("num_inter") / pl.col("num_union").clip(1)).alias("num_jacc"),
        (pl.col("ph_inter") / pl.col("ph_union").clip(1)).alias("ph_jacc"),
        ((pl.col("num_na") > 0) & (pl.col("num_nb") > 0) & (pl.col("num_inter") == 0)).alias("num_conflict"),
        (pl.col("num_na") - pl.col("num_inter")).alias("num_a_miss"),
        (pl.col("num_nb") - pl.col("num_inter")).alias("num_b_extra"),
        ((pl.col("lg_na") > 0) & (pl.col("lg_nb") > 0) & (pl.col("lg_inter") == 0)).alias("lg_conflict"),
        (pl.col("st_both") & ~pl.col("st_eq")).alias("st_conflict"),
        (pl.col("len_nc") - pl.col("len_nc_b")).abs().alias("dlen_nc"),
        (pl.col("ntok") - pl.col("ntok_b")).alias("dntok"),
        # name-exact match: phonetic core identical, or glued phonetic name identical
        ((pl.col("ncp") == pl.col("ncp_b")) & (pl.col("ncp") != "")).cast(pl.Int8).alias("name_exact"),
        ((pl.col("glued") == pl.col("glued_b")) & (pl.col("glued") != "")).cast(pl.Int8).alias("glued_exact"),
        # is one side's address missing (unknown, not a conflict)?
        ((pl.col("anull") == 1) | (pl.col("anull_b") == 1)).cast(pl.Int8).alias("addr_missing"),
    )
    # strong name + missing address: the case the model was wrongly discarding
    L = L.with_columns(
        (((pl.col("ncp") == pl.col("ncp_b")) | (pl.col("glued") == pl.col("glued_b")))
         & ((pl.col("anull") == 1) | (pl.col("anull_b") == 1))).cast(pl.Int8).alias("exactname_noaddr"),
        # rare exact name (few records share it) is very strong evidence on its own
        pl.when((pl.col("ncp") == pl.col("ncp_b")) & (pl.col("ncp") != ""))
          .then(pl.col("cgt_b") + pl.col("cg1")).otherwise(999.0).cast(pl.Float32).alias("exactname_share"),
    )
    keep = ["num_inter", "num_na", "num_nb", "num_jacc", "num_conflict", "num_a_miss", "num_b_extra",
            "ph_inter", "ph_jacc", "lg_inter", "lg_conflict", "st_both", "st_eq", "st_conflict",
            "web_b", "nat_b", "anull", "anull_b", "src_b", "len_nc", "len_nc_b", "len_at", "len_at_b",
            "dlen_nc", "ntok", "dntok", "cg1", "cg1_b", "cgt_b", "cgf1", "cgf1_b",
            "name_exact", "glued_exact", "addr_missing", "exactname_noaddr", "exactname_share"]
    out = pl.DataFrame(f).with_columns(L.select(keep).with_columns(pl.all().cast(pl.Float32)))
    out = pl.concat([c.drop("ia", "ib"), out], how="horizontal")
    for col, tag in (("cos_name", "cn"), ("cos_addr", "ca"), ("cos_full", "cf"), ("tsr_nc", "tn")):
        out = out.with_columns(
            pl.col(col).rank("ordinal", descending=True).over("id1").cast(pl.Float32).alias(f"rk1_{tag}"),
            (pl.col(col) - pl.col(col).max().over("id1")).alias(f"gap1_{tag}"))
    if truth is not None:
        out = out.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["id1", "id2"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    return out


def add_context(df, col, tag):
    """Rank / gap / second-best margin of `col` within the Source 1 group (id1) and the target group (id2).

    Window expressions are materialized first and the second-best value is attached with a
    join: nesting a windowed rank inside another window is quadratic in polars.
    """
    df = df.with_columns(
        pl.col(col).rank("ordinal", descending=True).over("id1").cast(pl.Float32).alias(f"rk1_{tag}"),
        pl.col(col).rank("ordinal", descending=True).over("id2").cast(pl.Float32).alias(f"rk2_{tag}"),
        (pl.col(col) - pl.col(col).max().over("id1")).alias(f"gap1_{tag}"),
        (pl.col(col) - pl.col(col).max().over("id2")).alias(f"gap2_{tag}"),
    )
    sec = df.filter(pl.col(f"rk2_{tag}") == 2).select("id2", pl.col(col).alias("_sec"))
    df = df.join(sec, on="id2", how="left").with_columns(
        (pl.col(col) - pl.col("_sec").fill_null(0.0)).alias(f"marg2_{tag}")).drop("_sec")
    if tag == "blk":
        df = df.with_columns(pl.len().over("id1").cast(pl.Float32).alias("n1"),
                             pl.len().over("id2").cast(pl.Float32).alias("n2"))
    return df


def load_truth():
    """Exploded ground-truth pairs (id1, id2) for the training split."""
    gt = pl.read_parquet(os.path.join(CACHE, "train_ground_truth.parquet"))
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("id1"), pl.col("matched_entity_ids").alias("id2")))


K_KEEP = int(os.environ.get("BER_KKEEP", "15"))   # forward rank kept per Source 1
R_KEEP = int(os.environ.get("BER_RKEEP", "2"))    # reverse rank kept per target
N_CHUNK = int(os.environ.get("BER_NCHUNK", "4"))


def pruned_candidates(split):
    """Full candidate table + blocking context, pruned to the final candidate set.

    Context features over id2 (the target) are computed here on the *full* table so
    they mean the same thing on train and test.
    """
    cand = pl.read_parquet(os.path.join(WORK, f"{split}_cand.parquet"))
    cand = add_context(cand, "s_blk", "blk")
    cand = cand.filter((pl.col("rk1_blk") <= K_KEEP) | (pl.col("rk2_blk") <= R_KEEP))
    return cand.with_columns(pl.col("in_fwd").cast(pl.Int8), pl.col("in_rev").cast(pl.Int8))


def cheap_frame(rec, mats, cand):
    """Tier-0 features for ALL retrieved pairs: blocking context + IDF cosines (fully vectorized)."""
    idx = rec.select("id", "ri")
    c = (cand.join(idx, left_on="id1", right_on="id").rename({"ri": "ia"})
             .join(idx, left_on="id2", right_on="id").rename({"ri": "ib"}))
    ia, ib = c["ia"].to_numpy(), c["ib"].to_numpy()
    c = c.drop("ia", "ib").with_columns([pl.Series(k, rowdot(M, ia, ib)) for k, M in mats.items()])
    for col, tag in (("cos_name", "cn"), ("cos_addr", "ca"), ("cos_full", "cf"), ("cos_num", "cd")):
        c = c.with_columns(
            pl.col(col).rank("ordinal", descending=True).over("id1").cast(pl.Float32).alias(f"q0rk1_{tag}"),
            (pl.col(col) - pl.col(col).max().over("id1")).alias(f"q0gap1_{tag}"),
            pl.col(col).rank("ordinal", descending=True).over("id2").cast(pl.Float32).alias(f"q0rk2_{tag}"),
            (pl.col(col) - pl.col(col).max().over("id2")).alias(f"q0gap2_{tag}"))
    return c


MODELS_DIR = os.environ.get("BER_MODELS", os.path.join(os.environ.get("BER_OUTPUTS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "models"))


def cheap_run(split):
    """Pass 1: tier-0 features for every retrieved pair, one parquet per country chunk."""
    cand = pruned_candidates(split)
    rec_all = load_norm(split)
    for ctry in rec_all["country"].unique().sort().to_list():
        paths = [os.path.join(WORK, f"{split}_cheap_{ctry}_{j}.parquet") for j in range(N_CHUNK)]
        if all(os.path.exists(p) for p in paths):
            continue
        t0 = time.time()
        rec, mats = prepare_country(rec_all, ctry)
        s1ids = rec.filter(pl.col("src") == 1).select(pl.col("id").alias("id1"))
        cc_all = cand.join(s1ids, on="id1", how="semi")
        for j, p in enumerate(paths):
            cheap_frame(rec, mats, cc_all.filter(pl.col("id1").hash(seed=7) % N_CHUNK == j)).write_parquet(p)
        print(split, ctry, "cheap", cc_all.height, f"{time.time() - t0:.0f}s", flush=True)
        progress.update(f"Features ({split})", msg=f"{split} {ctry}: tier-0 features for {cc_all.height:,} pairs")
        del rec, mats, cc_all


def run(split):
    """Pass 2: full features only for pairs surviving the tier-0 pruner (BER_TIERED=0: all pairs)."""
    import json
    import xgboost as xgb
    stage = "Features (train)" if split == "train" else "Features + scoring (test)"
    tiered = os.environ.get("BER_TIERED", "1") == "1"
    truth = load_truth() if split == "train" else None
    rec_all = load_norm(split)
    countries = rec_all["country"].unique().sort().to_list()
    if tiered:
        cfg0 = json.load(open(os.path.join(MODELS_DIR, "prune0.json")))
        b0 = xgb.Booster()
        b0.load_model(os.path.join(MODELS_DIR, "prune0_model.json"))
    else:
        cand = pruned_candidates(split)
    total, done = len(countries) * N_CHUNK, 0
    for ctry in countries:
        paths = [os.path.join(WORK, f"{split}_feat_{ctry}_{j}.parquet") for j in range(N_CHUNK)]
        if all(os.path.exists(p) for p in paths):
            done += N_CHUNK
            continue
        t0 = time.time()
        rec, mats = prepare_country(rec_all, ctry)
        if not tiered:
            s1ids = rec.filter(pl.col("src") == 1).select(pl.col("id").alias("id1"))
            cc_all = cand.join(s1ids, on="id1", how="semi")
        for j, p in enumerate(paths):
            if not os.path.exists(p):
                if tiered:
                    ch = pl.read_parquet(os.path.join(WORK, f"{split}_cheap_{ctry}_{j}.parquet"))
                    p0 = b0.predict(xgb.DMatrix(ch.select(cfg0["features"]).to_numpy().astype(np.float32)))
                    cc = ch.with_columns(pl.Series("p0", p0)).filter(pl.col("p0") >= cfg0["tau0"])
                    cc = cc.select([c for c in cc.columns if not c.startswith("cos_") and c != "p0"])
                    print(split, ctry, j, "tier-0 kept", cc.height, "of", ch.height, flush=True)
                else:
                    cc = cc_all.filter(pl.col("id1").hash(seed=7) % N_CHUNK == j)
                feat = country_features(rec, mats, cc, truth)
                feat.write_parquet(p)
                print(split, ctry, j, feat.shape, f"{time.time() - t0:.0f}s", flush=True)
                del feat, cc
            done += 1
            progress.update(stage, done, total, msg=f"{split} {ctry} chunk {j + 1}/{N_CHUNK} featurized "
                                                    f"({time.time() - t0:.0f}s)")
        del rec, mats
    progress.update(stage, total, total, state="done")


def fit_prune0():
    """Tier-0 pruner on cheap features (fit buckets < 30); tau0 chosen on holdout buckets >= 50
    as the largest cut losing <= 0.1% of true pairs."""
    import glob
    import json
    import xgboost as xgb
    truth = load_truth()
    files = sorted(glob.glob(os.path.join(WORK, "train_cheap_*_*.parquet")))
    h = pl.col("id1").hash(seed=11)
    fx = [(h % 5).cast(pl.Int8).alias("fold"), ((h // 5) % 100).cast(pl.Int8).alias("bucket")]
    lab = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))

    def load(p, lo, hi):
        """Rows of one cheap file whose Source 1 bucket is in [lo, hi), with labels."""
        d = pl.read_parquet(p).with_columns(fx).filter((pl.col("bucket") >= lo) & (pl.col("bucket") < hi))
        return d.join(lab, on=["id1", "id2"], how="left").with_columns(pl.col("y").fill_null(0))

    # fit on half of the model-fitting buckets (plenty of rows for a 30-feature model; bounded memory)
    tr = pl.concat([load(p, 0, 15) for p in files])
    cols = [c for c in tr.columns if c not in ("id1", "id2", "y", "fold", "bucket")]
    params = dict(tree_method="hist", device="cuda" if os.environ.get("BER_CPU") != "1" else "cpu",
                  objective="binary:logistic", max_depth=8, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.9, eval_metric="logloss", seed=42)
    b = xgb.train(params, xgb.DMatrix(tr.select(cols).to_numpy().astype(np.float32), label=tr["y"].to_numpy()), 400)
    del tr
    # evaluate on holdout buckets file by file
    taus = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    lost, kept, npos, n1 = {t: 0 for t in taus}, {t: 0 for t in taus}, 0, 0
    for p in files:
        ho = load(p, 50, 100)
        p0 = b.predict(xgb.DMatrix(ho.select(cols).to_numpy().astype(np.float32)))
        y = ho["y"].to_numpy()
        npos += int(y.sum())
        n1 += ho["id1"].n_unique()
        for t in taus:
            lost[t] += int(((p0 < t) & (y == 1)).sum())
            kept[t] += int((p0 >= t).sum())
    tau0 = 0.0
    for t in taus:
        print(f"  tier-0 tau0={t}: positives lost {lost[t]}/{npos} ({lost[t] / max(npos, 1):.5f}), "
              f"pairs/S1 {kept[t] / max(n1, 1):.2f}", flush=True)
        if lost[t] / max(npos, 1) <= 0.001:
            tau0 = t
    os.makedirs(MODELS_DIR, exist_ok=True)
    b.save_model(os.path.join(MODELS_DIR, "prune0_model.json"))
    json.dump({"features": cols, "tau0": tau0}, open(os.path.join(MODELS_DIR, "prune0.json"), "w"), indent=1)
    print("  chosen tau0", tau0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    run(ap.parse_args().split)
