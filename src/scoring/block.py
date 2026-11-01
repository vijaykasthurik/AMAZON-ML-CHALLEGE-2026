"""Candidate generation (blocking) — two-way sparse TF-IDF retrieval per country.

Each record becomes a bag of typed tokens:
  n:<phonetic core-name token>   g:<phonetic glued core name>   a:<address word>
  d:<address number>             x:<number>_<first address word>  (house-number+street key)
weighted by IDF over the country's full record pool (tokens with df == 1 or df > DF_CAP
are dropped from blocking only). Rows are L2-normalized so the dot product is a cosine.

Forward pass: every Source 1 record keeps its top K_FWD Source 2/3 records.
Reverse pass: every Source 2/3 record keeps its top K_REV Source 1 records.
The union of both is the candidate set. The reverse pass is the key idea: since a
Source 2/3 record belongs to at most one Source 1 entity, "which Source 1 does this
record prefer?" is as informative as "which records does Source 1 prefer?".
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from config import WORK
from data import load_norm
from normalize import phon
import progress

DF_CAP = int(os.environ.get("BER_DFCAP", "3000"))
K_FWD = int(os.environ.get("BER_KFWD", "40"))
K_REV = int(os.environ.get("BER_KREV", "6"))
BLOCK_V = int(os.environ.get("BER_BLOCKV", "2"))
TRI_WORDS = int(os.environ.get("BER_TRIW", "2"))
TRI_MODE = os.environ.get("BER_TRIMODE", "first")
K_REV_SPARSE = int(os.environ.get("BER_KREVSP", "0"))
THREADS = int(os.environ.get("BER_THREADS", "12"))


HASH_BITS = 23


def docs_for(df: pl.DataFrame, word_df=None):
    """One space-joined typed-token document per record (same token scheme as `tokens`)."""
    out = []
    for nc, ncp, at, num in df.select("nc", "ncp", "at", "num").iter_rows():
        nt = ncp.split() if ncp else []
        t = ["n:" + x for x in nt]
        # unordered name-word pairs: rare even when each word is common
        for i in range(min(len(nt), 5)):
            for j in range(i + 1, min(len(nt), 5)):
                t.append("p:" + nt[i] + "_" + nt[j])
        g = phon(nc.replace(" ", "")) if nc else ""
        if g:
            t.append("g:" + g)
        aw = list(dict.fromkeys(at.split()))[:8] if at else []
        t += ["a:" + x for x in aw]
        if g:
            t += ["ga:" + g + "_" + x for x in aw]
        nums = num.split()[:4] if num else []
        for x in nums:
            t.append("d:" + x)
            if g:
                t.append("gd:" + g + "_" + x)
            t += ["x:" + x + "_" + w for w in aw]
            t += ["nd:" + w + "_" + x for w in nt[:4]]
        if BLOCK_V >= 2:
            # adjacent address-word pairs: survive a completely different (DBA) name
            t += ["ab:" + aw[i] + "_" + aw[i + 1] for i in range(len(aw) - 1)]
            # name character trigrams x leading address words: survive typos in the name
            raw = nc.replace(" ", "") if nc else ""
            tri = list(dict.fromkeys(raw[i:i + 3] for i in range(len(raw) - 2)))[:14]
            anchor = aw[:TRI_WORDS]
            if word_df is not None:
                # rarest (but shared, df >= 2) address words anchor the typo keys: generic leading
                # words such as "th" / "road" / "main" (common in Indian addresses) carry no identity
                cand_w = [w for w in aw if word_df.get(w, 0) >= 2]
                anchor = sorted(cand_w, key=lambda w: word_df.get(w, 0))[:TRI_WORDS] or anchor
            for w in anchor:
                t += ["tw:" + q + "_" + w for q in tri]
        out.append(" ".join(t))
    return out


def hashed_matrix(docs, df_cap=None, min_df=2):
    """IDF-weighted L2-normalized CSR from token documents via feature hashing (low memory)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=2 ** HASH_BITS, analyzer=str.split, alternate_sign=False,
                           norm=None, binary=True, dtype=np.float32)
    m = hv.transform(docs).tocsr()
    dfreq = np.bincount(m.indices, minlength=m.shape[1])
    idf = (np.log(m.shape[0]) - np.log(np.maximum(dfreq, 1))).astype(np.float32)
    bad = dfreq < min_df
    if df_cap is not None:
        bad |= dfreq > df_cap
    idf[bad] = 0
    m.data = idf[m.indices]
    m.eliminate_zeros()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    norms[norms == 0] = 1
    return (sp.diags((1 / norms).astype(np.float32)) @ m).tocsr()


def topk_pairs(A, B, k):
    """Row-wise top-k of A @ B.T as (row, col, score) arrays."""
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=0.0, n_threads=THREADS)
    C = C.tocoo()
    return C.row.astype(np.int32), C.col.astype(np.int32), C.data.astype(np.float32)


def block_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Two-way top-k retrieval inside one country's record pool; returns candidate pairs."""
    word_df = None
    if TRI_MODE == "rare":
        wc = df.select(pl.col("at").str.split(" ").list.unique().alias("w")).explode("w")                .filter(pl.col("w").is_not_null() & (pl.col("w") != "")).group_by("w").len()
        word_df = dict(zip(wc["w"].to_list(), wc["len"].to_list()))
    M = hashed_matrix(docs_for(df, word_df), df_cap=DF_CAP)
    is1 = (df["src"] == 1).to_numpy()
    i1, it = np.where(is1)[0], np.where(~is1)[0]
    A, T = M[i1], M[it]
    fr, fc, fs = topk_pairs(A, T, K_FWD)
    rr, rc, rs = topk_pairs(T, A, K_REV)
    if K_REV_SPARSE > K_REV:
        # targets with (almost) no address can only be resolved by name, and common names have
        # many look-alike Source 1 records: give them a deeper reverse search
        nw = df["at"].fill_null("").str.count_matches(" ").to_numpy() + (df["at"].fill_null("") != "").to_numpy()
        sp_idx = np.where(nw[it] <= 1)[0]
        if len(sp_idx):
            r2, c2, s2 = topk_pairs(T[sp_idx], A, K_REV_SPARSE)
            rr, rc, rs = (np.concatenate([rr, sp_idx[r2].astype(np.int32)]), np.concatenate([rc, c2]),
                          np.concatenate([rs, s2]))
    fwd = pl.DataFrame({"a": fr, "b": fc, "s_fwd": fs})
    rev = pl.DataFrame({"a": rc, "b": rr, "s_rev": rs}).unique(["a", "b"])
    pairs = fwd.join(rev, on=["a", "b"], how="full", coalesce=True)
    pairs = pairs.with_columns(pl.coalesce("s_fwd", "s_rev").alias("s_blk"))
    ids = df["id"]
    a_ids = ids.gather(pl.Series(i1)).to_numpy()
    b_ids = ids.gather(pl.Series(it)).to_numpy()
    return pl.DataFrame({
        "id1": a_ids[pairs["a"].to_numpy()], "id2": b_ids[pairs["b"].to_numpy()],
        "s_blk": pairs["s_blk"], "in_fwd": pairs["s_fwd"].is_not_null(),
        "in_rev": pairs["s_rev"].is_not_null()})


def run(split: str):
    """Generate the candidate set for one split and save cache/{split}_cand.parquet."""
    stage = f"Blocking ({split})"
    allrec = load_norm(split, columns=["id", "country", "src", "nc", "ncp", "at", "num"])
    countries = allrec["country"].unique().sort().to_list()
    out = []
    for ci, c in enumerate(countries):
        t0 = time.time()
        df = allrec.filter(pl.col("country") == c)
        pairs = block_frame(df)
        out.append(pairs)
        n1 = int((df["src"] == 1).sum())
        progress.update(stage, ci + 1, len(countries),
                        msg=f"{split} {c}: {n1:,} S1 vs {df.height - n1:,} S2/S3 -> {pairs.height:,} pairs "
                            f"({time.time() - t0:.0f}s)")
        print(split, c, n1, df.height - n1, pairs.height, f"{time.time() - t0:.0f}s", flush=True)
        del df
    cand = pl.concat(out)
    cand.write_parquet(os.path.join(WORK, f"{split}_cand.parquet"))
    progress.update(stage, len(countries), len(countries), state="done")
    return cand


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    run(ap.parse_args().split)
