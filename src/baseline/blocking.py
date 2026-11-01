"""Candidate generation (blocking) with a numba sparse top-K engine.

Every record is represented by two TF-IDF vectors of character 3-grams:

  * name channel    - 3-grams of the compact core name (plus the alias, if any)
  * address channel - 3-grams of the core address (word boundaries kept)

Because normalised strings only use the alphabet [space a-z 0-9 _], a 3-gram
id is computed arithmetically (base 38), so no vocabulary has to be built or
stored. IDF weights come from Source 1 of the same country, and each channel is
L2-normalised, so the dot product of two channel vectors is a cosine similarity.

Retrieval runs *from* every Source 2 / Source 3 record *into* the Source 1
records of the same country (each S2/S3 record belongs to at most one S1
entity, so this is the natural direction). A single parallel pass accumulates
name and address scores for every S1 record sharing a (non-stop) 3-gram with
the query, then keeps three top-K lists: best by name, best by address and
best by name+address. Their union is the candidate set.

3-grams that occur in more than ``max_df`` Source 1 records of a country are
dropped from the inverted index. They carry almost no identity information and
dominate the cost of the sparse product.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numba as nb
import numpy as np
import polars as pl
import pyarrow as pa
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from keys import KEY_DIM, keys_csr  # noqa: E402

BASE = 38
GRAM_DIM = BASE ** 3          # size of one channel's feature space
NAME_OFF, ADDR_OFF = 0, GRAM_DIM


# --------------------------------------------------------------------------
# String -> 3-gram CSR (numba)
# --------------------------------------------------------------------------

def _arrow_buffers(values: pl.Series):
    """Return (offsets int64, data uint8) of a polars Utf8 series (nulls -> '')."""
    arr = values.fill_null("").to_arrow()
    arr = pa.concat_arrays(arr.chunks) if isinstance(arr, pa.ChunkedArray) else arr
    arr = arr.cast(pa.large_string())
    bufs = arr.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64)[arr.offset: arr.offset + len(arr) + 1]
    data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
    return offsets.copy(), data


@nb.njit(cache=True)
def _code(b):
    """Map an ASCII byte of a normalised string to 0..37."""
    if b == 32:
        return 0
    if 97 <= b <= 122:
        return b - 96
    if 48 <= b <= 57:
        return b - 48 + 27
    return 37


@nb.njit(parallel=True, cache=True)
def _grams_csr(offsets, data, offset_feat):
    """Build a CSR (row -> sorted unique 3-gram ids, counts) for padded strings."""
    n = len(offsets) - 1
    ub = np.zeros(n + 1, np.int64)
    for i in range(n):
        L = offsets[i + 1] - offsets[i]
        ub[i + 1] = ub[i] + (L if L > 0 else 0)  # padded length L+2 -> L grams
    ids = np.empty(ub[n], np.int32)
    cnt = np.empty(ub[n], np.float32)
    nnz = np.zeros(n, np.int64)
    for i in nb.prange(n):
        s, e = offsets[i], offsets[i + 1]
        L = e - s
        if L <= 0:
            continue
        tmp = np.empty(L, np.int32)
        for k in range(L):
            c0 = _code(data[s + k - 1]) if k >= 1 else 0
            c1 = _code(data[s + k])
            c2 = _code(data[s + k + 1]) if k + 1 < L else 0
            tmp[k] = c0 * BASE * BASE + c1 * BASE + c2 + offset_feat
        tmp.sort()
        w = ub[i]
        prev = -1
        m = 0
        for k in range(L):
            if tmp[k] != prev:
                ids[w + m] = tmp[k]
                cnt[w + m] = 1.0
                prev = tmp[k]
                m += 1
            else:
                cnt[w + m - 1] += 1.0
        nnz[i] = m
    indptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + nnz[i]
    out_ids = np.empty(indptr[n], np.int32)
    out_cnt = np.empty(indptr[n], np.float32)
    for i in nb.prange(n):
        a, b = ub[i], indptr[i]
        for k in range(nnz[i]):
            out_ids[b + k] = ids[a + k]
            out_cnt[b + k] = cnt[a + k]
    return indptr, out_ids, out_cnt


def grams(values: pl.Series, offset_feat: int):
    """3-gram CSR pieces for a Utf8 series."""
    off, data = _arrow_buffers(values)
    return _grams_csr(off, data, offset_feat)


@nb.njit(parallel=True, cache=True)
def _tfidf_rows(indptr, ids, cnt, idf):
    """In-place sublinear TF * IDF and L2 row normalisation."""
    n = len(indptr) - 1
    for i in nb.prange(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            w = (1.0 + math.log(cnt[p])) * idf[ids[p]]
            cnt[p] = w
            s += w * w
        if s > 0:
            inv = 1.0 / math.sqrt(s)
            for p in range(indptr[i], indptr[i + 1]):
                cnt[p] *= inv


@nb.njit(parallel=True, cache=True)
def _idf_norm_channels(indptr, ids, idf, split):
    """Binary-TF x IDF weights, L2-normalised separately in each channel."""
    n = len(indptr) - 1
    w = np.empty(len(ids), np.float32)
    for i in nb.prange(n):
        s0 = 0.0
        s1 = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            v = idf[ids[p]]
            w[p] = v
            if ids[p] < split:
                s0 += v * v
            else:
                s1 += v * v
        i0 = 1.0 / math.sqrt(s0) if s0 > 0 else 0.0
        i1 = 1.0 / math.sqrt(s1) if s1 > 0 else 0.0
        for p in range(indptr[i], indptr[i + 1]):
            w[p] *= i0 if ids[p] < split else i1
    return w


def channel_matrix(values: pl.Series, offset_feat: int, idf: np.ndarray):
    """TF-IDF + L2 CSR matrix (scipy) of one text channel."""
    indptr, ids, cnt = grams(values, offset_feat)
    _tfidf_rows(indptr, ids, cnt, idf)
    return sp.csr_matrix((cnt, ids, indptr), shape=(len(indptr) - 1, 2 * GRAM_DIM))


def doc_freq(values: pl.Series, offset_feat: int) -> np.ndarray:
    """Document frequency of every 3-gram id over a series."""
    _, ids, _ = grams(values, offset_feat)
    return np.bincount(ids, minlength=2 * GRAM_DIM).astype(np.float64)


# --------------------------------------------------------------------------
# Sparse top-K search (numba)
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _push(topi, tops, j, s):
    """Insert (j, s) into a descending top-K list if it qualifies."""
    K = len(topi)
    if s <= tops[K - 1]:
        return
    k = K - 1
    while k > 0 and tops[k - 1] < s:
        tops[k] = tops[k - 1]
        topi[k] = topi[k - 1]
        k -= 1
    tops[k] = s
    topi[k] = j


@nb.njit(parallel=True, cache=True)
def _search(q_indptr, q_ids, q_w, x_indptr, x_ids, x_w, n_x, K, w_name, w_addr, n_threads, split):
    """For every query row return top-K ids by name, by address and by combined.

    Returns idx[nq, 3, K] (int32, -1 = empty) and the name / address scores of
    each returned candidate: sn[nq, 3, K], sa[nq, 3, K].
    """
    nq = len(q_indptr) - 1
    idx = np.full((nq, 3, K), -1, np.int32)
    sn = np.zeros((nq, 3, K), np.float32)
    sa = np.zeros((nq, 3, K), np.float32)
    chunk = (nq + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        acc_n = np.zeros(n_x, np.float32)
        acc_a = np.zeros(n_x, np.float32)
        seen = np.zeros(n_x, np.uint8)
        touched = np.empty(n_x, np.int32)
        topi = np.empty(K, np.int32)
        tops = np.empty(K, np.float32)
        lo = t * chunk
        hi = min(nq, lo + chunk)
        for i in range(lo, hi):
            nt = 0
            for p in range(q_indptr[i], q_indptr[i + 1]):
                f = q_ids[p]
                w = q_w[p]
                is_addr = f >= split
                for pp in range(x_indptr[f], x_indptr[f + 1]):
                    j = x_ids[pp]
                    if seen[j] == 0:
                        seen[j] = 1
                        touched[nt] = j
                        nt += 1
                    if is_addr:
                        acc_a[j] += w * x_w[pp]
                    else:
                        acc_n[j] += w * x_w[pp]
            for m in range(3):
                topi[:] = -1
                tops[:] = -1.0
                for u in range(nt):
                    j = touched[u]
                    if m == 0:
                        s = acc_n[j]
                    elif m == 1:
                        s = acc_a[j]
                    else:
                        s = w_name * acc_n[j] + w_addr * acc_a[j]
                    _push(topi, tops, j, s)
                for k in range(K):
                    j = topi[k]
                    if j >= 0 and tops[k] > 0:
                        idx[i, m, k] = j
                        sn[i, m, k] = acc_n[j]
                        sa[i, m, k] = acc_a[j]
            for u in range(nt):
                j = touched[u]
                seen[j] = 0
                acc_n[j] = 0.0
                acc_a[j] = 0.0
    return idx, sn, sa


@nb.njit(parallel=True, cache=True)
def _limit_probe(indptr, ids, w, keep, P_name, P_addr, split):
    """Per row keep only the P highest-weight *indexed* grams of each channel.

    These are the rarest grams of the record, so they are the ones that can
    locate its match; frequent grams would only add cost.
    """
    n = len(indptr) - 1
    cnt = np.zeros(n, np.int64)
    for i in nb.prange(n):
        a = 0
        b = 0
        for p in range(indptr[i], indptr[i + 1]):
            if keep[ids[p]]:
                if ids[p] < split:
                    a += 1
                else:
                    b += 1
        cnt[i] = min(a, P_name) + min(b, P_addr)
    out_ptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        out_ptr[i + 1] = out_ptr[i] + cnt[i]
    out_ids = np.empty(out_ptr[n], np.int32)
    out_w = np.empty(out_ptr[n], np.float32)
    for i in nb.prange(n):
        o = out_ptr[i]
        for ch in range(2):
            P = P_name if ch == 0 else P_addr
            m = 0
            for p in range(indptr[i], indptr[i + 1]):
                if keep[ids[p]] and ((ids[p] < split) == (ch == 0)):
                    m += 1
            if m == 0:
                continue
            loc = np.empty(m, np.int64)
            m = 0
            for p in range(indptr[i], indptr[i + 1]):
                if keep[ids[p]] and ((ids[p] < split) == (ch == 0)):
                    loc[m] = p
                    m += 1
            ws = np.empty(m, np.float32)
            for k in range(m):
                ws[k] = w[loc[k]]
            order = np.argsort(-ws)
            for k in range(min(m, P)):
                out_ids[o] = ids[loc[order[k]]]
                out_w[o] = w[loc[order[k]]]
                o += 1
    return out_ptr, out_ids, out_w


@nb.njit(parallel=True, cache=True)
def _exact_pairs(q_indptr, q_ids, q_w, x_indptr, x_ids, x_w, pq, pc, split):
    """Exact name / address cosine for (query row, S1 row) pairs (sorted CSRs)."""
    n = len(pq)
    sn = np.zeros(n, np.float32)
    sa = np.zeros(n, np.float32)
    for k in nb.prange(n):
        a, ae = q_indptr[pq[k]], q_indptr[pq[k] + 1]
        b, be = x_indptr[pc[k]], x_indptr[pc[k] + 1]
        dn = 0.0
        da = 0.0
        while a < ae and b < be:
            ia = q_ids[a]
            ib = x_ids[b]
            if ia == ib:
                if ia < split:
                    dn += q_w[a] * x_w[b]
                else:
                    da += q_w[a] * x_w[b]
                a += 1
                b += 1
            elif ia < ib:
                a += 1
            else:
                b += 1
        sn[k] = dn
        sa[k] = da
    return sn, sa


# --------------------------------------------------------------------------
# High level
# --------------------------------------------------------------------------

def name_text(df: pl.DataFrame) -> pl.Series:
    """Text of the name channel: compact core name plus compact alias."""
    return (pl.when(pl.col("na") != "")
            .then(pl.col("nk") + pl.lit(" ") + pl.col("na").str.replace_all(" ", ""))
            .otherwise(pl.col("nk"))).alias("name_text")


def _csr(values_name: pl.Series, values_addr: pl.Series, idf: np.ndarray):
    """Sorted CSR (indptr, ids, weights) of name + address channels."""
    X = channel_matrix(values_name, NAME_OFF, idf) + channel_matrix(values_addr, ADDR_OFF, idf)
    X.sort_indices()
    return X.indptr.astype(np.int64), X.indices.astype(np.int32), X.data.astype(np.float32)


def _inverted(indptr, ids, w, n_rows, dim, keep):
    """Feature -> rows inverted lists (CSC pieces) restricted to ``keep`` features."""
    X = sp.csr_matrix((w, ids, indptr), shape=(n_rows, dim))
    X = (X @ sp.diags(keep.astype(np.float32))).tocsc()
    X.eliminate_zeros()
    X.sort_indices()
    return X.indptr.astype(np.int64), X.indices.astype(np.int32), X.data.astype(np.float32)


class CountryIndex:
    """Two inverted indexes over the Source 1 records of one country.

    * gram index - character 3-grams of name / address (typo tolerant);
      grams in more than ``max_df_frac`` of the records are not indexed.
    * key index  - composite word keys (see keys.py) that stay selective
      even when every single word is common; keys in more than ``max_df_key``
      records are not indexed.
    """

    def __init__(self, s1: pl.DataFrame, max_df_frac: float = 0.002, min_max_df: int = 300,
                 max_df_key: int = 300, pool=None):
        t0 = time.time()
        self.n = s1.height
        self.pool = pool
        names = s1.select(name_text(s1))["name_text"]
        addrs = s1["ac"]
        # ---- 3-gram index
        df = doc_freq(names, NAME_OFF) + doc_freq(addrs, ADDR_OFF)
        self.idf = (np.log((self.n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        max_df = max(min_max_df, int(max_df_frac * self.n))
        self.keep = (df > 0) & (df <= max_df)
        self.r_indptr, self.r_ids, self.r_w = _csr(names, addrs, self.idf)
        self.x_indptr, self.x_ids, self.x_w = _inverted(
            self.r_indptr, self.r_ids, self.r_w, self.n, 2 * GRAM_DIM, self.keep)
        # ---- composite key index
        kp, kids = keys_csr(s1["nc"].to_list(), s1["na"].to_list(), s1["ac"].to_list(), pool)
        kdf = np.bincount(kids, minlength=2 * KEY_DIM).astype(np.float64)
        self.kidf = (np.log((self.n + 1.0) / (kdf + 1.0)) + 1.0).astype(np.float32)
        self.kkeep = (kdf > 0) & (kdf <= max_df_key)
        self.k_indptr, self.k_ids = kp, kids
        self.k_w = _idf_norm_channels(kp, kids, self.kidf, KEY_DIM)
        self.kx_indptr, self.kx_ids, self.kx_w = _inverted(
            self.k_indptr, self.k_ids, self.k_w, self.n, 2 * KEY_DIM, self.kkeep)
        print(f"  index: {self.n} S1 rows | grams: max_df {max_df}, {len(self.x_ids)} postings "
              f"| keys: {int(self.kkeep.sum())} active, {len(self.kx_ids)} postings "
              f"({time.time() - t0:.0f}s)", flush=True)

    def search(self, q: pl.DataFrame, q_rows: np.ndarray, s1_rows: np.ndarray, K: int = 10,
               P_name: int = 12, P_addr: int = 16, w_name: float = 1.0, w_addr: float = 1.0,
               batch: int = 300_000) -> pl.DataFrame:
        """Candidate pairs for every query row, with exact scores and ranks.

        Six top-K lists are gathered per query (gram name / address / combined
        and key name / address / combined). Every pair in their union gets exact
        gram cosines (sn, sa) and key cosines (kn, ka), and its rank within the
        query's union for each of the six scores (rk_*; 0 = best).
        """
        parts = []
        nt = nb.get_num_threads()
        for b in range(0, q.height, batch):
            qb = q.slice(b, batch)
            qi, qids, qw = _csr(qb.select(name_text(qb))["name_text"], qb["ac"], self.idf)
            pi, pids, pw = _limit_probe(qi, qids, qw, self.keep, P_name, P_addr, GRAM_DIM)
            idx_g, _, _ = _search(pi, pids, pw, self.x_indptr, self.x_ids, self.x_w,
                                  self.n, K, w_name, w_addr, nt, GRAM_DIM)
            ki, kids = keys_csr(qb["nc"].to_list(), qb["na"].to_list(), qb["ac"].to_list(), self.pool)
            kw = _idf_norm_channels(ki, kids, self.kidf, KEY_DIM)
            pki, pkids, pkw = _limit_probe(ki, kids, kw, self.kkeep, 64, 64, KEY_DIM)
            idx_k, _, _ = _search(pki, pkids, pkw, self.kx_indptr, self.kx_ids, self.kx_w,
                                  self.n, K, w_name, w_addr, nt, KEY_DIM)
            nq = idx_g.shape[0]
            flat = np.concatenate([idx_g.reshape(nq, -1), idx_k.reshape(nq, -1)], axis=1)
            lq = np.repeat(np.arange(nq, dtype=np.int64), flat.shape[1])
            lc = flat.reshape(-1).astype(np.int64)
            m = lc >= 0
            pairs = np.unique(lq[m] * self.n + lc[m])
            lq, lc = pairs // self.n, pairs % self.n
            sn, sa = _exact_pairs(qi, qids, qw, self.r_indptr, self.r_ids, self.r_w, lq, lc, GRAM_DIM)
            kn, ka = _exact_pairs(ki, kids, kw, self.k_indptr, self.k_ids, self.k_w, lq, lc, KEY_DIM)
            fr = pl.DataFrame({"q": q_rows[b:b + batch][lq].astype(np.int32),
                               "c": s1_rows[lc].astype(np.int32),
                               "sn": sn, "sa": sa, "kn": kn, "ka": ka})
            rk = lambda e, nm: e.rank("ordinal", descending=True).over("q").cast(pl.Int16).sub(1).alias(nm)
            fr = fr.with_columns(
                rk(pl.col("sn"), "rk_n"), rk(pl.col("sa"), "rk_a"),
                rk(w_name * pl.col("sn") + w_addr * pl.col("sa"), "rk_c"),
                rk(pl.col("kn"), "rk_kn"), rk(pl.col("ka"), "rk_ka"),
                rk(pl.col("kn") + pl.col("ka"), "rk_kc"),
            )
            parts.append(fr)
        return pl.concat(parts)
