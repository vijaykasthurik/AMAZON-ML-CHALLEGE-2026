"""Composite blocking keys.

Character 3-grams find records that share rare spellings, but many businesses
are made only of common words ("Primary Care Group", "Jefferson Street"): each
3-gram is frequent, so they get pruned from the index. What *is* rare is the
combination. This module turns every record into a bag of composite keys:

  name channel
    w:<word>                 each core-name word
    p:<word>|<word>          every unordered pair of core-name words (<= 6 words)
  address channel
    a:<token>                each core-address token
    b:<token>|<token>        adjacent address tokens ("jefferson st")
    n:<number>|<word>        house number followed by the street word
    x:<name word>|<number>   name word crossed with an address number

Keys are hashed with CRC32 into a 2^22 space per channel (collisions are
harmless at this size). The same TF-IDF + top-K engine as the 3-gram index
is then used on these keys.
"""

from __future__ import annotations

import zlib
import numpy as np

KEY_BITS = 22
KEY_DIM = 1 << KEY_BITS          # size of one channel
NAME_KEY_OFF, ADDR_KEY_OFF = 0, KEY_DIM


def _h(s: str) -> int:
    """Deterministic 22-bit hash of a key string."""
    return zlib.crc32(s.encode()) & (KEY_DIM - 1)


def record_keys(nc: str, na: str, ac: str) -> list:
    """All composite key ids of one record (name ids first, then address ids)."""
    words = [w for w in (nc + " " + na).split() if len(w) > 1]
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
    words = seen[:6]
    out = [_h("w:" + w) for w in words]
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = (words[i], words[j]) if words[i] < words[j] else (words[j], words[i])
            out.append(_h("p:" + a + "|" + b))
    toks = ac.split()
    nums = [t for t in toks if t.isdigit()][:3]
    addr = [_h("a:" + t) for t in toks]
    for i in range(len(toks) - 1):
        addr.append(_h("b:" + toks[i] + "|" + toks[i + 1]))
        if toks[i].isdigit() and not toks[i + 1].isdigit():
            addr.append(_h("n:" + toks[i] + "|" + toks[i + 1]))
    for w in words[:4]:
        for n in nums:
            addr.append(_h("x:" + w + "|" + n))
    return out + [KEY_DIM + k for k in addr]


def _chunk_keys(args):
    """Worker: key ids for a chunk of records, as flat (row, key) arrays."""
    start, rows = args
    r, k = [], []
    for i, (nc, na, ac) in enumerate(rows):
        ks = set(record_keys(nc or "", na or "", ac or ""))
        r.extend([start + i] * len(ks))
        k.extend(ks)
    return np.asarray(r, np.int64), np.asarray(k, np.int32)


def keys_csr(nc: list, na: list, ac: list, pool=None, chunk: int = 20_000):
    """CSR pieces (indptr, sorted key ids) of the composite keys of many records.

    ``pool`` is an optional multiprocessing pool reused across calls.
    """
    rows = list(zip(nc, na, ac))
    jobs = [(s, rows[s:s + chunk]) for s in range(0, len(rows), chunk)]
    if pool is not None and len(jobs) > 1:
        res = pool.map(_chunk_keys, jobs, chunksize=1)
    else:
        res = [_chunk_keys(j) for j in jobs]
    r = np.concatenate([x[0] for x in res]) if res else np.zeros(0, np.int64)
    k = np.concatenate([x[1] for x in res]) if res else np.zeros(0, np.int32)
    order = np.lexsort((k, r))
    r, k = r[order], k[order]
    indptr = np.zeros(len(rows) + 1, np.int64)
    np.add.at(indptr, r + 1, 1)
    return np.cumsum(indptr), k
