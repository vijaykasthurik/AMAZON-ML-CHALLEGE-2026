"""Word-alignment similarity between two token strings (numba).

String-level scores such as token_set_ratio blur an important difference:

  * a noisy true match changes words by a letter or two
    ("Wilblims" / "Williams", "Engnernhig" / "Engineering")
  * a look-alike business replaces a whole word
    ("Roofers Services" / "Roofers Securities", "Glyphova" / "Glyphyx")

For every word of one side we find its closest word on the other side
(normalised Levenshtein similarity), then summarise per side:
  min   - how badly the worst word aligns
  mean  - average alignment
  unm   - number of words with no counterpart above 0.75
The same routine applied to the list of address numbers tells a typo in a
house number ("9758" / "9757") from a different house number.

Strings are the normalised ASCII fields, so the edit distance works on bytes.
"""

from __future__ import annotations

import numba as nb
import numpy as np

MAX_TOK = 16      # tokens per side considered (addresses can be long)
UNMATCHED = 0.75  # below this best similarity a word counts as unmatched


@nb.njit(cache=True)
def _lev_sim(d1, s1, e1, d2, s2, e2, row):
    """1 - Levenshtein(a, b) / max(len a, len b) on byte slices."""
    la = e1 - s1
    lb = e2 - s2
    if la == 0 and lb == 0:
        return 1.0
    for j in range(lb + 1):
        row[j] = j
    for i in range(1, la + 1):
        prev = row[0]
        row[0] = i
        ca = d1[s1 + i - 1]
        for j in range(1, lb + 1):
            cur = row[j]
            cost = 0 if ca == d2[s2 + j - 1] else 1
            v = prev + cost
            if row[j] + 1 < v:
                v = row[j] + 1
            if row[j - 1] + 1 < v:
                v = row[j - 1] + 1
            row[j] = v
            prev = cur
    m = la if la > lb else lb
    return 1.0 - row[lb] / m


@nb.njit(cache=True)
def _tokens(data, s, e, alpha_only, starts, ends):
    """Split data[s:e] on spaces; optionally drop all-digit tokens. Returns count."""
    n = 0
    i = s
    while i < e and n < MAX_TOK:
        while i < e and data[i] == 32:
            i += 1
        if i >= e:
            break
        j = i
        has_alpha = False
        while j < e and data[j] != 32:
            if not (48 <= data[j] <= 57):
                has_alpha = True
            j += 1
        if (not alpha_only) or has_alpha:
            starts[n] = i
            ends[n] = j
            n += 1
        i = j
    return n


@nb.njit(parallel=True, cache=True)
def _align(qo, qd, co, cd, alpha_only):
    n = len(qo) - 1
    out = np.full((n, 6), np.nan, np.float32)
    for i in nb.prange(n):
        qs = np.empty(MAX_TOK, np.int64)
        qe = np.empty(MAX_TOK, np.int64)
        cs = np.empty(MAX_TOK, np.int64)
        ce = np.empty(MAX_TOK, np.int64)
        row = np.empty(256, np.int64)
        nq = _tokens(qd, qo[i], qo[i + 1], alpha_only, qs, qe)
        nc = _tokens(cd, co[i], co[i + 1], alpha_only, cs, ce)
        if nq == 0 or nc == 0:
            continue
        sim = np.zeros((nq, nc), np.float32)
        for a in range(nq):
            for b in range(nc):
                if qe[a] - qs[a] < 255 and ce[b] - cs[b] < 255:
                    sim[a, b] = _lev_sim(qd, qs[a], qe[a], cd, cs[b], ce[b], row)
        # q side
        mn = 1.0
        tot = 0.0
        unm = 0
        for a in range(nq):
            best = 0.0
            for b in range(nc):
                if sim[a, b] > best:
                    best = sim[a, b]
            mn = min(mn, best)
            tot += best
            if best < UNMATCHED:
                unm += 1
        out[i, 0] = mn
        out[i, 1] = tot / nq
        out[i, 2] = unm
        # c side
        mn = 1.0
        tot = 0.0
        unm = 0
        for b in range(nc):
            best = 0.0
            for a in range(nq):
                if sim[a, b] > best:
                    best = sim[a, b]
            mn = min(mn, best)
            tot += best
            if best < UNMATCHED:
                unm += 1
        out[i, 3] = mn
        out[i, 4] = tot / nc
        out[i, 5] = unm
    return out


MAX_NUM = 4  # numbers per side used for the arithmetic gap


@nb.njit(cache=True)
def _ndigits(v):
    """Number of decimal digits of a non-negative integer."""
    c = 1
    while v >= 10:
        v //= 10
        c += 1
    return c


@nb.njit(cache=True)
def _parse_nums(data, s, e, out):
    """Parse up to MAX_NUM space-separated integers from data[s:e]; returns count."""
    n = 0
    i = s
    while i < e and n < MAX_NUM:
        while i < e and data[i] == 32:
            i += 1
        if i >= e:
            break
        v = 0
        k = 0
        while i < e and data[i] != 32:
            if k < 15 and 48 <= data[i] <= 57:
                v = v * 10 + (data[i] - 48)
                k += 1
            i += 1
        if k > 0:
            out[n] = v
            n += 1
    return n


@nb.njit(parallel=True, cache=True)
def _num_gaps(qo, qd, co, cd):
    n = len(qo) - 1
    out = np.full((n, 8), np.nan, np.float32)
    for i in nb.prange(n):
        a = np.empty(MAX_NUM, np.int64)
        b = np.empty(MAX_NUM, np.int64)
        na = _parse_nums(qd, qo[i], qo[i + 1], a)
        nb_ = _parse_nums(cd, co[i], co[i + 1], b)
        if na == 0 or nb_ == 0:
            continue
        # signed relations: query number minus S1 number
        out[i, 5] = a[0] - b[0]
        # numbers present on one side only; when exactly one differs on each side
        # (an "address copy"), keep its signed difference
        qo_n = 0
        qo_v = 0
        for x in range(na):
            found = False
            for y in range(nb_):
                if a[x] == b[y]:
                    found = True
                    break
            if not found:
                qo_n += 1
                qo_v = a[x]
        co_n = 0
        co_v = 0
        for y in range(nb_):
            found = False
            for x in range(na):
                if a[x] == b[y]:
                    found = True
                    break
            if not found:
                co_n += 1
                co_v = b[y]
        if qo_n == 1 and co_n == 1:
            out[i, 6] = qo_v - co_v
        elif qo_n == 0 and co_n == 0:
            out[i, 6] = 0.0
        out[i, 7] = qo_n + co_n
        d0 = abs(a[0] - b[0])
        m0 = max(a[0], b[0])
        out[i, 0] = d0                                   # first number vs first number
        out[i, 1] = d0 / m0 if m0 > 0 else 0.0           # relative gap
        best = -1
        bestrel = 1.0
        for x in range(na):
            for y in range(nb_):
                d = abs(a[x] - b[y])
                if best < 0 or d < best:
                    best = d
                m = max(a[x], b[y])
                rel = d / m if m > 0 else 0.0
                if rel < bestrel:
                    bestrel = rel
        out[i, 2] = best                                  # closest pair of numbers
        out[i, 3] = bestrel
        # same number of digits in the first numbers (a shift keeps the length,
        # a dropped / added digit does not)
        out[i, 4] = 1.0 if _ndigits(a[0]) == _ndigits(b[0]) else 0.0
    return out


def number_gaps(q_series, c_series) -> dict:
    """Arithmetic house-number features for row-aligned ``anum`` series.

    Text similarity sees "165" / "167" as a one-character typo. Look-alike
    businesses in this data are address copies with the house number shifted
    *up*: on labelled validation pairs where exactly one number differs, a
    signed difference (query minus S1) of +3..+21 was a true match 1 time in
    19, a negative difference 9 times in 9. So the signed gap is kept as well
    as the absolute one:
      hn_gap / hn_relgap         first number vs first number (absolute / relative)
      num_gap_min / num_relgap_min  closest pair of numbers
      hn_samelen                 same digit count (a shift keeps it, a dropped digit does not)
      hn_sd                      signed first-number difference
      num_sd                     signed difference of the single differing number (0 = same set)
      num_ndiff                  how many numbers appear on one side only
    """
    from blocking import _arrow_buffers
    qo, qd = _arrow_buffers(q_series)
    co, cd = _arrow_buffers(c_series)
    res = _num_gaps(qo, qd, co, cd)
    names = ["hn_gap", "hn_relgap", "num_gap_min", "num_relgap_min", "hn_samelen",
             "hn_sd", "num_sd", "num_ndiff"]
    return {nm: res[:, k] for k, nm in enumerate(names)}


FREQ_BITS = 22
RARE_FLOOR = float(np.log(5 / 250_000))  # reachable by every country table on train and test
FREQ_MASK = (1 << FREQ_BITS) - 1


@nb.njit(cache=True)
def _fnv(data, s, e):
    """64-bit FNV-1a hash of data[s:e], folded to the frequency-table size."""
    h = np.uint64(14695981039346656037)
    for i in range(s, e):
        h = (h ^ np.uint64(data[i])) * np.uint64(1099511628211)
    return np.int64(h & np.uint64(FREQ_MASK))


@nb.njit(cache=True)
def _count_tokens(off, data, table):
    """Document frequency of every word (hashed) over a list of name strings."""
    starts = np.empty(MAX_TOK, np.int64)
    ends = np.empty(MAX_TOK, np.int64)
    for i in range(len(off) - 1):
        n = _tokens(data, off[i], off[i + 1], False, starts, ends)
        for k in range(n):
            dup = False
            for j in range(k):  # count a word once per record
                if ends[j] - starts[j] == ends[k] - starts[k]:
                    same = True
                    for t in range(ends[k] - starts[k]):
                        if data[starts[j] + t] != data[starts[k] + t]:
                            same = False
                            break
                    if same:
                        dup = True
                        break
            if not dup:
                table[_fnv(data, starts[k], ends[k])] += 1


def token_freq_table(names) -> np.ndarray:
    """log relative document frequency of each name word among ``names`` (hashed table).

    Built from the Source 1 core names of the split being scored, so it is
    label free and means the same thing on train and test ("how common is this
    word among business names here").
    """
    from blocking import _arrow_buffers
    off, data = _arrow_buffers(names)
    counts = np.zeros(1 << FREQ_BITS, np.int64)
    _count_tokens(off, data, counts)
    n = max(1, len(off) - 1)
    # clip the rare end: below ~1 in 50,000 names the value would depend on how
    # many S1 records a country has (train US 1.3M vs test France 259k), which
    # would make the feature a hidden country / split indicator
    return np.maximum(np.log((counts + 1.0) / n), RARE_FLOOR).astype(np.float32)


@nb.njit(parallel=True, cache=True)
def _unmatched_freq(qo, qd, co, cd, tables, tidx):
    n = len(qo) - 1
    out = np.full((n, 5), np.nan, np.float32)
    for i in nb.prange(n):
        table = tables[tidx[i]]
        qs = np.empty(MAX_TOK, np.int64)
        qe = np.empty(MAX_TOK, np.int64)
        cs = np.empty(MAX_TOK, np.int64)
        ce = np.empty(MAX_TOK, np.int64)
        row = np.empty(256, np.int64)
        nq = _tokens(qd, qo[i], qo[i + 1], False, qs, qe)
        nc = _tokens(cd, co[i], co[i + 1], False, cs, ce)
        if nq == 0 or nc == 0:
            continue
        sim = np.zeros((nq, nc), np.float32)
        for a in range(nq):
            for b in range(nc):
                if qe[a] - qs[a] < 255 and ce[b] - cs[b] < 255:
                    sim[a, b] = _lev_sim(qd, qs[a], qe[a], cd, cs[b], ce[b], row)
        qmax = -1e9
        qmin = 1e9
        cmax = -1e9
        cmin = 1e9
        rare = 1e9
        for a in range(nq):
            f = table[_fnv(qd, qs[a], qe[a])]
            best = 0.0
            for b in range(nc):
                if sim[a, b] > best:
                    best = sim[a, b]
            if best < UNMATCHED:
                qmax = max(qmax, f)
                qmin = min(qmin, f)
        for b in range(nc):
            f = table[_fnv(cd, cs[b], ce[b])]
            rare = min(rare, f)
            best = 0.0
            for a in range(nq):
                if sim[a, b] > best:
                    best = sim[a, b]
            if best < UNMATCHED:
                cmax = max(cmax, f)
                cmin = min(cmin, f)
        if qmax > -1e9:
            out[i, 0] = qmax
            out[i, 1] = qmin
        if cmax > -1e9:
            out[i, 2] = cmax
            out[i, 3] = cmin
        out[i, 4] = rare
    return out


def unmatched_word_freq(q_series, c_series, tables, tidx) -> dict:
    """How common are the words that fail to align? (generic filler vs real word swap)

    uf_q_max / uf_q_min : log frequency of the most / least common unmatched query word
    uf_c_max / uf_c_min : same for the S1 name
    uf_c_rarest         : log frequency of the rarest S1 name word (how specific the name is)
    NaN where a side has no unmatched word. ``tables[tidx[i]]`` is the frequency
    table used for row i (one table per country, so French type words count as
    generic within France just like "services" within the US).
    """
    from blocking import _arrow_buffers
    qo, qd = _arrow_buffers(q_series)
    co, cd = _arrow_buffers(c_series)
    res = _unmatched_freq(qo, qd, co, cd, tables, tidx.astype(np.int64))
    names = ["uf_q_max", "uf_q_min", "uf_c_max", "uf_c_min", "uf_c_rarest"]
    return {nm: res[:, k] for k, nm in enumerate(names)}


def word_align(q_series, c_series, alpha_only: bool, prefix: str) -> dict:
    """Six alignment features for row-aligned polars string series."""
    from blocking import _arrow_buffers
    qo, qd = _arrow_buffers(q_series)
    co, cd = _arrow_buffers(c_series)
    res = _align(qo, qd, co, cd, alpha_only)
    names = ["q_min", "q_mean", "q_unm", "c_min", "c_mean", "c_unm"]
    return {f"{prefix}_{nm}": res[:, k] for k, nm in enumerate(names)}
