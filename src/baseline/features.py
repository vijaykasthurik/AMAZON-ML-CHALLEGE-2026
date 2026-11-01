"""Pair features for the matching model.

For every candidate pair (q = Source 2/3 record, c = Source 1 record) we build
country-agnostic features only. There is deliberately no country feature and
no "state differs" flag: the test set contains France, where one source writes
the region and another the department, so such a flag learned on US/India
would wrongly veto French matches.

Feature groups
  retrieval  : name / address TF-IDF cosines and ranks from the blocking stage
  name       : rapidfuzz similarities of the core / compact / full names,
               alias similarity, legal-form agreement, token counts
  address    : rapidfuzz similarities of the core addresses and number overlap
  frequency  : how common the S1 name / address is (generic names need the
               address to decide)
  context    : how this pair compares with the other candidates of the same
               query and of the same S1 entity (label free)
  source     : whether the query comes from Source 3, and whether its name was
               written in an Indic script
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from align import number_gaps, token_freq_table, unmatched_word_freq, word_align
from preprocess import ADDR_ABBR, LEGAL_CANON

WORKERS = -1
USE_ALIGN = True       # word-alignment features (align.py)
USE_LEGAL_DIFF = True  # signed per-token legal-form differences
USE_ACRONYM = True     # name-is-initials-of-the-other features
# legal tokens common enough in training to get their own difference flag
LEGAL_FLAGS = ["pvt", "ltd", "inc", "llc", "corp", "co", "llp", "pc", "lp", "pllc"]
USE_NUMGAP = True      # arithmetic house-number gaps + legal disjoint / subset
USE_WORDFREQ = True    # how common the non-aligning name words are
# "... id 30420" registration ids and 7+ digit phone numbers appended to names
ID_WORDS = r"\s*\b(?:id \d{3,}|\d{7,})\b"
# words that the noise process inserts into true matches (checked on labelled
# validation pairs: an inserted Center/Services/Partners was a true match 34 of 36 times)
NOISE_ADD = ["center", "centre", "service", "services", "partners"]
NOISE_AFTER_LEGAL = (r"\b(?:pvt|ltd|llc|inc|corp|co|llp|lp|pc|pllc|plc)\b(?: \S+)? "
                     r"(?:center|centre|service|services|partners)$")
# legal forms grouped into families, so a change inside a family (SAS -> SASU,
# Inc -> Corp) is told apart from a change of company type (SARL -> SA, Ltd -> LLP)
LEGAL_FAMILY = {"pvt": "a", "ltd": "a", "plc": "a", "inc": "b", "corp": "b", "llc": "c",
                "pllc": "c", "llp": "d", "lp": "d", "sas": "e", "sasu": "e", "sa": "e",
                "sarl": "f", "eurl": "f", "sci": "g", "snc": "h", "co": "i", "ets": "j"}
# leading honorifics: "M/s" becomes the two tokens "m s" after normalisation
HONORIFIC = r"^(?:(?:m s|messrs|dr|shri|sri|smt|kum)\s+)+"
TOKEN_FREQ = None      # word-frequency tables of the current split's S1 names
COUNTRY_TABLE = {}     # country -> row of TOKEN_FREQ (0 = all countries pooled)


def set_token_freq(s1_core_names: pl.Series, s1_country: pl.Series):
    """Word-frequency tables of the split's S1 core names: pooled + one per country.

    Frequencies are relative to the number of S1 records of that country, so
    they mean the same thing on train and test and for a country seen only in
    test ("how common is this word among this country's business names").
    """
    global TOKEN_FREQ, COUNTRY_TABLE
    countries = sorted(s1_country.unique().to_list())
    tabs = [token_freq_table(s1_core_names)]
    for c in countries:
        tabs.append(token_freq_table(s1_core_names.filter(s1_country == c)))
    TOKEN_FREQ = np.stack(tabs)
    COUNTRY_TABLE = {c: k + 1 for k, c in enumerate(countries)}


def clean_names(R: pl.DataFrame) -> pl.DataFrame:
    """Drop registration ids and leading honorifics from nc / nn; rebuild nk from nc."""
    nc = pl.col("nc").str.replace_all(ID_WORDS, "").str.strip_chars()
    nc = pl.when(nc.str.replace(HONORIFIC, "") != "").then(nc.str.replace(HONORIFIC, "")).otherwise(nc)
    nn = pl.col("nn").str.replace_all(ID_WORDS, "").str.strip_chars()
    nn = pl.when(nn.str.replace(HONORIFIC, "") != "").then(nn.str.replace(HONORIFIC, "")).otherwise(nn)
    R = R.with_columns(nc.alias("nc"), nn.alias("nn"))
    return R.with_columns(pl.col("nc").str.replace_all(" ", "").alias("nk"))


def clean_address(R: pl.DataFrame) -> pl.DataFrame:
    """Canonicalise French address artefacts the normaliser missed.

    "N° 12" leaves a stray "n" before the number, "Q." is quai, and "Appt" /
    "Bat" (apartment / building) are unit words like the filler already removed.
    """
    ac = (pl.col("ac").str.replace_all(r"\bn (\d)", "$1").str.replace_all(r"\bq\b", "quai")
          .str.replace_all(r"\b(?:appt|apprt|app|bat)\b", "").str.replace_all(r"\s+", " ")
          .str.strip_chars())
    return R.with_columns(ac.alias("ac"))


def _cp(scorer, a, b):
    """Row-aligned similarity of two string lists, as float32 in [0, 1]."""
    return (process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32)
            / (1.0 if scorer is JaroWinkler.normalized_similarity else 100.0)).astype(np.float32)


def record_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record helper columns used by the pair features."""
    return df.select(
        "nn", "nc", "nk", "nl", "na", "ac", "anum", "country",
        pl.col("business_name").str.contains(r"[ऀ-ൿ]").fill_null(False).alias("native"),
        (pl.col("ac") == "").alias("addr_empty"),
        pl.col("nc").str.count_matches(" ").add(1).cast(pl.Int16).alias("n_tok"),
    )


def frequency_table(s1: pl.DataFrame) -> pl.DataFrame:
    """Per S1 record: how many S1 records of the same country share its name / address."""
    return s1.select(
        pl.len().over(["country", "nc"]).cast(pl.Float32).alias("f_c_name"),
        pl.len().over(["country", "ac"]).cast(pl.Float32).alias("f_c_addr"),
        pl.len().over(["country", "nc", "ac"]).cast(pl.Float32).alias("f_c_both"),
    )


def add_scores(pairs: pl.DataFrame) -> pl.DataFrame:
    """Combined retrieval scores: 3-gram (comb) and composite-key (kcomb)."""
    return pairs.with_columns((pl.col("sn") + pl.col("sa")).alias("comb"),
                              (pl.col("kn") + pl.col("ka")).alias("kcomb"))


def c_context(pairs: pl.DataFrame):
    """Source-1 side context, computed once on the *full* candidate table.

    Returns (pairs with c_rank / c_krank added, per-S1 statistics table).
    """
    pairs = pairs.with_columns(
        pl.col("comb").rank("ordinal", descending=True).over("c").cast(pl.Float32).alias("c_rank"),
        pl.col("kcomb").rank("ordinal", descending=True).over("c").cast(pl.Float32).alias("c_krank"),
    )
    stats = pairs.group_by("c").agg(
        pl.len().cast(pl.Float32).alias("c_ncand"),
        pl.col("comb").max().alias("c_max"),
        pl.col("kcomb").max().alias("c_kmax"),
    )
    return pairs, stats


def q_context(sub: pl.DataFrame, c_stats: pl.DataFrame) -> pl.DataFrame:
    """Query side context (needs every candidate of each query in ``sub``)."""
    # maintain_order: callers align these rows with labels / folds by position
    sub = sub.join(c_stats, on="c", how="left", maintain_order="left").with_columns(
        (pl.col("c_max") - pl.col("comb")).alias("c_gap"),
        (pl.col("c_kmax") - pl.col("kcomb")).alias("c_kgap"),
        pl.len().over("q").cast(pl.Float32).alias("q_ncand"),
        (pl.col("comb").max().over("q") - pl.col("comb")).alias("q_gap"),
        (pl.col("kcomb").max().over("q") - pl.col("kcomb")).alias("q_kgap"),
        pl.col("comb").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("q_rank"),
        pl.col("kcomb").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("q_krank"),
        (pl.col("sn").max().over("q") - pl.col("sn")).alias("q_gap_n"),
        (pl.col("sa").max().over("q") - pl.col("sa")).alias("q_gap_a"),
    )
    return sub.with_columns(
        (pl.col("comb").max().over("q")
         - pl.when(pl.col("q_rank") == 2).then(pl.col("comb")).otherwise(None)
         .max().over("q").fill_null(0.0)).alias("q_margin"),
    )


def query_chunks(pairs: pl.DataFrame, size: int):
    """Yield slices of a q-sorted pair table that never split a query."""
    q = pairs["q"].to_numpy()
    start = 0
    n = len(q)
    while start < n:
        end = min(start + size, n)
        while end < n and q[end] == q[end - 1]:
            end += 1
        yield pairs.slice(start, end - start)
        start = end


def pair_features(pairs: pl.DataFrame, rec_q: pl.DataFrame, rec_c: pl.DataFrame,
                  freq_c: pl.DataFrame, q_is_s3: np.ndarray) -> pl.DataFrame:
    """Compute every pair feature for a chunk of candidate pairs.

    pairs  : frame with q, c, retrieval scores / ranks and context columns
    rec_q  : record_table of the concatenated S2+S3 table (indexed by q)
    rec_c  : record_table of S1 (indexed by c)
    freq_c : frequency_table of S1
    """
    qi = pairs["q"].to_numpy()
    ci = pairs["c"].to_numpy()
    Q = rec_q[qi]
    C = rec_c[ci]
    F = freq_c[ci]

    # registration ids ("Laxmi Foundation (ID: 10654)") and leading honorifics
    # ("M/s", "Sri", "Dr") are dropped before comparing names; French address
    # artefacts are canonicalised before comparing addresses
    Q, C = clean_names(Q), clean_names(C)
    Q, C = clean_address(Q), clean_address(C)
    qnc, cnc = Q["nc"].to_list(), C["nc"].to_list()
    qnk, cnk = Q["nk"].to_list(), C["nk"].to_list()
    qnn, cnn = Q["nn"].to_list(), C["nn"].to_list()
    qac, cac = Q["ac"].to_list(), C["ac"].to_list()
    qnum, cnum = Q["anum"].to_list(), C["anum"].to_list()
    qna = Q["na"].to_list()

    out = {
        "n_ratio": _cp(fuzz.ratio, qnc, cnc),
        "n_tsort": _cp(fuzz.token_sort_ratio, qnc, cnc),
        "n_tset": _cp(fuzz.token_set_ratio, qnc, cnc),
        "n_partial": _cp(fuzz.partial_ratio, qnc, cnc),
        "n_jw": _cp(JaroWinkler.normalized_similarity, qnk, cnk),
        "n_full": _cp(fuzz.token_sort_ratio, qnn, cnn),
        "n_alias": _cp(fuzz.token_set_ratio, qna, cnc),
        "a_ratio": _cp(fuzz.ratio, qac, cac),
        "a_tsort": _cp(fuzz.token_sort_ratio, qac, cac),
        "a_tset": _cp(fuzz.token_set_ratio, qac, cac),
        "a_partial": _cp(fuzz.partial_ratio, qac, cac),
        "num_ratio": _cp(fuzz.ratio, qnum, cnum),
    }
    if USE_ALIGN:
        out.update(word_align(Q["nc"], C["nc"], False, "wa_n"))
        out.update(word_align(Q["ac"], C["ac"], True, "wa_a"))
        out.update(word_align(Q["anum"], C["anum"], False, "wa_num"))
    if USE_NUMGAP:
        out.update(number_gaps(Q["anum"], C["anum"]))
    if USE_WORDFREQ:
        assert TOKEN_FREQ is not None, "call set_token_freq() with the split's S1 names first"
        tidx = Q["country"].replace_strict(COUNTRY_TABLE, default=0, return_dtype=pl.Int64).to_numpy()
        out.update(unmatched_word_freq(Q["nc"], C["nc"], TOKEN_FREQ, tidx))
    feats = pl.DataFrame(out)

    qn = Q["anum"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    cn = C["anum"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    num = pl.DataFrame({"qn": qn, "cn": cn}).select(
        pl.col("qn").list.len().alias("q_nnum"),
        pl.col("cn").list.len().alias("c_nnum"),
        pl.col("qn").list.set_intersection(pl.col("cn")).list.len().alias("num_common"),
        pl.col("qn").list.first().is_in(pl.col("cn")).fill_null(False).alias("q_first_in_c"),
        pl.col("cn").list.first().is_in(pl.col("qn")).fill_null(False).alias("c_first_in_q"),
    ).with_columns(
        (pl.col("num_common") / (pl.col("q_nnum") + pl.col("c_nnum") - pl.col("num_common"))
         .clip(1, None)).cast(pl.Float32).alias("num_jacc"),
    )

    qnl, cnl = Q["nl"], C["nl"]
    legal = pl.DataFrame({"a": qnl, "b": cnl}).select(
        ((pl.col("a") != "") & (pl.col("a") == pl.col("b"))).alias("legal_eq"),
        ((pl.col("a") != "") & (pl.col("b") != "") & (pl.col("a") != pl.col("b"))).alias("legal_diff"),
        (pl.col("a") != "").alias("q_legal"),
        (pl.col("b") != "").alias("c_legal"),
    )
    if USE_NUMGAP:  # country-agnostic legal relations (cover SARL / SAS etc. on test)
        sa = pl.col("a").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        sb = pl.col("b").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        fam = lambda e: e.list.eval(pl.element().replace(LEGAL_FAMILY)).list.unique()  # noqa: E731
        rel = pl.DataFrame({"a": qnl, "b": cnl}).select(
            sa.alias("sa"), sb.alias("sb")).select(
            (pl.col("sa").list.len() + pl.col("sb").list.len()
             - 2 * pl.col("sa").list.set_intersection(pl.col("sb")).list.len()).alias("legal_symdiff"),
            ((pl.col("sa").list.len() > 0) & (pl.col("sb").list.len() > 0)
             & (fam(pl.col("sa")).list.set_intersection(fam(pl.col("sb"))).list.len() == 0))
            .alias("legal_fam_diff"),
            ((pl.col("sa").list.len() > 0) & (pl.col("sb").list.len() > 0)
             & (pl.col("sa").list.set_intersection(pl.col("sb")).list.len() == 0)).alias("legal_disjoint"),
            ((pl.col("sa").list.len() > 0) & (pl.col("sb").list.len() > 0)
             & (pl.col("sa").list.set_intersection(pl.col("sb")).list.len() > 0)
             & (pl.col("sa").list.len() != pl.col("sb").list.len())).alias("legal_subset"),
        )
        legal = pl.concat([legal, rel], how="horizontal")
    # which legal token differs, and on which side: +1 only the query has it,
    # -1 only the S1 record has it (dropping "ltd" is noise, dropping "pvt" is not)
    lq = [pl.col("a").str.contains(rf"\b{t}\b").cast(pl.Int8) for t in LEGAL_FLAGS]
    lc = [pl.col("b").str.contains(rf"\b{t}\b").cast(pl.Int8) for t in LEGAL_FLAGS]
    ldiff = pl.DataFrame({"a": qnl, "b": cnl}).select(
        *[(a - b).alias(f"ld_{t}") for t, a, b in zip(LEGAL_FLAGS, lq, lc)],
        (pl.col("a").str.replace_all(r"\b(?:" + "|".join(LEGAL_FLAGS) + r")\b", "").str.strip_chars()
         != pl.col("b").str.replace_all(r"\b(?:" + "|".join(LEGAL_FLAGS) + r")\b", "").str.strip_chars())
        .cast(pl.Int8).alias("ld_other"),
    ) if USE_LEGAL_DIFF else None

    base = pairs.select(
        "sn", "sa", "kn", "ka", "comb", "kcomb", "rk_n", "rk_a", "rk_c", "rk_kn", "rk_ka",
        "rk_kc", "q_ncand", "c_ncand", "q_gap", "q_kgap", "c_gap", "c_kgap", "q_rank",
        "q_krank", "c_rank", "c_krank", "q_gap_n", "q_gap_a", "q_margin",
    ).cast(pl.Float32)
    extra = pl.DataFrame({
        "q_is_s3": q_is_s3[qi].astype(np.int8),
        "q_native": Q["native"].cast(pl.Int8),
        "q_addr_empty": Q["addr_empty"].cast(pl.Int8),
        "c_addr_empty": C["addr_empty"].cast(pl.Int8),
        "q_ntok": Q["n_tok"].cast(pl.Float32),
        "c_ntok": C["n_tok"].cast(pl.Float32),
        "q_has_alias": (Q["na"] != "").cast(pl.Int8),
        "nk_eq": (Q["nk"] == C["nk"]).cast(pl.Int8),
        "ac_eq": (Q["ac"] == C["ac"]).cast(pl.Int8),
        "len_n_diff": (Q["nk"].str.len_chars().cast(pl.Int32)
                       - C["nk"].str.len_chars().cast(pl.Int32)).abs().cast(pl.Float32),
    })
    parts = [base, feats, num.cast({c: pl.Float32 for c in num.columns}), legal.cast(pl.Int8)]
    if ldiff is not None:
        parts.append(ldiff.cast(pl.Int8))
    if USE_ACRONYM:
        parts.append(acronym_features(Q, C))
    if USE_WORDFREQ:
        parts.append(word_change_features(Q, C))
    return pl.concat(parts + [extra, F], how="horizontal")


STREET_TYPES = sorted(set(ADDR_ABBR.values()))


def word_change_features(Q: pl.DataFrame, C: pl.DataFrame) -> pl.DataFrame:
    """Which name words were added or removed, and do the street words agree?

    dq_n / dc_n     number of core-name words only in the query / only in S1
    dq_noise        how many of the query-only words are known noise insertions
    q_noise_legal   query ends with "<legal form> [word] Center/Services/Partners" and S1 does not
    addr_ajacc      Jaccard of the non-numeric, non-street-type address words (street and
                    town names), independent of component order
    """
    words = lambda e: e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))  # noqa: E731
    d = pl.DataFrame({"qt": words(Q["nc"]), "ct": words(C["nc"])}).select(
        pl.col("qt").list.set_difference(pl.col("ct")).alias("dq"),
        pl.col("ct").list.set_difference(pl.col("qt")).alias("dc"))
    addr = lambda e: e.str.split(" ").list.eval(  # noqa: E731
        pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().str.contains(r"^\d+$")
                            & ~pl.element().is_in(STREET_TYPES)))
    a = pl.DataFrame({"qa": addr(Q["ac"]), "ca": addr(C["ac"])}).select(
        (pl.col("qa").list.set_intersection(pl.col("ca")).list.len()
         / pl.col("qa").list.set_union(pl.col("ca")).list.len().clip(1, None)).alias("addr_ajacc"))
    return pl.DataFrame({
        "dq_n": d["dq"].list.len().cast(pl.Float32),
        "dc_n": d["dc"].list.len().cast(pl.Float32),
        "dq_noise": d["dq"].list.eval(pl.element().is_in(NOISE_ADD)).list.sum().cast(pl.Float32),
        "q_noise_legal": (Q["nn"].str.contains(NOISE_AFTER_LEGAL)
                          & ~C["nn"].str.contains(NOISE_AFTER_LEGAL)).cast(pl.Int8),
        "addr_ajacc": a["addr_ajacc"].cast(pl.Float32),
    })


def _initials(words: pl.Series) -> pl.Series:
    """First letter of every word ("maison tracto" -> "mt")."""
    return words.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")


def acronym_features(Q: pl.DataFrame, C: pl.DataFrame) -> pl.DataFrame:
    """Is one name the initials of the other? ("MT" for "Maison Tracto SAS").

    Checked against the initials of the core name and of the full name (with
    legal words), in both directions, for compact names of 2-5 letters.
    """
    qk, ck = Q["nk"], C["nk"]
    ql, cl = qk.str.len_chars(), ck.str.len_chars()
    q_is = ((qk == _initials(C["nc"])) | (qk == _initials(C["nn"]))) & (ql >= 2) & (ql <= 5)
    c_is = ((ck == _initials(Q["nc"])) | (ck == _initials(Q["nn"]))) & (cl >= 2) & (cl <= 5)
    return pl.DataFrame({
        "acr_q_of_c": q_is.cast(pl.Int8), "acr_c_of_q": c_is.cast(pl.Int8),
        "q_short": (ql <= 4).cast(pl.Int8), "c_short": (cl <= 4).cast(pl.Int8),
    })


