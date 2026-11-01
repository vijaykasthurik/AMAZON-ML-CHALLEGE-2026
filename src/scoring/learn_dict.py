"""Learn a native-script -> Latin token dictionary from TRAINING ground-truth pairs only.

For every labelled pair where the Source 2/3 record is written in an Indic script
and the Source 1 record is Latin, each native token is aligned to the Source 1
token with the most similar phonetic skeleton (after rough unidecode). Votes are
aggregated and a token is kept when its winner has >= 3 votes and >= 60% share.
"""
import collections
import json
import os
import re

import polars as pl
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

from config import CACHE
from normalize import INDIC, phon

TOK = re.compile(r"[^\s,]+")


def latin_tokens(s):
    """Lowercase ASCII word tokens of a Latin string."""
    return [t for t in re.split(r"[^a-z0-9]+", unidecode(s).lower()) if t]


def main(max_pairs=600_000):
    """Collect aligned (native, latin) token votes and write cache/translit.json."""
    gt = pl.read_parquet(os.path.join(CACHE, "train_ground_truth.parquet"))
    ex = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    s1 = pl.read_parquet(os.path.join(CACHE, "train_source1.parquet")).filter(pl.col("country") == "India")
    oth = pl.concat([pl.read_parquet(os.path.join(CACHE, f"train_source{i}.parquet")) for i in (2, 3)])
    oth = oth.filter(pl.col("country") == "India").filter(
        pl.col("business_name").str.contains(r"[ऀ-෿]") |
        pl.col("business_address").fill_null("").str.contains(r"[ऀ-෿]"))
    pairs = (ex.join(s1, left_on="source1_entity_id", right_on="entity_id")
               .join(oth, left_on="matched_entity_ids", right_on="entity_id", suffix="_m"))
    pairs = pairs.head(max_pairs)
    print("native pairs", pairs.height, flush=True)
    votes = collections.defaultdict(collections.Counter)
    for n1, a1, n2, a2 in pairs.select("business_name", "business_address",
                                        "business_name_m", "business_address_m").iter_rows():
        for src, dst in ((n2, n1), (a2 or "", a1 or "")):
            if not INDIC.search(src):
                continue
            lat = latin_tokens(dst)
            if not lat:
                continue
            lat_ph = [phon(t) for t in lat]
            # also allow adjacent bigrams ("tamil nadu" <- one native token)
            big = [lat[i] + " " + lat[i + 1] for i in range(len(lat) - 1)]
            big_ph = [phon(lat[i] + lat[i + 1]) for i in range(len(lat) - 1)]
            for t in TOK.findall(src):
                t = t.strip(".,;:()[]-")
                if not INDIC.search(t):
                    continue
                p = phon(re.sub(r"[^a-z0-9]", "", unidecode(t).lower()))
                if not p:
                    continue
                best, bs = None, 0.0
                for cand, cp in zip(lat + big, lat_ph + big_ph):
                    sc = Levenshtein.normalized_similarity(p, cp)
                    if sc > bs:
                        best, bs = cand, sc
                if best is not None and bs >= 0.5:
                    votes[t][best] += 1
    d = {}
    for t, c in votes.items():
        w, n = c.most_common(1)[0]
        if n >= 3 and n / sum(c.values()) >= 0.6:
            d[t] = w
    with open(os.path.join(CACHE, "translit.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print("dictionary size", len(d), "of", len(votes), "native tokens", flush=True)
    for k in list(d)[:40]:
        print(k, "->", d[k])


if __name__ == "__main__":
    main()
