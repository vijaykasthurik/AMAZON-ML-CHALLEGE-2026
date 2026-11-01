"""Measure noise-operation rates between Source 1 records and their true Source 2/3 matches.

The rates parameterize synth.py, which applies the same noise to French Source 1 records
to build a labelled synthetic French validation set (France has no training labels).
"""
import json
import os
import re

import polars as pl
from unidecode import unidecode

from config import CACHE, OUTPUTS
from features import load_truth

LEGAL = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "pvt", "private",
         "incorporated", "llp", "pc", "plc", "lp"}


def words(s):
    """Lowercase alphanumeric words."""
    return re.findall(r"[a-z0-9]+", unidecode(s or "").lower())


def main(n=60000):
    """Sample linked pairs and tally which transformations separate them."""
    tr = load_truth().sample(n, seed=1)
    s1 = pl.read_parquet(os.path.join(CACHE, "train_source1.parquet"))
    oth = pl.concat([pl.read_parquet(os.path.join(CACHE, f"train_source{i}.parquet")) for i in (2, 3)])
    d = (tr.join(s1, left_on="id1", right_on="entity_id")
           .join(oth, left_on="id2", right_on="entity_id", suffix="_m"))
    d = d.filter(~pl.col("business_name_m").str.contains(r"[ऀ-෿]"))
    c = {k: 0 for k in ["n", "name_same", "upper", "lower", "junk_prefix", "legal_drop", "legal_add",
                        "shuffle", "dup_word", "typo", "leet", "accent", "extra_word", "drop_word",
                        "website", "aka", "addr_null", "addr_upper", "addr_reorder", "addr_num_change",
                        "addr_num_missing", "addr_drop_part", "addr_num_prefix", "addr_extra_part",
                        "addr_same"]}
    for n1, a1, n2, a2 in d.select("business_name", "business_address", "business_name_m",
                                    "business_address_m").iter_rows():
        c["n"] += 1
        w1, w2 = words(n1), words(n2)
        c["name_same"] += n1 == n2
        c["upper"] += n2 == n2.upper() and n1 != n1.upper()
        c["lower"] += n2 == n2.lower() and n1 != n1.lower()
        c["junk_prefix"] += bool(re.match(r"^\W{2,}", n2))
        c["legal_drop"] += bool(set(w1) & LEGAL) and not (set(w2) & LEGAL)
        c["legal_add"] += bool(set(w2) & LEGAL) and not (set(w1) & LEGAL)
        c["shuffle"] += sorted(w1) == sorted(w2) and w1 != w2
        c["dup_word"] += len(w2) != len(set(w2)) and len(w1) == len(set(w1))
        c["leet"] += bool(re.search(r"[a-z][0-9][a-z]|[0-9][a-z]{3}", unidecode(n2).lower())) and \
            not re.search(r"[a-z][0-9][a-z]|[0-9][a-z]{3}", unidecode(n1).lower())
        c["accent"] += bool(re.search(r"[^\x00-\x7f]", n2)) and not re.search(r"[^\x00-\x7f]", n1)
        s1w, s2w = set(w1) - LEGAL, set(w2) - LEGAL
        c["extra_word"] += len(s2w - s1w) > 0 and s1w <= s2w
        c["drop_word"] += len(s1w - s2w) > 0 and s2w <= s1w and len(s2w) > 0
        c["typo"] += len(s1w ^ s2w) == 2 and len(s1w) == len(s2w)
        c["website"] += bool(re.search(r"\.(com|net|org|in)$|^@", n2.lower()))
        c["aka"] += bool(re.search(r"a/k/a|d/b/a|formerly|dba", n2.lower()))
        if a2 is None or a2.strip() == "":
            c["addr_null"] += 1
            continue
        c["addr_same"] += a1 == a2
        c["addr_upper"] += a2 == a2.upper() and a1 != a1.upper()
        p1 = [p.strip().lower() for p in a1.split(",")]
        p2 = [p.strip().lower() for p in a2.split(",")]
        c["addr_reorder"] += sorted(p1) == sorted(p2) and p1 != p2
        c["addr_drop_part"] += len(p2) < len(p1)
        c["addr_extra_part"] += len(p2) > len(p1)
        n1s, n2s = set(re.findall(r"\d+", a1)), set(re.findall(r"\d+", a2))
        c["addr_num_missing"] += bool(n1s) and not n2s
        c["addr_num_change"] += bool(n1s) and bool(n2s) and n1s != n2s
        c["addr_num_prefix"] += bool(re.search(r"#|\bno\.?\s|n°|door no|h\.?no", a2.lower()))
    rates = {k: round(v / c["n"], 4) for k, v in c.items() if k != "n"}
    rates["n"] = c["n"]
    json.dump(rates, open(os.path.join(OUTPUTS, "noise_rates.json"), "w"), indent=1)
    for k, v in rates.items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
