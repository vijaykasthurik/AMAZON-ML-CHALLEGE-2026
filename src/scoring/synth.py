"""Synthetic labelled validation for countries without training labels (France).

For a sample of real Source 1 records we generate noisy copies with the noise rates
measured on real training links (noise_stats.py -> outputs/noise_rates.json):
  * positives  : 1-2 noisy variants that SHOULD match their Source 1 record
  * negatives  : "siblings" sharing the name but with another house number/street, or
                 sharing the address with a different name — they should NOT match
They are inserted into the full real record pool of the country, and the exact
production pipeline (blocking -> features -> stage 1/2 -> decision) scores them.

Calibration: running the same procedure on India/US (where real labels exist) tells
us how synthetic metrics relate to real ones, so France numbers can be interpreted.

    python synth.py --split test --country France --n 3000
"""
import argparse
import json
import os
import random
import re

import numpy as np
import polars as pl

from block import block_frame
from config import CACHE, OUTPUTS
from data import load_norm
from decide import expected_f, one_owner
from features import add_context, country_features, prepare_country, K_KEEP, R_KEEP
from normalize import set_dict
from preprocess import _work
from train import MODELS, stage2_frame, S2_EXTRA

# measured by noise_stats.py on training links; the embedded copy is used where outputs/ does not have it
_RATES_TRAIN = {"name_same": 0.0499, "upper": 0.1232, "lower": 0.0757, "junk_prefix": 0.0148, "legal_drop": 0.1668,
                "legal_add": 0.0676, "shuffle": 0.0617, "dup_word": 0.0388, "typo": 0.1827, "leet": 0.0286,
                "accent": 0.0728, "extra_word": 0.1016, "drop_word": 0.0589, "website": 0.0595, "aka": 0.0125,
                "addr_null": 0.0472, "addr_upper": 0.3145, "addr_reorder": 0.0295, "addr_num_change": 0.2633,
                "addr_num_missing": 0.0514, "addr_drop_part": 0.2071, "addr_num_prefix": 0.1895,
                "addr_extra_part": 0.0536, "addr_same": 0.0233}
_rp = os.path.join(OUTPUTS, "noise_rates.json")
RATES = json.load(open(_rp)) if os.path.exists(_rp) else _RATES_TRAIN
ACC = {"a": "àâá", "e": "éèêë", "i": "íîï", "o": "óôö", "u": "úûü", "c": "ç"}
LEET = {"o": "0", "l": "1", "e": "3", "s": "5", "a": "4", "i": "1"}
JUNK = ["--", ">>", "***", "<<", "##", "..."]
FR_ADDR = [("Rue", ["R.", "R", "RUE"]), ("Avenue", ["Av.", "AVE", "AV"]), ("Boulevard", ["Bd", "BLVD", "BD"]),
           ("Allée", ["All.", "ALL"]), ("Place", ["Pl.", "PL"]), ("Chemin", ["Ch.", "CH"]),
           ("Impasse", ["Imp.", "IMP"]), ("Route", ["Rte", "RTE"])]
EN_ADDR = [("Street", ["St", "ST"]), ("Road", ["Rd", "RD"]), ("Avenue", ["Ave", "AVE"]),
           ("Drive", ["Dr", "DR"]), ("Lane", ["Ln", "LN"]), ("Boulevard", ["Blvd"]), ("Court", ["Ct"])]
FR_LEGAL = ["SARL", "SAS", "EURL", "SA", "SCI", "SNC", "SASU", "S.A.R.L.", "S.A.S"]
EN_LEGAL = ["Inc", "LLC", "Ltd", "Corp", "Co", "Private Limited", "Pvt Ltd", "LLP"]
FR_EXTRA = ["France", "(France)", "Groupe", "& Fils", "Services", "Club", "International"]
EN_EXTRA = ["Services", "Center", "Holdings", "Group", "International", "Solutions"]


def p(key):
    """Probability of a noise operation."""
    return RATES.get(key, 0.0)


def typo(w, r):
    """One random character edit."""
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.random()
    if op < 0.3:
        return w[:i] + w[i + 1:]
    if op < 0.6:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op < 0.8:
        return w[:i] + r.choice("aeioulnrst") + w[i:]
    return w[:i] + r.choice("aeioulnrst") + w[i + 1:]


def noisy_name(name, fr, r):
    """Apply measured name-noise operations."""
    legal = FR_LEGAL if fr else EN_LEGAL
    ws = name.split()
    if ws and ws[-1].upper().strip(".") in {l.upper().strip(".") for l in legal} and r.random() < p("legal_drop") * 1.5:
        ws = ws[:-1]
    elif r.random() < p("legal_add"):
        ws.append(r.choice(legal))
    if len(ws) > 1 and r.random() < p("shuffle"):
        r.shuffle(ws)
    if ws and r.random() < p("dup_word"):
        j = r.randrange(len(ws)); ws.insert(j, ws[j])
    if ws and r.random() < p("typo"):
        j = r.randrange(len(ws)); ws[j] = typo(ws[j], r)
    if r.random() < p("extra_word"):
        ws.insert(r.randrange(len(ws) + 1), r.choice(FR_EXTRA if fr else EN_EXTRA))
    if len(ws) > 2 and r.random() < p("drop_word"):
        ws.pop(r.randrange(len(ws)))
    s = " ".join(ws)
    if r.random() < p("accent"):
        idx = [k for k, ch in enumerate(s) if ch.lower() in ACC]
        if idx:
            k = r.choice(idx); s = s[:k] + r.choice(ACC[s[k].lower()]) + s[k + 1:]
    if r.random() < p("leet"):
        idx = [k for k, ch in enumerate(s) if ch.lower() in LEET]
        if idx:
            k = r.choice(idx); s = s[:k] + LEET[s[k].lower()] + s[k + 1:]
    if r.random() < p("website"):
        s = re.sub(r"[^a-z0-9]", "", s.lower()) + r.choice([".com", ".fr" if fr else ".net"])
    u = r.random()
    if u < p("upper"):
        s = s.upper()
    elif u < p("upper") + p("lower"):
        s = s.lower()
    if r.random() < p("junk_prefix"):
        s = r.choice(JUNK) + " " + s
    return s


def noisy_addr(addr, fr, r):
    """Apply measured address-noise operations."""
    if r.random() < p("addr_null"):
        return None
    for full, abbrs in (FR_ADDR if fr else EN_ADDR):
        if full.lower() in addr.lower() and r.random() < 0.5:
            addr = re.sub(full, r.choice(abbrs), addr, flags=re.I)
    parts = [x.strip() for x in addr.split(",") if x.strip()]
    if len(parts) > 2 and r.random() < p("addr_drop_part"):
        parts.pop(r.randrange(1, len(parts)))
    if len(parts) > 1 and r.random() < p("addr_reorder"):
        r.shuffle(parts)
    s = ", ".join(parts)
    nums = re.findall(r"\d+", s)
    if nums and r.random() < p("addr_num_missing"):
        s = re.sub(r"\b\d+\w?\b\s*", "", s, count=1)
    elif nums and r.random() < p("addr_num_change"):
        n = nums[0]
        alt = r.choice([n + r.choice("ABCD"), n[:-1] if len(n) > 1 else n + "0", typo(n + "0", r)[:len(n) + 1]])
        s = s.replace(n, alt, 1)
    if nums and r.random() < p("addr_num_prefix"):
        s = r.choice(["N° ", "No ", "# ", "## "]) + s
    if r.random() < p("addr_upper"):
        s = s.upper()
    return s


def sibling(name, addr, fr, r, pool_names):
    """A hard negative: same name at a different house number, or same address with another name."""
    if r.random() < 0.5 and addr and re.search(r"\d+", addr):
        n = re.search(r"\d+", addr).group(0)
        other = str(int(n) + r.choice([-7, -3, 2, 5, 11, 40])) if int(n) > 40 else str(int(n) + 13)
        return name, addr.replace(n, other, 1)
    return r.choice(pool_names), addr


def generate(raw1: pl.DataFrame, fr: bool, seed=0):
    """Build synthetic (id, name, address, country, src, owner, label) rows."""
    r = random.Random(seed)
    names = raw1["business_name"].to_list()
    rows = []
    for k, (eid, nm, ad, ct) in enumerate(raw1.select("entity_id", "business_name", "business_address",
                                                      "country").iter_rows()):
        for j in range(r.choice([1, 1, 2])):
            src = r.choice([2, 3])
            rows.append((f"S{src}-SYNP{k}_{j}", noisy_name(nm, fr, r), noisy_addr(ad or "", fr, r), ct, src, eid, 1))
        if r.random() < 0.6:
            sn, sa = sibling(nm, ad, fr, r, names)
            src = r.choice([2, 3])
            rows.append((f"S{src}-SYNN{k}", noisy_name(sn, fr, r), noisy_addr(sa or "", fr, r), ct, src, eid, 0))
    return pl.DataFrame(rows, schema=["id", "business_name", "business_address", "country", "src", "owner", "label"],
                        orient="row")


def normalize_rows(syn: pl.DataFrame) -> pl.DataFrame:
    """Run the production normalizer on synthetic raw rows."""
    set_dict(json.load(open(os.path.join(CACHE, "translit.json"), encoding="utf-8")))
    out = _work(list(zip(syn["business_name"].to_list(), syn["business_address"].to_list(),
                         syn["country"].to_list())))
    names = ["nf", "nc", "ncp", "web", "nat", "at", "num", "state", "anull"]
    nd = pl.DataFrame({n: list(c) for n, c in zip(names, zip(*out))})
    return pl.concat([syn.select("id", "country", pl.col("src").cast(pl.Int8)), nd], how="horizontal") \
             .with_columns(pl.col("web").cast(pl.Int8), pl.col("nat").cast(pl.Int8), pl.col("anull").cast(pl.Int8))


def score(pairs_feat: pl.DataFrame, split: str = "test"):
    """Stage 1 + stage 2 with the saved fold models (averaged)."""
    import xgboost as xgb
    c1 = json.load(open(os.path.join(MODELS, "stage1_features.json")))
    c2 = json.load(open(os.path.join(MODELS, "stage2_features.json")))

    def avg(tag, X):
        bs = []
        for f in range(5):
            b = xgb.Booster(); b.load_model(os.path.join(MODELS, f"{tag}_f{f}.json")); bs.append(b)
        d = xgb.DMatrix(X)
        return np.mean([b.predict(d) for b in bs], axis=0)

    p1 = avg("s1", pairs_feat.select(c1).to_numpy().astype(np.float32))
    full = stage2_frame(pairs_feat.select(["id1", "id2"] + S2_EXTRA).with_columns(pl.Series("p1", p1)))
    if any(c.startswith("sib_") for c in c2):
        import sib
        full = sib.add_siblings(full, split)
    return full.with_columns(pl.Series("p2", avg("s2", full.select(c2).to_numpy().astype(np.float32))))


def main():
    """Generate, insert, run the pipeline, report synthetic precision/recall/false-merge rates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--country", default="France")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    fr = a.country == "France"
    pool = load_norm(a.split).filter(pl.col("country") == a.country)
    raw1 = pl.read_parquet(os.path.join(CACHE, f"{a.split}_source1.parquet"))
    raw1 = raw1.join(pool.filter(pl.col("src") == 1).select(pl.col("id").alias("entity_id")), on="entity_id",
                     how="semi")
    raw1 = raw1.sample(min(a.n, raw1.height), seed=7)
    syn = generate(raw1, fr)
    synn = normalize_rows(syn)
    uni = pl.concat([pool, synn.select(pool.columns)])
    cand = block_frame(uni)
    cand = add_context(cand, "s_blk", "blk")
    cand = cand.filter((pl.col("rk1_blk") <= K_KEEP) | (pl.col("rk2_blk") <= R_KEEP))
    cand = cand.with_columns(pl.col("in_fwd").cast(pl.Int8), pl.col("in_rev").cast(pl.Int8))
    # only the sampled Source 1 records' candidate lists matter, plus every pair touching a synthetic id
    s_ids = raw1["entity_id"]
    keep = cand.filter(pl.col("id1").is_in(s_ids.implode()) | pl.col("id2").str.contains("SYN"))
    touched = keep.select("id2").unique()
    sub = cand.join(touched, on="id2", how="semi")  # full target context for scored ids
    sub = pl.concat([sub, cand.filter(pl.col("id1").is_in(s_ids.implode()))]).unique(["id1", "id2"])
    rec, mats = prepare_country(uni, a.country)
    # mirror the tiered inference path: cheap (tier-0) features, then full features on top
    from features import cheap_frame
    cheap = cheap_frame(rec, mats, sub)
    cheap = cheap.select([c for c in cheap.columns if not c.startswith("cos_")])
    feat = country_features(rec, mats, cheap, None)
    sc = score(feat, a.split)
    base = one_owner(sc.select("id1", "id2", pl.col("p2").alias("p")))
    cfg = json.load(open(os.path.join(MODELS, "decision.json")))
    sel = expected_f(base, alpha=cfg["param"]) if cfg["rule"].endswith("expF") else base.filter(pl.col("p") >= cfg["param"])
    lab = syn.select(pl.col("id").alias("id2"), "owner", "label")
    ss = sel.join(lab, on="id2")
    pos = lab.filter(pl.col("label") == 1)
    found = cand.join(pos, on="id2").filter(pl.col("id1") == pl.col("owner"))
    res = {
        "country": a.country, "split": a.split, "n_s1": a.n, "n_pos": pos.height,
        "n_neg": lab.height - pos.height,
        "blocking_recall": found.height / pos.height,
        "pos_recall": ss.filter((pl.col("label") == 1) & (pl.col("id1") == pl.col("owner"))).height / pos.height,
        "pos_wrong_owner": ss.filter((pl.col("label") == 1) & (pl.col("id1") != pl.col("owner"))).height / pos.height,
        "neg_false_accept": ss.filter(pl.col("label") == 0).height / max(lab.height - pos.height, 1),
    }
    tp = ss.filter((pl.col("label") == 1) & (pl.col("id1") == pl.col("owner"))).height
    res["syn_precision"] = tp / max(ss.height, 1)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUTPUTS, f"synth_{a.split}_{a.country}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
