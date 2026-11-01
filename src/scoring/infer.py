"""Test inference: features -> stage 1 (fold-averaged) -> stage 2 (fold-averaged) -> decision -> TSVs.

Assumes block.py test and features.py test have been run (reproduce.py runs everything).
Writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

from config import OUTPUT, OUTPUTS, WORK
from data import s1_ids
from train import MODELS, NF, S2_EXTRA, apply_rule, feature_files, stage2_frame
import progress


def load_models(tag):
    """Load the NF fold boosters of a stage."""
    bs = []
    for f in range(NF):
        b = xgb.Booster()
        b.load_model(os.path.join(MODELS, f"{tag}_f{f}.json"))
        if os.environ.get("BER_CPU") != "1":
            b.set_param({"device": "cuda"})  # GPU prediction: ~30x faster than 4 CPU cores
        bs.append(b)
    return bs


def avg_predict(bs, X):
    """Average probability of all fold models."""
    d = xgb.DMatrix(X)
    return np.mean([b.predict(d) for b in bs], axis=0).astype(np.float32)


def write_lists(pairs: pl.DataFrame, s1_ids: pl.Series, col: str, path: str):
    """Write one row per Source 1 id with a comma-joined sorted id list (empty when none)."""
    g = pairs.group_by("id1").agg(pl.col("id2").unique().sort().str.join(",").alias(col))
    out = pl.DataFrame({"source1_entity_id": s1_ids}).join(
        g.rename({"id1": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            fh.write(f"{a}\t{b}\n")


def adapted_variant(full: pl.DataFrame, cols3, cfg, ids1):
    """Second submission variant: for countries without training labels, re-score the cross-encoder
    feature with a copy adapted on synthetic pairs derived from that country's own test records,
    re-run the stage-2 model on those rows only, and write matching_results_adapted.tsv."""
    import ce
    from data import load_norm
    s1c = load_norm("test", sources=(1,), columns=["id", "country"])
    trained = set(load_norm("train", sources=(1,), columns=["country"])["country"].unique().to_list())
    for country in sorted(set(s1c["country"].unique().to_list()) - trained):
        cids = s1c.filter(pl.col("country") == country).select(pl.col("id").alias("id1"))
        model, tok = ce.adapt_country("test", country)
        rows = full.with_row_index("_row").join(cids, on="id1", how="semi").filter(pl.col("ce").is_not_nan())
        d = ce.attach_text(rows.select("_row", "id1", "id2"), ce.texts("test"))
        s = ce.score(model, tok, d["t1"].to_list(), d["t2"].to_list())
        cev = full["ce"].to_numpy().copy()
        cev[d["_row"].to_numpy()] = s
        full = full.with_columns(pl.Series("ce", cev))
        idx = full.with_row_index("_row").join(cids, on="id1", how="semi")["_row"].to_numpy()
        sub = full[idx]
        p = full["p2ce"].to_numpy().copy()
        p[idx] = avg_predict(load_models("s2ce"), sub.select(cols3).to_numpy().astype(np.float32))
        full = full.with_columns(pl.Series("p2ce", p))
        print(f"adapted {country}: re-scored {len(s):,} cross-encoder pairs, {len(idx):,} candidate rows", flush=True)
    full.select("id1", "id2", "p1", "p2", "ce", "p2ce").write_parquet(os.path.join(OUTPUTS, "test_scores_adapted.parquet"))
    sel = apply_rule(full, cfg["pred"], cfg["rule"], cfg["param"], cfg.get("gate", 0.0))
    write_lists(sel, ids1, "matched_entity_ids", os.path.join(OUTPUT, "matching_results_adapted.tsv"))


def main():
    """Score all test candidates and write the submission files."""
    cfg = json.load(open(os.path.join(MODELS, "decision.json")))
    cols1 = json.load(open(os.path.join(MODELS, "stage1_features.json")))
    cols2 = json.load(open(os.path.join(MODELS, "stage2_features.json")))
    b1 = load_models("s1")
    parts = []
    files = feature_files("test")
    for k, p in enumerate(files):
        d = pl.read_parquet(p)
        p1 = avg_predict(b1, d.select(cols1).to_numpy().astype(np.float32))
        parts.append(d.select(["id1", "id2"] + S2_EXTRA).with_columns(pl.Series("p1", p1)))
        progress.update("Features + scoring (test)", k + 1, len(files), msg=f"stage-1 scored {os.path.basename(p)}")
    scored = pl.concat(parts)
    del parts
    # learned pruning stage (meta-blocking): the final matcher only sees pairs with p1 >= TAU;
    # this pruned set IS candidate_pairs.tsv
    tau = float(os.environ.get("BER_TAU", cfg.get("tau", 0.003)))
    full = stage2_frame(scored.filter(pl.col("p1") >= tau))
    if "sib_n" in cols2:
        import sib
        full = sib.add_siblings(full, "test")
    n1 = s1_ids("test").len()
    print(f"pruning tau={tau}: {scored.height:,} retrieved -> {full.height:,} candidates "
          f"({scored.height / n1:.2f} -> {full.height / n1:.2f} per Source 1)", flush=True)
    del scored
    b2 = load_models("s2")
    full = full.with_columns(pl.Series("p2", avg_predict(b2, full.select(cols2).to_numpy().astype(np.float32))))
    keep = ["id1", "id2", "p1", "p2"]
    if cfg["pred"] == "p2ce":
        import ce
        full = ce.add_ce(full, "test")
        cols3 = json.load(open(os.path.join(MODELS, "stage2ce_features.json")))
        full = full.with_columns(pl.Series("p2ce", avg_predict(load_models("s2ce"),
                                                               full.select(cols3).to_numpy().astype(np.float32))))
        keep += ["ce", "p2ce"]
    full.select(keep).write_parquet(os.path.join(OUTPUTS, "test_scores.parquet"))

    sel = apply_rule(full, cfg["pred"], cfg["rule"], cfg["param"], cfg.get("gate", 0.0))

    ids1 = s1_ids("test")
    cand = full.select("id1", "id2")
    write_lists(sel, ids1, "matched_entity_ids", os.path.join(OUTPUT, "matching_results.tsv"))
    write_lists(cand, ids1, "candidate_entity_ids", os.path.join(OUTPUT, "candidate_pairs.tsv"))
    n_match = sel.height
    progress.update("Write submission", 1, 1, state="done",
                    msg=f"wrote matching_results.tsv: {ids1.len():,} rows, {n_match:,} matched ids")
    print("rows", ids1.len(), "matched ids", n_match,
          "singletons predicted", ids1.len() - sel["id1"].n_unique(), flush=True)
    if cfg["pred"] == "p2ce" and os.environ.get("BER_FR_ADAPT") == "1":
        try:  # optional second variant; never allowed to break the main submission files
            adapted_variant(full, cols3, cfg, ids1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("adapted variant failed:", e, flush=True)


if __name__ == "__main__":
    main()
