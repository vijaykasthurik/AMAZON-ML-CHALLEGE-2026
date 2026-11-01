"""Single entry point for loading normalized records (with an optional smoke-test subset).

BER_SMOKE=1 keeps ~0.5% of Source 1 entities, all their true matches (train), and ~0.5%
of the other records, so the whole pipeline can be exercised end-to-end in about a minute.
"""
import os
import polars as pl
from config import CACHE

SMOKE = os.environ.get("BER_SMOKE") == "1"


def _smoke(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """Deterministic small subset that keeps train links intact."""
    keep = pl.col("id").hash(seed=3) % 200 == 0
    if split != "train":
        return df.filter(keep)
    gt = pl.read_parquet(os.path.join(CACHE, "train_ground_truth.parquet"))
    gt = gt.filter(pl.col("source1_entity_id").hash(seed=3) % 200 == 0)
    linked = (gt["matched_entity_ids"].fill_null("").str.split(",").explode())
    return df.filter(keep | pl.col("id").is_in(linked.implode()))


DROP_S1 = float(os.environ.get("BER_DROPS1", "0"))


def load_norm(split: str, sources=(1, 2, 3), columns=None) -> pl.DataFrame:
    """Normalized records of a split for the given sources (smoke-filtered when enabled).

    Density matching (BER_DROPS1): the test set has ~2.29 unmatched Source 2/3 records per Source 1
    entity versus ~1.22 in train. Removing a random DROP_S1 fraction of TRAIN Source 1 entities turns
    their true matches into realistic distractors and reproduces the test density (0.187 -> 2.29).
    """
    fr = [pl.read_parquet(os.path.join(CACHE, f"{split}_s{i}_norm.parquet"), columns=columns) for i in sources]
    df = pl.concat(fr)
    if split == "train" and DROP_S1 > 0 and "id" in df.columns:
        drop = pl.col("id").str.starts_with("S1-") & (pl.col("id").hash(seed=99) % 1000 < int(DROP_S1 * 1000))
        df = df.filter(~drop)
    return _smoke(df, split) if SMOKE else df


def s1_ids(split: str) -> pl.Series:
    """All Source 1 ids of a split (smoke-filtered when enabled)."""
    return load_norm(split, sources=(1,), columns=["id", "country"])["id"]
