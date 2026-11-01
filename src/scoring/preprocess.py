"""Normalize every source record (train + test) into compact matching fields.

Output: cache/{split}_s{1,2,3}_norm.parquet with columns
  id, country, src, nf (full name), nc (core name), ncp (sorted phonetic core tokens),
  web (name looks like a website/handle), nat (name in Indic script),
  at (address tokens), num (address numbers), state, anull (address missing)
"""
import json
import os
import sys
from multiprocessing import Pool

import polars as pl

from config import CACHE
from normalize import INDIC, norm_addr, norm_name, phon, set_dict
import progress

CHUNK = 100_000


def _init(d):
    """Worker initializer: install the learned transliteration dictionary."""
    set_dict(d)


def _work(rows):
    """Normalize a list of (name, address, country) tuples."""
    out = []
    for name, addr, country in rows:
        nf, nc, web = norm_name(name or "")
        at, num, st = norm_addr(addr, country)
        ncp = " ".join(sorted({phon(t) for t in nc.split()}))
        out.append((nf, nc, ncp, web, int(bool(INDIC.search(name or ""))), at, num, st,
                    int(addr is None or addr.strip() == "" or addr.strip().lower() in ("none", "null"))))
    return out


def main():
    """Normalize all six source files with a process pool."""
    with open(os.path.join(CACHE, "translit.json"), encoding="utf-8") as f:
        d = json.load(f)
    jobs = [(sp, i) for sp in ("test", "train") for i in (1, 2, 3)]
    with Pool(int(os.environ.get("BER_WORKERS", "12")), initializer=_init, initargs=(d,)) as pool:
        for k, (sp, i) in enumerate(jobs):
            out = os.path.join(CACHE, f"{sp}_s{i}_norm.parquet")
            if os.path.exists(out):
                continue
            df = pl.read_parquet(os.path.join(CACHE, f"{sp}_source{i}.parquet"))
            names = ["nf", "nc", "ncp", "web", "nat", "at", "num", "state", "anull"]

            def chunks():
                """Yield row tuples in slices so the parent never holds all of them at once."""
                for j in range(0, df.height, CHUNK):
                    sl = df.slice(j, CHUNK)
                    yield list(zip(sl["business_name"].to_list(), sl["business_address"].to_list(),
                                   sl["country"].to_list()))

            parts = []
            for r in pool.imap(_work, chunks()):
                cols = list(zip(*r))
                parts.append(pl.DataFrame({n: list(c) for n, c in zip(names, cols)}))
            nd = pl.concat(parts)
            nd = pl.concat([df.select(pl.col("entity_id").alias("id"), "country"), nd], how="horizontal")
            nd = nd.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"),
                                 pl.col("web").cast(pl.Int8), pl.col("nat").cast(pl.Int8),
                                 pl.col("anull").cast(pl.Int8))
            nd.write_parquet(out)
            del parts, nd
            progress.update("Preprocess", k + 1, len(jobs), msg=f"normalized {sp} source{i}: {df.height:,} rows")
            print(sp, i, df.height, flush=True)
    progress.update("Preprocess", len(jobs), len(jobs), state="done")


if __name__ == "__main__":
    main()
