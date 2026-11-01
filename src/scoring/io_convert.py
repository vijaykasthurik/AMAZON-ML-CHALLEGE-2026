"""Convert the raw TSV files into parquet once, so later stages load in seconds.

Quoting is disabled: names/addresses can contain stray quote characters and the
files are plain tab-separated with no quoting convention.
"""
import os
import polars as pl
from config import CACHE, src_path


def read_tsv(path: str) -> pl.DataFrame:
    """Read a challenge TSV with every column as string and no quote handling."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0,
                       missing_utf8_is_empty_string=False)


def main():
    """Convert all train/test source files and the ground truth to parquet."""
    for split in ("train", "test"):
        for name in ("source1", "source2", "source3") + (("ground_truth",) if split == "train" else ()):
            out = os.path.join(CACHE, f"{split}_{name}.parquet")
            if os.path.exists(out):
                continue
            df = read_tsv(src_path(split, name))
            df.write_parquet(out)
            print(split, name, df.shape, flush=True)


if __name__ == "__main__":
    main()
