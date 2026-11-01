"""Pre-upload checks: official validator + distribution sanity vs training priors.

Usage: python check_submission.py [matching.tsv] [candidate.tsv]
"""
import os
import subprocess
import sys

import polars as pl

from config import DATA_DIR, OUTPUT

VALIDATOR = os.path.join(os.path.dirname(DATA_DIR), "utils", "validate_submission.py")


def stats(path):
    """Row count, predicted-singleton rate and mean list length of a submission file."""
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    k = d[d.columns[1]].fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    print(os.path.basename(path), "rows", d.height, "unique ids", d[d.columns[0]].n_unique(),
          "empty rate", round(float((k == 0).mean()), 4), "mean list len", round(float(k.mean()), 3),
          "max", int(k.max()))


def main():
    """Run the official validator and print stats; exit non-zero on failure."""
    m = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT, "matching_results.tsv")
    c = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUTPUT, "candidate_pairs.tsv")
    stats(m)
    stats(c)
    print("training prior: empty rate 0.0558, mean true list len 3.46")
    r = subprocess.run([sys.executable, VALIDATOR, "--matching", m, "--candidate", c,
                        "--test-dir", os.path.join(DATA_DIR, "test")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
