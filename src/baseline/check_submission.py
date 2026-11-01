"""Pre-upload checks for the two output files.

Runs the organisers' validator (utils/validate_submission.py, with the optional
ID-existence check switched on) and a few extra checks of our own:
  * exact header strings and a single tab per line
  * one row per test Source 1 entity, nothing missing or extra
  * every matched id is also a candidate id for the same entity
  * match statistics (share of empty rows, mean list length) for a sanity look

Usage: python check_submission.py <output_dir> [<student_resource_dir>]
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def read_lists(path: str, col: str) -> dict:
    """Read a results TSV into {source1_id: [ids]} while checking its layout."""
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == f"source1_entity_id\t{col}", f"bad header in {path}: {header!r}"
        for n, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            assert line.count("\t") == 1, f"{path}:{n} has {line.count(chr(9))} tabs"
            s1, rest = line.split("\t")
            assert s1 not in out, f"duplicate row {s1}"
            ids = rest.split(",") if rest else []
            assert len(ids) == len(set(ids)), f"duplicate ids in row {s1}"
            assert all(i.startswith(("S2-", "S3-")) for i in ids), f"bad prefix in row {s1}"
            out[s1] = ids
    return out


def main(out_dir: str, resource_dir: str):
    """Run every check; exit 1 on any failure."""
    match = os.path.join(out_dir, "matching_results.tsv")
    cand = os.path.join(out_dir, "candidate_pairs.tsv")
    test_dir = os.path.join(config.DATA_DIR, "test")

    m = read_lists(match, "matched_entity_ids")
    c = read_lists(cand, "candidate_entity_ids")
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        next(f)
        required = [line.split("\t", 1)[0] for line in f if line.strip()]
    req = set(required)
    assert set(m) == req, f"matching rows differ from test S1 ({len(set(m) ^ req)} ids)"
    assert set(c) == req, f"candidate rows differ from test S1 ({len(set(c) ^ req)} ids)"
    not_sub = sum(1 for k, v in m.items() if not set(v) <= set(c[k]))
    assert not_sub == 0, f"{not_sub} rows have matches outside their candidates"

    n_empty = sum(1 for v in m.values() if not v)
    n_ids = sum(len(v) for v in m.values())
    n_cand = sum(len(v) for v in c.values())
    print(f"rows={len(m)}  empty={n_empty} ({n_empty / len(m):.3%})  "
          f"matched ids={n_ids} (mean {n_ids / len(m):.2f})  candidate ids={n_cand} "
          f"(mean {n_cand / len(m):.2f})")

    validator = os.path.join(resource_dir, "utils", "validate_submission.py")
    r = subprocess.run([sys.executable, validator, "--matching", match, "--candidate", cand,
                        "--test-dir", test_dir, "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else config.OUTPUT_DIR
    res = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(config.DATA_DIR)
    main(out_dir, res)
