"""Build the final team ensemble and its audited candidate set.

The four inputs are outputs produced by the component pipelines included with
the submission: the baseline prediction, secondary prediction, and the v11/v12
pair-score tables. France is copied from the baseline without modification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def read_lists(path: str, value_col: str) -> tuple[list[str], dict[str, set[str]]]:
    frame = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    order = frame["source1_entity_id"].to_list()
    values = frame[value_col].fill_null("").to_list()
    return order, {key: (set(value.split(",")) if value else set())
                   for key, value in zip(order, values)}


def read_scores(path: str, score_col: str) -> dict[tuple[str, str], float]:
    frame = pl.read_parquet(path, columns=["id1", "id2", score_col])
    return {(a, b): float(score) for a, b, score in frame.iter_rows()
            if score is not None}


def read_countries(test_source1: str) -> dict[str, str]:
    frame = pl.read_csv(test_source1, separator="\t", columns=["entity_id", "country"])
    return dict(frame.iter_rows())


def write_lists(path: Path, order: list[str], values: dict[str, set[str]], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"source1_entity_id\t{column}\n")
        for source1_id in order:
            out.write(source1_id + "\t" + ",".join(sorted(values[source1_id])) + "\n")


def build(args: argparse.Namespace) -> None:
    order, baseline = read_lists(args.baseline, "matched_entity_ids")
    order2, secondary = read_lists(args.secondary, "matched_entity_ids")
    if order != order2:
        raise ValueError("Baseline and secondary files have different Source 1 rows/order")
    countries = read_countries(args.test_source1)
    v11 = read_scores(args.v11_scores, args.score_column)
    v12 = read_scores(args.v12_scores, args.score_column)
    final = {key: set(value) for key, value in baseline.items()}

    # Conservative removals apply only to the two trained countries.
    for source1_id in order:
        if countries[source1_id] not in {"US", "India"}:
            continue
        for target_id in list(final[source1_id]):
            s11 = v11.get((source1_id, target_id))
            s12 = v12.get((source1_id, target_id))
            if s11 is not None and s11 < args.v11_remove:
                final[source1_id].remove(target_id)
                continue
            if target_id not in secondary[source1_id] and s11 is not None and s12 is not None:
                if (s11 + s12) / 2.0 < args.blend_remove:
                    final[source1_id].remove(target_id)

    # Add only pairs accepted independently by both score variants. If a target
    # is proposed for more than one Source 1 row, the strongest joint score owns it.
    proposals: dict[str, tuple[float, str]] = {}
    for source1_id in order:
        if countries[source1_id] not in {"US", "India"}:
            continue
        for target_id in secondary[source1_id] - final[source1_id]:
            s11 = v11.get((source1_id, target_id))
            s12 = v12.get((source1_id, target_id))
            if s11 is None or s12 is None or s11 < args.add_threshold or s12 < args.add_threshold:
                continue
            strength = min(s11, s12)
            previous = proposals.get(target_id)
            if previous is None or (strength, source1_id) > previous:
                proposals[target_id] = (strength, source1_id)
    current_owners: dict[str, set[str]] = {}
    for owner, target_ids in final.items():
        for target_id in target_ids:
            current_owners.setdefault(target_id, set()).add(owner)
    for target_id, (_, source1_id) in proposals.items():
        for owner in current_owners.get(target_id, set()):
            if owner != source1_id:
                final[owner].discard(target_id)
        final[source1_id].add(target_id)

    output_dir = Path(args.output_dir)
    matching_path = output_dir / "matching_results.tsv"
    write_lists(matching_path, order, final, "matched_entity_ids")

    # The audited final candidate set is the exact union of every v11-scored pair
    # and every selected final match. This also retains secondary additions.
    candidates = {key: set() for key in order}
    for source1_id, target_id in v11:
        candidates[source1_id].add(target_id)
    for source1_id in order:
        candidates[source1_id].update(final[source1_id])
    write_lists(output_dir / "candidate_pairs.tsv", order, candidates, "candidate_entity_ids")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--v11-scores", required=True)
    ap.add_argument("--v12-scores", required=True)
    ap.add_argument("--test-source1", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--score-column", default="p2ce")
    ap.add_argument("--v11-remove", type=float, default=0.05)
    ap.add_argument("--blend-remove", type=float, default=0.30)
    ap.add_argument("--add-threshold", type=float, default=0.95)
    return ap


if __name__ == "__main__":
    build(parser().parse_args())
