"""Evaluation metric for the Business Entity Resolution challenge.

The official score is a macro-averaged F-beta with beta = 0.5. For every Source 1
entity we compare the predicted set of S2/S3 ids with the true set, compute
F0.5 for that entity, and then average over *all* Source 1 entities.

Special cases (taken from the problem statement):
  * true set empty and predicted set empty  -> 1.0 (correct singleton)
  * true set empty and predicted set not    -> 0.0 (false merge on a singleton)
  * true set non-empty and predicted empty  -> 0.0 (precision undefined, recall 0)
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

BETA = 0.5


def parse_id_list(value) -> Set[str]:
    """Turn a comma separated id string (possibly empty / NaN / None) into a set."""
    if value is None:
        return set()
    if isinstance(value, float):  # pandas NaN
        return set()
    value = str(value).strip()
    if not value:
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def fbeta_single(pred: Set[str], true: Set[str], beta: float = BETA) -> float:
    """F-beta for one Source 1 entity, with the singleton rules applied."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_fbeta(pred: Dict[str, Set[str]], true: Dict[str, Set[str]],
                beta: float = BETA, ids: Iterable[str] | None = None) -> float:
    """Macro F-beta over the Source 1 ids in ``ids`` (default: every id in ``true``).

    Ids missing from ``pred`` are treated as an empty prediction.
    """
    ids = list(true.keys()) if ids is None else list(ids)
    if not ids:
        return 0.0
    total = 0.0
    for s1 in ids:
        total += fbeta_single(pred.get(s1, set()), true.get(s1, set()), beta)
    return total / len(ids)


def score_files(pred_path: str, true_path: str) -> float:
    """Score a matching_results.tsv against a ground-truth tsv (same format)."""
    def load(path, col):
        out = {}
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            assert header[0] == "source1_entity_id" and header[1] == col, header
            for line in f:
                s1, _, rest = line.rstrip("\n").partition("\t")
                if s1:
                    out[s1] = parse_id_list(rest)
        return out

    true = load(true_path, "matched_entity_ids")
    pred = load(pred_path, "matched_entity_ids")
    return macro_fbeta(pred, true)


if __name__ == "__main__":
    import sys
    print(f"macro F0.5 = {score_files(sys.argv[1], sys.argv[2]):.6f}")
