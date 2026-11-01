"""Official metric: macro-averaged per-Source-1-entity F0.5.

For every Source 1 entity: if truth and prediction are both empty -> 1.0;
if exactly one of them is empty -> 0.0; otherwise F0.5 from set precision/recall.
"""
from typing import Dict, Iterable, Set

BETA2 = 0.25


def f05(pred: Set[str], true: Set[str]) -> float:
    """F0.5 for one Source 1 entity given predicted and true match sets."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Dict[str, Iterable[str]], truth: Dict[str, Iterable[str]]) -> float:
    """Average F0.5 over every entity in `truth`; entities missing from `pred` count as empty."""
    tot = 0.0
    for k, t in truth.items():
        tot += f05(set(pred.get(k, ())), set(t))
    return tot / max(len(truth), 1)


if __name__ == "__main__":
    # README worked example: P=2/3, R=1 -> 0.714
    assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
    assert f05(set(), set()) == 1.0            # correct singleton
    assert f05({"a"}, set()) == 0.0            # false merge on a singleton
    assert f05(set(), {"a"}) == 0.0            # missed everything
    assert f05({"x"}, {"a"}) == 0.0            # all wrong
    assert f05({"a", "a"}, {"a"}) == 1.0       # duplicates collapse in a set
    assert macro_f05({"1": ["a"]}, {"1": ["a"], "2": []}) == 1.0
    assert macro_f05({}, {"1": ["a"], "2": []}) == 0.5
    print("metric tests passed")
