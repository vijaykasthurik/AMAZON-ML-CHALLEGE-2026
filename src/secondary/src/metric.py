"""
Evaluation metric for Amazon ML Challenge 2026: Business Entity Resolution.
F_beta (beta = 0.5) macro-averaged across all Source 1 entities.
"""

from typing import Dict, List, Set, Union, Optional
import numpy as np


def compute_f05_single(pred_set: Set[str], true_set: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.

    Parameters
    ----------
    pred_set : Set[str]
        Set of predicted matching entity IDs (S2 and/or S3).
    true_set : Set[str]
        Set of ground-truth matching entity IDs (S2 and/or S3).

    Returns
    -------
    float
        F_0.5 score in [0.0, 1.0].
    """
    # Case 1: Singleton (true matches is empty)
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Case 2: Missed all matches (predicted matches is empty)
    if len(pred_set) == 0:
        return 0.0

    # Case 3: Both non-empty
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    # F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    # Note: 1 + beta^2 = 1 + 0.25 = 1.25; beta^2 = 0.25
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_f05(
    predictions: Dict[str, Union[Set[str], List[str], str]],
    ground_truth: Dict[str, Union[Set[str], List[str], str]],
    all_source1_ids: Optional[List[str]] = None,
) -> float:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities.

    Parameters
    ----------
    predictions : Dict[str, Union[Set[str], List[str], str]]
        Mapping from source1_entity_id to predicted IDs (as set, list, or comma-separated str).
    ground_truth : Dict[str, Union[Set[str], List[str], str]]
        Mapping from source1_entity_id to ground truth IDs (as set, list, or comma-separated str).
    all_source1_ids : Optional[List[str]]
        Explicit list of all required Source 1 entities. If None, uses keys of ground_truth.

    Returns
    -------
    float
        Macro-averaged F_0.5 score in [0.0, 1.0].
    """
    if all_source1_ids is None:
        eval_ids = list(ground_truth.keys())
    else:
        eval_ids = all_source1_ids

    if not eval_ids:
        return 0.0

    total_score = 0.0

    def parse_ids(val: Union[Set[str], List[str], str, None]) -> Set[str]:
        if val is None:
            return set()
        if isinstance(val, set):
            return val
        if isinstance(val, list):
            return set(val)
        if isinstance(val, str):
            s = val.strip()
            return {x.strip() for x in s.split(",") if x.strip()} if s else set()
        return set()

    for s1_id in eval_ids:
        true_set = parse_ids(ground_truth.get(s1_id))
        pred_set = parse_ids(predictions.get(s1_id))
        score = compute_f05_single(pred_set, true_set)
        total_score += score

    return total_score / len(eval_ids)


if __name__ == "__main__":
    # Self-test / unit-test
    # 1. Official example test
    p_ex = {"S2-00047", "S2-00193", "S3-00812"}
    t_ex = {"S2-00047", "S3-00812"}
    score_ex = compute_f05_single(p_ex, t_ex)
    assert abs(score_ex - 0.7142857) < 1e-4, f"Official example failed: {score_ex}"

    # 2. Perfect match
    assert compute_f05_single({"S2-1"}, {"S2-1"}) == 1.0

    # 3. Singleton correct
    assert compute_f05_single(set(), set()) == 1.0

    # 4. Singleton false positive (predicted something when true was empty)
    assert compute_f05_single({"S2-1"}, set()) == 0.0

    # 5. True match missed (predicted nothing when true had matches)
    assert compute_f05_single(set(), {"S2-1"}) == 0.0

    # 6. Completely disjoint predictions
    assert compute_f05_single({"S2-1"}, {"S2-2"}) == 0.0

    print("All metric unit tests passed successfully!")
