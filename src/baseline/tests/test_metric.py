"""Unit tests for src/metric.py (run: python tests/test_metric.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metric import fbeta_single, macro_fbeta, parse_id_list

def close(a, b):
    return abs(a - b) < 1e-3

# worked example from the problem statement: P=2/3, R=1 -> 0.714
assert close(fbeta_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 0.714)
# singleton rules
assert fbeta_single(set(), set()) == 1.0
assert fbeta_single({"S2-1"}, set()) == 0.0
assert fbeta_single(set(), {"S2-1"}) == 0.0
# perfect / no overlap
assert fbeta_single({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
assert fbeta_single({"S2-9"}, {"S2-1"}) == 0.0
# precision is weighted more than recall: P=1,R=0.5 beats P=0.5,R=1
assert fbeta_single({"S2-1"}, {"S2-1", "S2-2"}) > fbeta_single({"S2-1", "S2-3"}, {"S2-1"})
# parsing edge cases: empty, NaN, duplicates, spaces
assert parse_id_list("") == set() and parse_id_list(None) == set() and parse_id_list(float("nan")) == set()
assert parse_id_list("S2-1,S2-1, S3-2") == {"S2-1", "S3-2"}
# macro average with a missing prediction counts as empty
true = {"a": {"S2-1"}, "b": set(), "c": {"S3-1", "S3-2"}}
pred = {"a": {"S2-1"}, "c": {"S3-1"}}
exp = (1.0 + 1.0 + fbeta_single({"S3-1"}, {"S3-1", "S3-2"})) / 3
assert close(macro_fbeta(pred, true), exp)
# empty-everything prediction scores exactly the singleton rate
assert close(macro_fbeta({}, true), 1 / 3)
print("metric tests passed")
