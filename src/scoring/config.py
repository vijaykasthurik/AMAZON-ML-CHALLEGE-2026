"""Central paths and constants for the Business Entity Resolution pipeline.

CACHE  : read-only inputs (raw parquet, normalized records, ground truth, dictionary)
WORK   : everything the pipeline writes (candidates, features, OOF, test scores)
Both can be redirected with BER_CACHE / BER_WORK (used on Kaggle and for smoke tests).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get(
    "BER_DATA_DIR",
    os.path.join(os.path.dirname(ROOT), "DATASET THEY GIVEN", "student_resource", "dataset"),
)
CACHE = os.environ.get("BER_CACHE", os.path.join(ROOT, "cache"))
_default_work = os.path.join(ROOT, "cache_smoke" if os.environ.get("BER_SMOKE") == "1" else "cache")
WORK = os.environ.get("BER_WORK", _default_work)
OUTPUTS = os.environ.get("BER_OUTPUTS", os.path.join(ROOT, "outputs"))
OUTPUT = os.environ.get("BER_OUTPUT", os.path.join(ROOT, "output"))
SEED = 42

for _d in (WORK, OUTPUTS, OUTPUT):
    os.makedirs(_d, exist_ok=True)


def src_path(split: str, name: str) -> str:
    """Return the path of a raw TSV, e.g. src_path('train', 'source1')."""
    return os.path.join(DATA_DIR, split, f"{split}_{name}.tsv")
