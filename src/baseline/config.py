"""Central configuration: paths, seeds and a few global knobs.

Every script imports paths from here so the whole pipeline can be relocated by
editing (or overriding through environment variables) just this file.
"""

import os

# Folder that contains dataset/train and dataset/test (the unzipped student_resource).
DATA_DIR = os.environ.get("BER_DATA_DIR", "D:/aws/datasets/student_resource/dataset")
# Scratch folder for cached parquet / numpy artefacts (can get to a few GB).
WORK_DIR = os.environ.get("BER_WORK_DIR", "D:/aws/work")
# Where the two submission files are written.
OUTPUT_DIR = os.environ.get("BER_OUTPUT_DIR", os.path.join(WORK_DIR, "output"))

CACHE_DIR = os.path.join(WORK_DIR, "cache")
MODEL_DIR = os.path.join(WORK_DIR, "models")
LOG_DIR = os.path.join(WORK_DIR, "logs")

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 2)  # leave two cores for the OS

for _d in (WORK_DIR, OUTPUT_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)


def source_path(split: str, source: int) -> str:
    """Path of the raw TSV for ``split`` in {'train','test'} and source in {1,2,3}."""
    return os.path.join(DATA_DIR, split, f"{split}_source{source}.tsv")


def gt_path() -> str:
    """Path of the training ground truth TSV."""
    return os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
