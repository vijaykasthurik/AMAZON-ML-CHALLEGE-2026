"""Process-wide environment settings; import this before numpy / polars.

Worker processes on Windows are *spawned* and re-import the main script, so
every worker would otherwise load OpenBLAS with one buffer per CPU thread
(about 1 GB of committed memory per worker). The pipeline never uses BLAS, so
it is pinned to one thread.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
