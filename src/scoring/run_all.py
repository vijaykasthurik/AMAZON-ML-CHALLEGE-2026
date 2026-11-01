"""One command end-to-end: blocking -> features -> training -> test inference -> submission files.

    python run_all.py            # full pipeline (expects normalized parquet from preprocess.py)
    python run_all.py --from features_test   # resume from a stage
Each stage skips work whose outputs already exist, so a resumed run is cheap.
"""
import argparse
import os
import runpy
import sys
import time

STAGES = ["block_train", "eval_blocking", "cheap_train", "prune0", "features_train", "train",
          "block_test", "cheap_test", "features_test", "infer", "export_train"]


def stage(name):
    """Run one pipeline stage in-process."""
    t0 = time.time()
    print(f"==== {name} ====", flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    if name.startswith("block_"):
        import block
        split = name.split("_")[1]
        from config import WORK
        if not os.path.exists(os.path.join(WORK, f"{split}_cand.parquet")):
            block.run(split)
    elif name.startswith("cheap_"):
        import features
        features.cheap_run(name.split("_")[1])
    elif name == "prune0":
        import features
        features.fit_prune0()
    elif name.startswith("features_"):
        import features
        features.run(name.split("_")[1])
    elif name == "eval_blocking":
        runpy.run_path(os.path.join(here, "eval_blocking.py"), run_name="__main__")
    elif name == "train":
        import train
        train.main()
    elif name == "export_train":
        # CPU-prep notebook: publish train features + tier-0 pruner for a GPU-only training notebook
        import glob, shutil
        from config import WORK, OUTPUTS
        dst = os.path.join(OUTPUTS, "feat")
        os.makedirs(dst, exist_ok=True)
        for p in glob.glob(os.path.join(WORK, "train_feat_*_*.parquet")):
            shutil.move(p, dst)
        print("exported", len(os.listdir(dst)), "feature files", flush=True)
    elif name == "infer":
        import infer
        infer.main()
    print(f"==== {name} done in {time.time() - t0:.0f}s ====", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STAGES[0], choices=STAGES)
    ap.add_argument("--only", default=None, choices=STAGES)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    todo = [a.only] if a.only else STAGES[STAGES.index(a.start):]
    for s in todo:
        stage(s)
