"""Tiny progress reporter: every pipeline stage writes outputs/status.json,
which docs/progress.html (served by serve_progress.py) renders live."""
import json
import os
import time
from config import OUTPUTS

STATUS = os.path.join(OUTPUTS, "status.json")
STAGES = ["Preprocess", "Blocking (train)", "Blocking (test)", "Features (train)",
          "Train model", "Features + scoring (test)", "Write submission"]


def _load():
    """Read the current status file or create the default one."""
    try:
        with open(STATUS, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"stages": {s: {"state": "waiting", "done": 0, "total": 0} for s in STAGES},
                "log": [], "metrics": {}}


def _save(st):
    """Atomically write the status file."""
    st["updated"] = time.strftime("%d/%m/%Y, %I:%M:%S %p")
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATUS)


def update(stage, done=None, total=None, state="running", msg=None):
    """Record progress of `stage` (done/total units) and optionally append a log line."""
    st = _load()
    s = st["stages"].setdefault(stage, {"state": "waiting", "done": 0, "total": 0})
    if done is not None:
        s["done"] = done
    if total is not None:
        s["total"] = total
    s["state"] = state
    if msg:
        st["log"] = (st.get("log", []) + [time.strftime("%H:%M:%S ") + msg])[-12:]
    _save(st)


def metric(name, value):
    """Record a headline metric (e.g. validation F0.5) shown on the page."""
    st = _load()
    st.setdefault("metrics", {})[name] = value
    _save(st)
