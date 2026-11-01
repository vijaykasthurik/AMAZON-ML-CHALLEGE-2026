"""Cross-encoder re-scorer (cross-encoder/ms-marco-MiniLM-L6-v2, Apache-2.0, 22M params).

The transformer reads both records jointly ("name ; address ; numbers" for each side)
and is fine-tuned on training pairs from a DEDICATED slice of Source 1 entities
(hash buckets CE_LO..CE_HI-1) that the XGBoost models never fit on, so its scores are
out-of-sample for every row XGBoost trains or is evaluated on.

Only uncertain pairs (stage-1 probability between U_LO and U_HI) are scored; the rest
get NaN, which XGBoost treats as missing. This keeps GPU time to minutes.
"""
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import OUTPUTS, SEED
from data import load_norm

BASE = os.environ.get("BER_CE_BASE", "cross-encoder/ms-marco-MiniLM-L6-v2")
CE_DIR = os.environ.get("BER_CE_DIR", os.path.join(os.environ.get("BER_MODELS", os.path.join(OUTPUTS, "models")), "ce"))
CE_LO, CE_HI = 30, 50
U_LO, U_HI = 0.002, 0.998
MAXLEN = 96
EPOCHS = int(os.environ.get("BER_CE_EPOCHS", "1"))
BS = int(os.environ.get("BER_CE_BS", "128"))
LR = float(os.environ.get("BER_CE_LR", "3e-5"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Fp16(torch.nn.Module):
    """Runs the wrapped model under fp16 autocast inside forward, so every DataParallel
    replica thread gets it (autocast state is thread-local)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **enc):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            return self.m(**enc).logits.squeeze(-1).float()


def _parallel(model):
    """fp16 wrapper, spread over every visible GPU (Kaggle gives two T4s). Returns logits of shape (batch,)."""
    w = _Fp16(model)
    if DEV == "cuda" and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(w)
    return w


def texts(split: str) -> pl.DataFrame:
    """id -> compact normalized text used by the cross-encoder."""
    d = load_norm(split, columns=["id", "nf", "at", "num", "state"])
    return d.select("id", pl.concat_str([pl.col("nf"), pl.lit(" ; "), pl.col("at").fill_null(""), pl.lit(" ; "),
                                         pl.col("num").fill_null(""), pl.lit(" "), pl.col("state").fill_null("")])
                    .alias("t"))


def attach_text(pairs: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    """Add t1 / t2 text columns to a (id1, id2, ...) frame, preserving row order."""
    return (pairs.with_row_index("_r")
                 .join(tx.rename({"id": "id1", "t": "t1"}), on="id1", how="left")
                 .join(tx.rename({"id": "id2", "t": "t2"}), on="id2", how="left")
                 .sort("_r").drop("_r"))


def _batches(t1, t2, y, tok, bs, shuffle):
    """Tokenized mini-batches."""
    idx = np.arange(len(t1))
    if shuffle:
        np.random.default_rng(SEED).shuffle(idx)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        enc = tok([t1[k] for k in j], [t2[k] for k in j], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        yield enc, (torch.tensor(y[j], dtype=torch.float32) if y is not None else None)


def train_ce(pairs: pl.DataFrame, tx: pl.DataFrame, epochs=EPOCHS, bs=BS, lr=LR, log=print):
    """Fine-tune the cross-encoder on (id1, id2, y) pairs; saves to CE_DIR."""
    torch.manual_seed(SEED)
    d = attach_text(pairs, tx)
    t1, t2, y = d["t1"].to_list(), d["t2"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, ignore_mismatched_sizes=True).to(DEV)
    model = _fit(_parallel(base), tok, t1, t2, y, epochs, bs, lr, log)
    os.makedirs(CE_DIR, exist_ok=True)
    base.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)
    return model, tok


def _fit(model, tok, t1, t2, y, epochs, bs, lr, log):
    """Fine-tune `model` (a _parallel wrapper) in place on aligned text pairs; returns it in eval mode."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(t1) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        for enc, yy in _batches(t1, t2, y, tok, bs, True):
            enc = {k: v.to(DEV) for k, v in enc.items()}
            loss = lossf(model(**enc), yy.to(DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log(f"ce step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval()
    return model


def load_ce():
    """Load the fine-tuned cross-encoder."""
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(DEV).eval()
    return _parallel(model), tok


@torch.no_grad()
def score(model, tok, t1, t2, bs=512):
    """Match probability for aligned text lists."""
    out = []
    for enc, _ in _batches(t1, t2, None, tok, bs, False):
        enc = {k: v.to(DEV) for k, v in enc.items()}
        out.append(torch.sigmoid(model(**enc)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def add_ce(full: pl.DataFrame, split: str, model=None, tok=None) -> pl.DataFrame:
    """Add column `ce` (NaN for confident pairs) to a frame with id1, id2, p1."""
    if model is None:
        model, tok = load_ce()
    unc = full.with_row_index("_row").filter((pl.col("p1") > U_LO) & (pl.col("p1") < U_HI)).select("_row", "id1", "id2")
    tx = texts(split)
    d = attach_text(unc, tx)
    t0 = time.time()
    s = score(model, tok, d["t1"].to_list(), d["t2"].to_list())
    print(f"ce scored {len(s):,} uncertain pairs of {full.height:,} in {time.time() - t0:.0f}s", flush=True)
    ce = np.full(full.height, np.nan, dtype=np.float32)
    ce[d["_row"].to_numpy()] = s
    return full.with_columns(pl.Series("ce", ce))


_LEGAL = {"sarl", "sas", "eurl", "sa", "sci", "snc", "sasu", "ei", "inc", "llc", "ltd", "corp", "co", "pvt", "private",
          "limited", "llp", "pc", "pllc", "lp"}


def _typo(w, r):
    """One random character edit inside a token."""
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.random()
    if op < 0.3:
        return w[:i] + w[i + 1:]
    if op < 0.6:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + r.choice("aeioulnrst") + w[i + 1:]


def _noisy(nf, at, num, st, vocab, r):
    """Noisy copy of a normalized record (rates follow noise_stats.py on training links)."""
    ws = nf.split()
    if ws and ws[-1] in _LEGAL and r.random() < 0.17:
        ws = ws[:-1]
    elif r.random() < 0.07:
        ws.append(r.choice(sorted(_LEGAL)))
    if len(ws) > 1 and r.random() < 0.06:
        r.shuffle(ws)
    if ws and r.random() < 0.18:
        j = r.randrange(len(ws)); ws[j] = _typo(ws[j], r)
    if r.random() < 0.10:
        ws.insert(r.randrange(len(ws) + 1), r.choice(vocab))
    if len(ws) > 2 and r.random() < 0.06:
        ws.pop(r.randrange(len(ws)))
    name = "".join(ws) + " com" if (ws and r.random() < 0.06) else " ".join(ws)
    if r.random() < 0.05:
        return name, "", "", ""
    ta = at.split()
    if len(ta) > 2 and r.random() < 0.2:
        ta.pop(r.randrange(len(ta)))
    if ta and r.random() < 0.1:
        j = r.randrange(len(ta)); ta[j] = _typo(ta[j], r)
    if len(ta) > 1 and r.random() < 0.03:
        r.shuffle(ta)
    nums = num.split()
    if nums and r.random() < 0.05:
        nums = nums[1:]
    elif nums and r.random() < 0.04:
        nums[0] = _typo(nums[0] + "0", r)[:len(nums[0])]
    return name, " ".join(ta), " ".join(nums), (st if r.random() > 0.5 else "")


def adapt_pairs(split: str, country: str, n: int, seed=SEED) -> pl.DataFrame:
    """Labelled text pairs for a country without training labels, derived only from that split's own
    normalized Source 1 records: noisy copies (positives); same name at another house number, and another
    name at the same address (negatives); and 'twins' - a noisy copy of S1 record B paired with S1 record A
    when A and B share core name and house number (the hardest real negatives)."""
    import random
    r = random.Random(seed)
    s1 = load_norm(split, sources=(1,), columns=["id", "country", "nf", "nc", "at", "num", "state"])         .filter(pl.col("country") == country).with_columns(pl.col("at").fill_null(""), pl.col("num").fill_null(""),
                                                           pl.col("state").fill_null(""))
    vocab = s1["nc"].str.split(" ").explode().value_counts().sort("count", descending=True).head(200)["nc"].to_list()
    names = s1["nf"].to_list()
    twins = s1.filter(pl.col("num") != "").with_columns(pl.len().over(["nc", "num"]).alias("k")).filter(pl.col("k") > 1)
    tw = twins.select(pl.col("id").alias("a"), "nc", "num").join(twins.select(pl.col("id").alias("b"), "nc", "num"),
                                                                  on=["nc", "num"]).filter(pl.col("a") != pl.col("b"))
    twin_of = dict(zip(tw["b"].to_list(), tw["a"].to_list()))
    rows = []
    fmt = lambda nf, at, num, st: f"{nf} ; {at} ; {num} {st}"
    for i, nf, at, num, st in s1.sample(min(n, s1.height), seed=seed).select("id", "nf", "at", "num", "state").iter_rows():
        t1 = fmt(nf, at, num, st)
        for _ in range(r.choice([1, 1, 2])):
            rows.append((i, t1, fmt(*_noisy(nf, at, num, st, vocab, r)), 1))
        if num and r.random() < 0.3:
            n0 = num.split()[0]
            other = str(int(n0) + r.choice([-7, -3, -1, 1, 2, 5, 11])) if n0.isdigit() and int(n0) > 11 else n0 + "1"
            neg = _noisy(nf, at, num.replace(n0, other, 1), st, vocab, r)
            if other in neg[2].split():  # the distinguishing number must survive the noise, else it is not a negative
                rows.append((i, t1, fmt(*neg), 0))
        if r.random() < 0.3:
            rows.append((i, t1, fmt(*_noisy(r.choice(names), at, num, st, vocab, r)), 0))
        if i in twin_of:
            rows.append((twin_of[i], None, fmt(*_noisy(nf, at, num, st, vocab, r)), 0))
    d = pl.DataFrame(rows, schema=["id1", "t1", "t2", "y"], orient="row")
    tx = s1.select(pl.col("id").alias("id1"), pl.concat_str([pl.col("nf"), pl.lit(" ; "), pl.col("at"), pl.lit(" ; "),
                                                            pl.col("num"), pl.lit(" "), pl.col("state")]).alias("tx"))
    d = d.join(tx, on="id1", how="left").with_columns(pl.coalesce("t1", "tx").alias("t1")).drop("tx")
    return d.with_columns(pl.col("y").cast(pl.Int32)).sample(fraction=1.0, shuffle=True, seed=seed)


def adapt_country(split: str, country: str, n=int(os.environ.get("BER_ADAPT_N", "40000")), log=print):
    """Copy of the fitted cross-encoder, further fine-tuned on adapt_pairs() for one country.
    A held-out 10% of the synthetic pairs is scored before and after as a sanity check."""
    d = adapt_pairs(split, country, n)
    k = d.height // 10
    ho, tr = d.head(k), d.tail(d.height - k)
    model, tok = load_ce()

    def acc(m):
        s = score(m, tok, ho["t1"].to_list(), ho["t2"].to_list())
        return float(((s > 0.5) == (ho["y"].to_numpy() == 1)).mean())
    before = acc(model)
    model = _fit(model, tok, tr["t1"].to_list(), tr["t2"].to_list(), tr["y"].to_numpy().astype(np.float32),
                 1, BS, float(os.environ.get("BER_ADAPT_LR", "1e-5")), log)
    log(f"adapt {country}: {tr.height:,} synthetic pairs (pos {int(tr['y'].sum()):,}); "
        f"held-out synthetic accuracy {before:.4f} -> {acc(model):.4f}")
    return model, tok


def fit_from_features(files, fold_expr, max_pairs=int(os.environ.get("BER_CE_PAIRS", "3000000")), log=print):
    """Collect CE training pairs from the dedicated entity slice and fine-tune."""
    parts = []
    for p in files:
        d = pl.read_parquet(p, columns=["id1", "id2", "y", "rk1_blk", "tsr_nc"]).with_columns(fold_expr())
        d = d.filter((pl.col("bucket") >= CE_LO) & (pl.col("bucket") < CE_HI))
        # all positives + hard negatives (well-ranked or name-similar non-matches)
        d = d.filter((pl.col("y") == 1) | (pl.col("rk1_blk") <= 8) | (pl.col("tsr_nc") >= 0.6))
        parts.append(d.select("id1", "id2", "y"))
    tr = pl.concat(parts)
    if tr.height > max_pairs:
        tr = tr.sample(max_pairs, seed=SEED)
    log(f"ce train pairs {tr.height:,} pos {int(tr['y'].sum()):,}")
    return train_ce(tr, texts("train"), log=log)
