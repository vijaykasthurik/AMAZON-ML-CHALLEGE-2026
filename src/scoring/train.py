"""Two-stage training with 5-fold out-of-fold predictions grouped by Source 1 entity.

Stage 1 (pair model): XGBoost (GPU) on pair features -> p1.
Stage 2 (collective model): p1 re-scored with context of all competing pairs
  (rank / gap / margin of p1 within the Source 1 group and within the target group)
  -> p2.
Then the decision layer (one-owner + expected-F0.5 / threshold) is tuned on the OOF
p2 using the official macro F0.5 over EVERY training Source 1 entity.
"""
import glob
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import OUTPUTS, SEED, WORK
from data import load_norm
from decide import expected_f, macro_f05, one_owner, singleton_guard, threshold
from features import load_truth
import progress

MODELS = os.environ.get("BER_MODELS", os.path.join(OUTPUTS, "models"))
os.makedirs(MODELS, exist_ok=True)
NF = 5
SUB = int(os.environ.get("BER_SUB", "30"))  # % of Source 1 entities used to fit models
HOLD = int(os.environ.get("BER_HOLD", "50"))  # buckets >= HOLD are the untouched holdout (30..49 are reserved for the cross-encoder)
# extra model-fitting buckets taken from the old holdout range: [50, HOLD) joins [0, SUB) when HOLD > 50
# buckets >= EVAL are scored for decision tuning; with HOLD=100 they are out-of-fold (5-fold CV), not untouched
EVAL = int(os.environ.get("BER_EVAL", str(min(HOLD, 50) if HOLD >= 100 else HOLD)))

USE_CE = os.environ.get("BER_CE", "0") == "1"
P1_MIN = float(os.environ.get("BER_P1MIN", "0.001"))
USE_SIB = os.environ.get("BER_SIB", "0") == "1"
ID_COLS = ["id1", "id2", "y"]
PARAMS = dict(tree_method="hist", device="cpu" if os.environ.get("BER_CPU") == "1" else "cuda", nthread=-1, objective="binary:logistic", eval_metric="logloss",
              max_depth=9, learning_rate=float(os.environ.get("BER_LR", "0.1")), subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              max_bin=256, reg_lambda=2.0, seed=SEED)


def fold_expr():
    """Deterministic fold id and subsample bucket from a hash of the Source 1 id."""
    h = pl.col("id1").hash(seed=11)
    return [(h % NF).cast(pl.Int8).alias("fold"), ((h // NF) % 100).cast(pl.Int8).alias("bucket")]


def feature_files(split):
    """All per-country / per-chunk feature parquet files of a split."""
    return sorted(glob.glob(os.path.join(WORK, f"{split}_feat_*_*.parquet")))


def feat_cols(path):
    """Numeric feature column names (everything except ids and label)."""
    return [c for c in pl.read_parquet_schema(path) if c not in ID_COLS]


def stage2_frame(p1: pl.DataFrame) -> pl.DataFrame:
    """Collective features on the full (id1, id2, p1, ...) table (windows materialized, no nesting)."""
    d = p1.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("id1").cast(pl.Float32).alias("q_rk1"),
        pl.col("p1").rank("ordinal", descending=True).over("id2").cast(pl.Float32).alias("q_rk2"),
        (pl.col("p1") - pl.col("p1").max().over("id2")).alias("q_gap2"),
        (pl.col("p1") - pl.col("p1").max().over("id1")).alias("q_gap1"),
        pl.col("p1").sum().over("id1").alias("q_sum1"),
        pl.col("p1").sum().over("id2").alias("q_sum2"),
        (pl.col("p1") > 0.5).sum().over("id1").cast(pl.Float32).alias("q_n1hi"),
        (pl.col("p1") > 0.5).sum().over("id2").cast(pl.Float32).alias("q_n2hi"),
        (pl.col("p1") > 0.1).sum().over("id1").cast(pl.Float32).alias("q_n1mid"),
        pl.col("p1").max().over(["id1", "src_b"]).alias("q_max_src"),
    )
    s2 = d.filter(pl.col("q_rk2") == 2).select("id2", pl.col("p1").alias("q_sec2"))
    s1 = d.filter(pl.col("q_rk1") == 2).select("id1", pl.col("p1").alias("q_sec1"))
    d = d.join(s2, on="id2", how="left").join(s1, on="id1", how="left")          .with_columns(pl.col("q_sec2").fill_null(0.0), pl.col("q_sec1").fill_null(0.0))
    return d.with_columns((pl.col("p1") - pl.col("q_sec2")).alias("q_marg2"),
                          (pl.col("p1") - pl.col("q_sec1")).alias("q_marg1"))


S2_EXTRA = ["s_blk", "cos_name", "cos_addr", "cos_num", "tsr_nc", "r_at", "num_jacc", "num_conflict",
            "src_b", "n1", "n2", "anull_b", "web_b", "nat_b"]


def fit_folds(X, y, fold, tag, rounds=int(os.environ.get("BER_ROUNDS", "3000"))):
    """Train NF fold models with early stopping; return list of boosters."""
    boosters = []
    for f in range(NF):
        tr, va = fold != f, fold == f
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])
        b = xgb.train(PARAMS, dtr, rounds, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
        b.save_model(os.path.join(MODELS, f"{tag}_f{f}.json"))
        boosters.append(b)
        print(tag, "fold", f, "best_iter", b.best_iteration, "logloss", b.best_score, flush=True)
        progress.update("Train model", f + 1 + (NF if tag == "s2" else 0), 2 * NF,
                        msg=f"{tag} fold {f}: best iter {b.best_iteration}, val logloss {b.best_score:.5f}")
        del dtr, dva
    return boosters


def predict_fold(boosters, X, fold):
    """Out-of-fold prediction: each row scored by the model that did not train on its fold."""
    out = np.zeros(len(X), dtype=np.float32)
    for f in range(NF):
        m = fold == f
        if m.any():
            out[m] = boosters[f].predict(xgb.DMatrix(X[m]), iteration_range=(0, boosters[f].best_iteration + 1))
    return out


def main():
    """Fit stage 1 and 2, produce OOF predictions and tune the decision rule."""
    t0 = time.time()
    files = feature_files("train")
    cols = feat_cols(files[0])
    json.dump(cols, open(os.path.join(MODELS, "stage1_features.json"), "w"))
    # ---- stage 1 fit on the subsample
    sub = pl.concat([pl.read_parquet(p).with_columns(fold_expr()).filter(((pl.col("bucket") < SUB) | ((pl.col("bucket") >= 50) & (pl.col("bucket") < HOLD)))) for p in files])
    print("stage1 train rows", sub.height, "pos", int(sub["y"].sum()), flush=True)
    X = sub.select(cols).to_numpy().astype(np.float32)
    b1 = fit_folds(X, sub["y"].to_numpy(), sub["fold"].to_numpy(), "s1")
    del sub, X
    # ---- stage 1 OOF over ALL candidate pairs
    parts = []
    for p in files:
        d = pl.read_parquet(p).with_columns(fold_expr())
        p1 = predict_fold(b1, d.select(cols).to_numpy().astype(np.float32), d["fold"].to_numpy())
        # learned pruning: stage 2 (and the final candidate set) only sees pairs with p1 >= P1_MIN
        parts.append(d.select(["id1", "id2", "y", "fold", "bucket"] + S2_EXTRA).with_columns(pl.Series("p1", p1))
                      .filter(pl.col("p1") >= P1_MIN))
        print("scored", p, flush=True)
    full = stage2_frame(pl.concat(parts))
    if USE_SIB:
        import sib
        full = sib.add_siblings(full, "train")
    del parts
    s2cols = [c for c in full.columns if c not in ("id1", "id2", "y", "fold", "bucket")]
    json.dump(s2cols, open(os.path.join(MODELS, "stage2_features.json"), "w"))
    # ---- stage 2 (without cross-encoder)
    tr = full.filter(((pl.col("bucket") < SUB) | ((pl.col("bucket") >= 50) & (pl.col("bucket") < HOLD))))
    b2 = fit_folds(tr.select(s2cols).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy(), "s2")
    full = full.with_columns(pl.Series("p2", predict_fold(b2, full.select(s2cols).to_numpy().astype(np.float32),
                                                          full["fold"].to_numpy())))
    keep = ["id1", "id2", "y", "fold", "bucket", "p1", "p2"]
    # ---- optional stage 2 WITH cross-encoder score (A/B on the same folds / holdout)
    if USE_CE:
        import ce
        model, tok = ce.fit_from_features(files, fold_expr, log=lambda m: print(m, flush=True))
        full = ce.add_ce(full, "train", model, tok)
        s2ce = s2cols + ["ce"]
        json.dump(s2ce, open(os.path.join(MODELS, "stage2ce_features.json"), "w"))
        tr = full.filter(((pl.col("bucket") < SUB) | ((pl.col("bucket") >= 50) & (pl.col("bucket") < HOLD))))
        b3 = fit_folds(tr.select(s2ce).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy(), "s2ce")
        full = full.with_columns(pl.Series("p2ce", predict_fold(b3, full.select(s2ce).to_numpy().astype(np.float32),
                                                                full["fold"].to_numpy())))
        keep.append("p2ce")
    full.select(keep).write_parquet(os.path.join(WORK, "train_oof.parquet"))
    full.select(keep).write_parquet(os.path.join(OUTPUTS, "train_oof.parquet"))
    progress.update("Train model", 2 * NF, 2 * NF, state="done", msg=f"training done in {time.time() - t0:.0f}s")
    tune(full)


def apply_rule(df, pred, rule, param, gate=0.0):
    """Apply a decision rule (from the grid) to a frame with id1, id2 and prediction column `pred`.

    `gate`: singleton guard — abstain on entities whose top candidate probability is below it."""
    base = df.select("id1", "id2", pl.col(pred).alias("p"))
    own = one_owner(base) if rule.startswith("own") else base
    sel = expected_f(own, alpha=param) if rule.endswith("expF") else threshold(own, thr=param)
    return singleton_guard(sel, base, gate=gate)


def report_best(best, f_s, t_s, s1):
    """Per-country and per-fold macro F0.5 of the chosen rule (fold std = noise level)."""
    sel = apply_rule(f_s, best[1], best[2], best[3])
    hold = s1.filter(pl.col("bucket") >= EVAL)
    for c in hold["country"].unique().sort().to_list():
        ids = hold.filter(pl.col("country") == c)["id"]
        print("  country", c, round(macro_f05(sel.join(pl.DataFrame({"id1": ids}), on="id1", how="semi"),
                                              t_s.join(pl.DataFrame({"id1": ids}), on="id1", how="semi"), ids), 6))
    fs = []
    for f in range(NF):
        ids = hold.filter(pl.col("fold") == f)["id"]
        fs.append(macro_f05(sel.join(pl.DataFrame({"id1": ids}), on="id1", how="semi"),
                            t_s.join(pl.DataFrame({"id1": ids}), on="id1", how="semi"), ids))
    print("  folds", [round(x, 6) for x in fs], "std", round(float(np.std(fs)), 6), flush=True)


def prune_report(full, best, truth, s1):
    """Candidate-set size vs recall / macro F0.5 for the learned pruning threshold on p1 (holdout)."""
    hold = s1.filter(pl.col("bucket") >= EVAL)
    ids = hold["id"]
    idf = pl.DataFrame({"id1": ids})
    f_s = full.join(idf, on="id1", how="semi")
    t_s = truth.join(idf, on="id1", how="semi")
    rows = []
    for tau in (P1_MIN, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05):
        kept = f_s.filter(pl.col("p1") >= tau)
        rec = kept.join(t_s, on=["id1", "id2"], how="semi").height / max(t_s.height, 1)
        f = macro_f05(apply_rule(kept, best[1], best[2], best[3]), t_s, ids)
        rows.append((tau, kept.height / len(ids), rec, f))
        print(f"  prune tau={tau}: cand/S1 {kept.height / len(ids):.2f}  pair recall {rec:.5f}  macroF05 {f:.6f}",
              flush=True)
    # largest tau with no macro-F0.5 loss (<= 0.00005) and <= 0.1 point of candidate-recall loss
    ok = [r for r in rows if r[3] >= rows[0][3] - 0.00005 and r[2] >= rows[0][2] - 0.001]
    tau = max(r[0] for r in ok)
    dec = json.load(open(os.path.join(MODELS, "decision.json")))
    dec["tau"] = tau
    json.dump(dec, open(os.path.join(MODELS, "decision.json"), "w"), indent=1)
    print("  chosen tau", tau, flush=True)


def tune(full=None):
    """Grid-search the decision rule on OOF predictions; evaluate on held-out (non-fit) entities too."""
    if full is None:
        full = pl.read_parquet(os.path.join(WORK, "train_oof.parquet"))
    truth = load_truth()
    s1 = load_norm("train", sources=(1,), columns=["id", "country"])
    s1 = s1.with_columns(pl.col("id").alias("id1")).with_columns(fold_expr())
    res = []
    # evaluate on entities NOT used for fitting (bucket >= SUB) -> honest; also on all
    for scope, ids in (("holdout", s1.filter(pl.col("bucket") >= EVAL)["id"]),):
        idf = pl.DataFrame({"id1": ids})
        f_s = full.join(idf, on="id1", how="semi")
        t_s = truth.join(idf, on="id1", how="semi")
        for pcol in [c for c in ("p1", "p2", "p2ce") if c in full.columns]:
            base = f_s.select("id1", "id2", pl.col(pcol).alias("p"))
            own = one_owner(base)
            for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
                res.append((scope, pcol, "thr", thr, macro_f05(threshold(base, thr=thr), t_s, ids)))
                res.append((scope, pcol, "own+thr", thr, macro_f05(threshold(own, thr=thr), t_s, ids)))
            for a in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
                res.append((scope, pcol, "own+expF", a, macro_f05(expected_f(own, alpha=a), t_s, ids)))
        if scope == "holdout":
            best = max((r for r in res if r[0] == "holdout"), key=lambda r: r[4])
            # given the best rule, sweep a singleton guard: abstain when the entity's top prob is weak
            bp = best[1]
            bsel = apply_rule(f_s, best[1], best[2], best[3])
            scored = f_s.select("id1", "id2", pl.col(bp).alias("p"))
            gate_best, gf = 0.0, best[4]
            for gate in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                g = macro_f05(singleton_guard(bsel, scored, gate=gate), t_s, ids)
                res.append((scope, bp, f"{best[2]}+g{gate}", best[3], g))
                if g > gf + 1e-9:
                    gf, gate_best = g, gate
            best = (best[0], best[1], best[2], best[3], gf)
            print("BEST holdout", best, "gate", gate_best, flush=True)
            report_best((best[0], best[1], best[2], best[3], gf), f_s, t_s, s1)
            progress.metric("Validation macro F0.5 (holdout)", round(gf, 6))
            json.dump({"pred": best[1], "rule": best[2], "param": best[3], "gate": gate_best,
                       "holdout_macroF05": gf}, open(os.path.join(MODELS, "decision.json"), "w"), indent=1)
    prune_report(full, best, truth, s1)
    df = pl.DataFrame(res, schema=["scope", "pred", "rule", "param", "macroF05"], orient="row")
    pl.Config.set_tbl_rows(100)
    print(df.sort("macroF05", descending=True).head(40))
    df.write_csv(os.path.join(OUTPUTS, "decision_grid.csv"))


if __name__ == "__main__":
    import sys
    tune() if len(sys.argv) > 1 and sys.argv[1] == "tune" else main()
