"""
CatBoost Training & LightGBM + CatBoost Ensemble on DEV 5-Fold.
Loads cached 24 features from data/feature_cache_v2/.
Trains 5-fold CatBoostClassifier.
Computes CatBoost OOF predictions and blends with LightGBM v2 OOF predictions.
Evaluates per-country P/R/F0.5 and macro F0.5 across blend weights and tau.
Logs to experiments.csv.
"""
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append("src")
from metric import compute_f05_single

# Paths
LGBM_OOF_PATH = "data/splits/oof_v2.parquet"
CACHE_DIR = "data/feature_cache_v2"
CB_MODEL_PATH = "models/catboost_v2.cbm"
CB_OOF_PATH = "data/splits/oof_catboost_v2.parquet"
BLEND_OOF_PATH = "data/splits/oof_blend_v2.parquet"
EXPERIMENTS_CSV = "experiments.csv"
DEV_CV_PATH = "data/splits/dev_cv_5fold.parquet"
GT_PATH = "student_resource/dataset/train/train_ground_truth.tsv"

def load_cached_data():
    f_us = np.load(os.path.join(CACHE_DIR, "features_US.npz"))
    f_in = np.load(os.path.join(CACHE_DIR, "features_India.npz"))
    
    with open(os.path.join(CACHE_DIR, "meta_US.pkl"), "rb") as f:
        m_us = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "meta_India.pkl"), "rb") as f:
        m_in = pickle.load(f)
        
    X_us, y_us = f_us["X"], f_us["y"]
    X_in, y_in = f_in["X"], f_in["y"]
    
    X = np.vstack([X_us, X_in])
    y = np.concatenate([y_us, y_in])
    folds = np.concatenate([m_us["folds"], m_in["folds"]])
    s1_ids = m_us["s1_ids"] + m_in["s1_ids"]
    tgt_ids = m_us["tgt_ids"] + m_in["tgt_ids"]
    countries = ["US"] * len(X_us) + ["India"] * len(X_in)
    
    print(f"Loaded {len(X):,} pairs ({np.sum(y==1):,} positives, {np.mean(y)*100:.2f}%)")
    return X, y, folds, s1_ids, tgt_ids, countries

def main():
    print("=" * 70)
    print("ENSEMBLE: CatBoost 5-Fold Training & LightGBM Blend")
    print("=" * 70)
    
    t0 = time.time()
    X, y, folds, s1_ids, tgt_ids, countries = load_cached_data()
    
    # Load LightGBM OOF
    lgbm_oof = pd.read_parquet(LGBM_OOF_PATH)
    lgbm_prob = lgbm_oof["oof_prob"].values
    
    # Load Ground Truth
    gt_df = pd.read_csv(GT_PATH, sep="\t")
    gt_map = {}
    for _, r in gt_df.iterrows():
        m = r["matched_entity_ids"]
        gt_map[r["source1_entity_id"]] = set(m.split(",")) if pd.notna(m) and str(m).strip() else set()
        
    cb_oof_prob = np.zeros(len(y), dtype=np.float32)
    cb_models = []
    
    # 5-fold CV
    fold_f05_cb = []
    fold_f05_blend = []
    
    for fold in range(5):
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        pos_weight = float((len(y_tr) - np.sum(y_tr)) / np.sum(y_tr))
        
        cb = CatBoostClassifier(
            iterations=600,
            learning_rate=0.08,
            depth=6,
            scale_pos_weight=min(pos_weight, 5.0),
            loss_function='Logloss',
            eval_metric='Logloss',
            random_seed=42 + fold,
            verbose=200,
            task_type='CPU',
            thread_count=8,
        )
        
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=200)
        cb_models.append(cb)
        
        val_preds = cb.predict_proba(X_va)[:, 1]
        cb_oof_prob[val_mask] = val_preds
        
        # Test fold metric
        val_s1s = list(set([s1_ids[i] for i in np.where(val_mask)[0]]))
        
        # Best tau for blend on fold
        val_lgbm = lgbm_prob[val_mask]
        blend_fold = 0.5 * val_lgbm + 0.5 * val_preds
        
        # Score blend
        sub_df = pd.DataFrame({
            "s1": [s1_ids[i] for i in np.where(val_mask)[0]],
            "tgt": [tgt_ids[i] for i in np.where(val_mask)[0]],
            "prob": blend_fold,
        })
        
        above = sub_df[sub_df["prob"] >= 0.60].sort_values("prob", ascending=False)
        pred_map = {}
        used_tgt = set()
        for _, r in above.iterrows():
            s1, tgt = r["s1"], r["tgt"]
            if s1 not in pred_map:
                pred_map[s1] = set()
            if tgt not in used_tgt:
                pred_map[s1].add(tgt)
                used_tgt.add(tgt)
                
        f05_scores = [compute_f05_single(pred_map.get(s, set()), gt_map.get(s, set())) for s in val_s1s]
        score = np.mean(f05_scores)
        fold_f05_blend.append(score)
        print(f"Fold {fold+1}/5: Blend F0.5 (tau=0.60) = {score:.5f}")
        
    mean_blend = np.mean(fold_f05_blend)
    std_blend = np.std(fold_f05_blend)
    print(f"\nBlend (50% LGBM + 50% CatBoost) CV Mean ± Std: {mean_blend:.5f} ± {std_blend:.5f}")
    
    # Save CatBoost OOF
    cb_oof_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "target_entity_id": tgt_ids,
        "oof_prob": cb_oof_prob,
        "label": y,
    })
    cb_oof_df.to_parquet(CB_OOF_PATH, index=False)
    
    # Grid search blend weights and per-country thresholds
    print("\n--- GRID SEARCH BLEND WEIGHTS & PER-COUNTRY TAUS ---")
    best_overall_f05 = 0.0
    best_w = 0.5
    best_tau_us = 0.85
    best_tau_in = 0.85
    
    oof_df_all = pd.DataFrame({
        "s1": s1_ids,
        "tgt": tgt_ids,
        "country": countries,
        "lgbm": lgbm_prob,
        "cb": cb_oof_prob,
        "label": y,
    })
    
    for w in [0.3, 0.5, 0.7]:
        oof_df_all["blend"] = w * oof_df_all["lgbm"] + (1 - w) * oof_df_all["cb"]
        
        for tau in [0.70, 0.75, 0.80, 0.85, 0.90]:
            f05_by_country = {}
            for c in ["US", "India"]:
                sub = oof_df_all[oof_df_all["country"] == c]
                c_s1s = sub["s1"].unique()
                above = sub[sub["blend"] >= tau].sort_values("blend", ascending=False)
                pred_map = {}
                used = set()
                for _, r in above.iterrows():
                    s1, tgt = r["s1"], r["tgt"]
                    if s1 not in pred_map:
                        pred_map[s1] = set()
                    if tgt not in used:
                        pred_map[s1].add(tgt)
                        used.add(tgt)
                scores = [compute_f05_single(pred_map.get(s, set()), gt_map.get(s, set())) for s in c_s1s]
                f05_by_country[c] = np.mean(scores)
                
            weighted = 0.44994 * f05_by_country["US"] + 0.55006 * f05_by_country["India"]
            print(f"Weight LGBM={w:.1f} | tau={tau:.2f} -> US={f05_by_country['US']:.5f}, India={f05_by_country['India']:.5f}, Weighted={weighted:.5f}")
            if weighted > best_overall_f05:
                best_overall_f05 = weighted
                best_w = w
                
    print(f"\n--> BEST ENSEMBLE: Weight LGBM={best_w:.1f} | Weighted F0.5 = {best_overall_f05:.5f}")
    
    # Save final blend booster/weights
    oof_df_all["final_blend"] = best_w * oof_df_all["lgbm"] + (1 - best_w) * oof_df_all["cb"]
    oof_df_all[["s1", "tgt", "final_blend", "label"]].rename(
        columns={"s1": "source1_entity_id", "tgt": "target_entity_id", "final_blend": "oof_prob"}
    ).to_parquet(BLEND_OOF_PATH, index=False)
    
    # Log to experiments.csv
    runtime = time.time() - t0
    row = f"EXP_005,{time.strftime('%Y-%m-%d %H:%M')},LightGBM+CatBoost Ensemble,Blend LGBM v2 ({best_w:.1f}) + CatBoost ({1-best_w:.1f}) on 24 features,TBD,{best_overall_f05:.5f} (weighted),{runtime:.1f},COMPLETED,5-Fold CV. Best weighted F0.5={best_overall_f05:.5f}\n"
    with open(EXPERIMENTS_CSV, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"Logged EXP_005 to {EXPERIMENTS_CSV}")
    print(f"Total runtime: {runtime:.1f}s")

if __name__ == "__main__":
    main()
