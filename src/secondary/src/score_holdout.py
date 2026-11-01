"""
LOCKED HOLDOUT SCORER — Final Gate.

Rules:
- This script is called AT MOST 3 times total (v1, best candidate, final).
- Each call is logged to data/splits/holdout_use_log.jsonl with timestamp + version tag.
- The holdout IDs are read from data/splits/holdout_ids.parquet.
- The holdout GT is read fresh from train_ground_truth.tsv (ground truth is always correct).
- The prediction file (matching_results.tsv) must already be produced by the inference pipeline
  running on training data with holdout entities as test set.

Usage:
    python src/score_holdout.py --pred <matching_results.tsv> --version v1
    python src/score_holdout.py --pred <matching_results.tsv> --version v2_best
    python src/score_holdout.py --pred <matching_results.tsv> --version final
"""

import sys
import os
import json
import argparse
from datetime import datetime

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append("src")
from metric import compute_f05_single

HOLDOUT_IDS_PATH = "data/splits/holdout_ids.parquet"
HOLDOUT_LOG = "data/splits/calibrated_holdout_use_log.jsonl"
TRAIN_GT = "data/calibrated_holdout/ground_truth.tsv"
MAX_HOLDOUT_USES = 3

COUNTRY_WEIGHTS = {"US": 0.383, "India": 0.468, "France": 0.150}
TARGET_SCORE = 0.98999


def load_holdout_ids() -> dict:
    """Returns {entity_id: country}."""
    df = pl.read_parquet(HOLDOUT_IDS_PATH)
    return {r["source1_entity_id"]: r["country"] for r in df.iter_rows(named=True)}


def load_gt_for_holdout(holdout_ids: set) -> dict:
    """Returns {entity_id: set_of_matched_ids}."""
    gt_df = pl.scan_csv(TRAIN_GT, separator="\t").collect()
    result = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        result[r["source1_entity_id"]] = set(m.split(",")) if m else set()
    return result


def load_predictions(pred_path: str, holdout_ids: set) -> dict:
    """Read matching_results.tsv and return predictions for holdout entities."""
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            if sid not in holdout_ids:
                continue
            matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            preds[sid] = matches
    # Entities with no prediction row = empty prediction
    for sid in holdout_ids:
        if sid not in preds:
            preds[sid] = set()
    return preds


def score(version: str, pred_path: str, dry_run: bool = False):
    """Score the holdout and log the use."""
    # Check holdout exists
    if not os.path.exists(HOLDOUT_IDS_PATH):
        print("ERROR: Holdout IDs not found. Run src/holdout.py first.")
        sys.exit(1)

    # Count existing uses
    existing_uses = []
    if os.path.exists(HOLDOUT_LOG):
        with open(HOLDOUT_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_uses.append(json.loads(line))

    if not dry_run and len(existing_uses) >= MAX_HOLDOUT_USES:
        print(f"ERROR: Holdout already used {len(existing_uses)} times (max {MAX_HOLDOUT_USES}).")
        print("Existing uses:")
        for use in existing_uses:
            print(f"  {use['timestamp']} | version={use['version']} | weighted_f05={use.get('weighted_f05','?')}")
        sys.exit(1)

    # Load holdout
    holdout_map = load_holdout_ids()
    holdout_ids = set(holdout_map.keys())
    gt_dict = load_gt_for_holdout(holdout_ids)
    preds = load_predictions(pred_path, holdout_ids)

    print(f"\n{'='*70}")
    print(f"LOCKED HOLDOUT EVALUATION — version={version}")
    print(f"Holdout entities: {len(holdout_ids):,}")
    print(f"Prediction file: {pred_path}")
    print(f"Holdout use #{len(existing_uses)+1} of {MAX_HOLDOUT_USES}")
    print(f"{'='*70}")

    # Per-country scoring
    country_results = {}
    for country in ["US", "India", "France"]:
        c_ids = [sid for sid, c in holdout_map.items() if c == country]
        if not c_ids:
            continue
        f05_scores = []
        total_tp = total_pred = total_true = 0
        for sid in c_ids:
            pred_set = preds.get(sid, set())
            true_set = gt_dict.get(sid, set())
            f05 = compute_f05_single(pred_set, true_set)
            f05_scores.append(f05)
            tp = len(pred_set & true_set)
            total_tp += tp
            total_pred += len(pred_set)
            total_true += len(true_set)

        prec = total_tp / total_pred if total_pred > 0 else 1.0
        rec = total_tp / total_true if total_true > 0 else 0.0
        macro_f05 = float(np.mean(f05_scores))
        singleton_rate = sum(1 for sid in c_ids if not gt_dict.get(sid)) / len(c_ids)
        country_results[country] = {
            "n": len(c_ids), "macro_f05": macro_f05,
            "precision": prec, "recall": rec,
            "singleton_rate": singleton_rate,
        }
        note = "[MODEL-LABELED — NOT GOLD]" if country == "France" else ""
        print(f"  {country:8s} n={len(c_ids):6,} | F0.5={macro_f05:.5f} | "
              f"P={prec:.4f} R={rec:.4f} | singleton={singleton_rate:.2%} {note}")

    # Weighted F0.5
    weighted_f05 = 0.0
    total_weight = 0.0
    for country, w in COUNTRY_WEIGHTS.items():
        if country in country_results:
            weighted_f05 += w * country_results[country]["macro_f05"]
            total_weight += w
    if total_weight > 0:
        weighted_f05 /= total_weight

    print(f"\n  Weighted F0.5 (US×{COUNTRY_WEIGHTS['US']:.3f} + India×{COUNTRY_WEIGHTS['India']:.3f} + France×{COUNTRY_WEIGHTS['France']:.3f}):")
    print(f"  {'PASS' if weighted_f05 >= TARGET_SCORE else 'FAIL'} — {weighted_f05:.5f} (target {TARGET_SCORE})")

    # Density stability check
    print(f"\n  NOTE: Single density tested. Run run_density_stability_check.py for 4.5/5.75/8.0 table.")
    print(f"{'='*70}")

    # Log this use
    if not dry_run:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "pred_path": pred_path,
            "weighted_f05": round(weighted_f05, 6),
            "country_results": {
                k: {kk: round(vv, 5) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in country_results.items()
            },
            "passed_gate": weighted_f05 >= TARGET_SCORE,
        }
        with open(HOLDOUT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"  [Logged to {HOLDOUT_LOG}]")

    return weighted_f05, country_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score locked holdout")
    parser.add_argument("--pred", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--version", required=True, help="Version tag (v1, v2_best, final)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score without logging (for debugging only)")
    args = parser.parse_args()
    score(args.version, args.pred, dry_run=args.dry_run)
