"""
OOF Error Analysis — Top 50 worst predictions per country.
Reads OOF predictions from data/splits/oof_v2.parquet and
ground truth from train_ground_truth.tsv.

Categorizes errors as:
  MISSED_CANDIDATE — true match was never retrieved by blocking
  FALSE_MERGE      — low-quality pair predicted as match
  FALSE_REJECT     — high-score pair rejected by threshold
  CROSS_SCRIPT     — error involves Indic/non-Latin text

Usage:
    python src/analyze_oof_errors.py [--country US|India] [--n 50]
"""

import sys
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric import compute_f05_single

OOF_PATH = "data/splits/oof_v2.parquet"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"


def analyze_errors(country: str, n_worst: int, tau: float):
    """Identify and categorize the n_worst OOF errors for a country."""
    print(f"\n{'='*70}")
    print(f"OOF ERROR ANALYSIS — {country}, tau={tau}, top {n_worst} worst")
    print(f"{'='*70}")

    # Load OOF
    oof = pl.read_parquet(OOF_PATH)

    # Load S1 for country
    s1_df = pl.scan_csv(TRAIN_S1, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    s1_map = {r["entity_id"]: r for r in s1_df.iter_rows(named=True)}
    country_s1_ids = set(s1_map.keys())

    # Load GT
    gt_df = pl.scan_csv(TRAIN_GT, separator="\t").filter(
        pl.col("source1_entity_id").is_in(country_s1_ids)
    ).collect()
    gt_dict = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_dict[r["source1_entity_id"]] = set(m.split(",")) if m else set()

    # Load targets
    s2 = pl.scan_csv(TRAIN_S2, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    s3 = pl.scan_csv(TRAIN_S3, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    tgt_map = {r["entity_id"]: r for r in pl.concat([s2, s3]).iter_rows(named=True)}

    # Filter OOF to this country's S1 entities
    oof_c = oof.filter(pl.col("source1_entity_id").is_in(country_s1_ids))

    # Group by S1 entity
    s1_oof = defaultdict(list)
    for r in oof_c.iter_rows(named=True):
        s1_oof[r["source1_entity_id"]].append(
            (r["target_entity_id"], float(r["oof_prob"]), int(r["label"]))
        )

    # Compute per-entity F0.5 and find worst
    entity_scores = []
    for sid, cands in s1_oof.items():
        true_set = gt_dict.get(sid, set())
        pred_set = {t for t, p, _ in cands if p >= tau}
        f05 = compute_f05_single(pred_set, true_set)
        entity_scores.append((f05, sid, pred_set, true_set, cands))

    # Sort by worst F0.5
    entity_scores.sort(key=lambda x: x[0])
    worst = entity_scores[:n_worst]

    # Categorize and print
    category_counts = defaultdict(int)

    for rank, (f05, sid, pred_set, true_set, cands) in enumerate(worst, 1):
        s1_info = s1_map.get(sid, {})
        s1_name = s1_info.get("business_name", "")
        s1_addr = s1_info.get("business_address", "")

        # Categorize error
        has_non_ascii_s1 = any(ord(c) > 127 for c in (s1_name or ""))
        missed = true_set - {t for t, p, _ in cands}  # true matches never retrieved by blocking
        false_merges = pred_set - true_set
        false_rejects = {t for t, p, lbl in cands if lbl == 1 and p < tau}

        if missed:
            cat = "MISSED_CANDIDATE"
        elif false_merges:
            cat = "FALSE_MERGE"
        elif false_rejects:
            cat = "FALSE_REJECT"
        else:
            cat = "OTHER"

        # Refine with cross-script tag
        is_cross = has_non_ascii_s1
        for t, p, lbl in cands:
            if lbl == 1:
                tg = tgt_map.get(t, {})
                if tg and any(ord(c) > 127 for c in (tg.get("business_name") or "")):
                    is_cross = True
                    break
        if is_cross and cat != "OTHER":
            cat = f"CROSS_SCRIPT/{cat}"

        category_counts[cat] += 1

        if rank <= 20:  # Print top 20 worst
            print(f"\n  [{rank}] F0.5={f05:.3f} | {cat} | S1={sid}")
            print(f"       S1 name   : {s1_name[:80]}")
            print(f"       S1 addr   : {s1_addr[:80]}")
            if missed:
                for tid in list(missed)[:2]:
                    tg = tgt_map.get(tid, {})
                    print(f"       MISSED TGT: {tg.get('business_name','')[:60]} | {tg.get('business_address','')[:40]}")
            if false_merges:
                for tid in list(false_merges)[:2]:
                    tg = tgt_map.get(tid, {})
                    prob = next((p for t, p, _ in cands if t == tid), 0)
                    print(f"       FALSE_MERGE: prob={prob:.3f} | {tg.get('business_name','')[:60]}")
            if false_rejects:
                for tid in list(false_rejects)[:2]:
                    tg = tgt_map.get(tid, {})
                    prob = next((p for t, p, _ in cands if t == tid), 0)
                    print(f"       FALSE_REJECT: prob={prob:.3f} | {tg.get('business_name','')[:60]}")

    print(f"\n  Error category summary (top {n_worst} worst entities):")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:<35s}: {cnt:4d} ({cnt/n_worst*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="India", choices=["US", "India"])
    parser.add_argument("--n", type=int, default=50, help="Number of worst errors to analyze")
    parser.add_argument("--tau", type=float, default=0.50, help="Classification threshold")
    args = parser.parse_args()
    analyze_errors(args.country, args.n, args.tau)
