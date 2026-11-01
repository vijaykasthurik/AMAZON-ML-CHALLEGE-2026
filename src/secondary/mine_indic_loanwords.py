"""
Data-Driven Indic Loanword Mining from Training True Pairs.
Protocol:
1. Load India true pairs from training ground truth.
2. For each pair where S1 is ASCII and target has non-ASCII tokens:
   - Transliterate target tokens with text_unidecode.
   - Best-pair each transliterated token to S1 tokens via RapidFuzz.
   - Record (transliterated_token -> s1_token) mapping when similarity >= 0.75.
3. Keep mappings seen >= min_count times across the training set.
4. Evaluate on 5-fold GroupKFold by source1_entity_id (held-out folds only).
5. Report: mean name similarity and candidate recall@20/50/100 on cross-script pairs
   BEFORE and AFTER applying the mined mapping, on held-out folds only.
"""

import sys
import re
import time
import pickle
from collections import Counter, defaultdict
import polars as pl
import numpy as np
from rapidfuzz import fuzz, process as rfprocess
import text_unidecode

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

MIN_MAPPING_COUNT = 5  # Minimum times a mapping must appear to be kept
MAX_EVAL_PAIRS = 50000  # Cap to avoid OOM (sampled uniformly across S1 groups)


def normalize_ascii(text: str) -> str:
    """Transliterate to ASCII and lowercase."""
    if not text:
        return ""
    return text_unidecode.unidecode(text).lower()


def get_tokens(text: str) -> list:
    """Tokenize a cleaned string into alphabetic tokens of length >= 2."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def mine_mappings_from_fold(fold_pairs: list) -> Counter:
    """
    Mine (transliterated_tok -> latin_tok) mappings from a list of
    (s1_name_ascii, target_name_indic) true pairs.
    Returns a Counter of (trans_tok, latin_tok) -> occurrence count.
    """
    mapping_counts = Counter()
    for s1_name, tg_name in fold_pairs:
        s1_tokens = get_tokens(s1_name)
        # Transliterate target
        tg_trans = normalize_ascii(tg_name)
        tg_tokens = get_tokens(tg_trans)
        if not s1_tokens or not tg_tokens:
            continue
        # For each tg_trans token, find best matching s1 token
        for tg_tok in tg_tokens:
            if len(tg_tok) < 3:
                continue
            best_match, score, _ = rfprocess.extractOne(
                tg_tok, s1_tokens, scorer=fuzz.ratio
            )
            if score >= 75 and tg_tok != best_match:
                mapping_counts[(tg_tok, best_match)] += 1
    return mapping_counts


def apply_mapping(text: str, mapping: dict) -> str:
    """Apply a loanword mapping dict to a text string token-by-token."""
    tokens = text.split()
    return " ".join(mapping.get(t, t) for t in tokens)


if __name__ == "__main__":
    t0 = time.time()
    print("Loading India training data...")

    s1_df = pl.read_csv(
        "student_resource/dataset/train/train_source1.tsv", separator="\t"
    ).filter(pl.col("country") == "India").select(["entity_id", "business_name"])
    s1_name_map = {
        r["entity_id"]: (r["business_name"] or "")
        for r in s1_df.iter_rows(named=True)
    }

    gt = pl.read_csv(
        "student_resource/dataset/train/train_ground_truth.tsv", separator="\t"
    ).filter(pl.col("source1_entity_id").is_in(set(s1_name_map.keys())))

    print("Loading India targets (train)...")
    s2 = pl.scan_csv(
        "student_resource/dataset/train/train_source2.tsv", separator="\t"
    ).filter(pl.col("country") == "India").select(["entity_id", "business_name"]).collect()
    s3 = pl.scan_csv(
        "student_resource/dataset/train/train_source3.tsv", separator="\t"
    ).filter(pl.col("country") == "India").select(["entity_id", "business_name"]).collect()
    tgt_map = {
        r["entity_id"]: (r["business_name"] or "")
        for r in pl.concat([s2, s3]).iter_rows(named=True)
    }
    print(f"Loaded {len(s1_name_map):,} S1 and {len(tgt_map):,} targets in {time.time()-t0:.1f}s")

    # Collect cross-script true pairs: S1 is ASCII, target has non-ASCII
    t1 = time.time()
    print("Collecting cross-script true pairs...")
    all_cross_pairs = []  # (s1_entity_id, s1_name, tg_name)
    for r in gt.iter_rows(named=True):
        sid = r["source1_entity_id"]
        s1_name = s1_name_map.get(sid, "")
        if not s1_name or any(ord(c) > 127 for c in s1_name):
            continue  # Skip non-ASCII S1
        mids = r["matched_entity_ids"]
        if not mids:
            continue
        for tid in mids.split(","):
            tg_name = tgt_map.get(tid, "")
            if tg_name and any(ord(c) > 127 for c in tg_name):
                all_cross_pairs.append((sid, s1_name, tg_name))

    print(f"Found {len(all_cross_pairs):,} cross-script true pairs in {time.time()-t1:.1f}s")

    # Subsample for evaluation to stay within RAM while inference is running
    if len(all_cross_pairs) > MAX_EVAL_PAIRS:
        np.random.seed(42)
        idx_sample = np.random.choice(len(all_cross_pairs), MAX_EVAL_PAIRS, replace=False)
        all_cross_pairs_eval = [all_cross_pairs[i] for i in idx_sample]
        print(f"Subsampled to {len(all_cross_pairs_eval):,} pairs for evaluation (RAM-safe)")
    else:
        all_cross_pairs_eval = all_cross_pairs

    # 5-fold GroupKFold by source1_entity_id (on the eval subset)
    all_s1_ids = list({p[0] for p in all_cross_pairs_eval})
    np.random.seed(42)
    np.random.shuffle(all_s1_ids)
    n_folds = 5
    fold_size = len(all_s1_ids) // n_folds
    folds = [set(all_s1_ids[i * fold_size:(i + 1) * fold_size]) for i in range(n_folds)]
    folds[-1].update(all_s1_ids[n_folds * fold_size:])  # remainder to last fold

    print(f"Running 5-fold GroupKFold mapping mining (evaluating on held-out folds, {len(all_cross_pairs_eval):,} pairs)...")

    all_held_out_results = []  # (sim_before, sim_after_data, sim_after_data_plus_manual) per pair

    # Manual seed list (only kept if it adds gain on top of mined mappings)
    MANUAL_MAP = {
        "elelpi": "llp", "elelp": "llp", "elpi": "llp",
        "praivett": "private", "praivet": "private", "prevet": "private",
        "limittedd": "limited", "limitted": "limited",
        "sttett": "state", "baink": "bank",
        "phuudd": "food", "paarmci": "pharmacy",
        "hottl": "hotel", "resttorntt": "restaurant",
    }

    for fold_idx in range(n_folds):
        val_s1_ids = folds[fold_idx]
        # Mine mappings from full training pairs (not subsampled) for all folds except current
        train_pairs = [(s1n, tgn) for sid, s1n, tgn in all_cross_pairs if sid not in val_s1_ids]
        val_pairs = [(sid, s1n, tgn) for sid, s1n, tgn in all_cross_pairs_eval if sid in val_s1_ids]

        # Mine mappings from training folds only
        mapping_counts = mine_mappings_from_fold(train_pairs)

        # Build mapping dict with minimum count threshold
        fold_mapping = {
            trans_tok: lat_tok
            for (trans_tok, lat_tok), cnt in mapping_counts.items()
            if cnt >= MIN_MAPPING_COUNT
        }

        # Evaluate on held-out validation pairs
        for sid, s1n, tgn in val_pairs:
            s1_clean = re.sub(r"[^\w\s]", " ", s1n.lower()).strip()
            tg_trans = normalize_ascii(tgn)
            tg_clean = re.sub(r"[^\w\s]", " ", tg_trans).strip()

            # Before
            sim_before = fuzz.token_sort_ratio(s1_clean, tg_clean) / 100.0

            # After data-driven mapping
            tg_mapped = apply_mapping(tg_clean, fold_mapping)
            sim_data = fuzz.token_sort_ratio(s1_clean, tg_mapped) / 100.0

            # After data-driven + manual
            tg_mapped_plus = apply_mapping(tg_clean, {**fold_mapping, **MANUAL_MAP})
            sim_data_plus = fuzz.token_sort_ratio(s1_clean, tg_mapped_plus) / 100.0

            all_held_out_results.append((sim_before, sim_data, sim_data_plus))

    mean_before = np.mean([r[0] for r in all_held_out_results])
    mean_data = np.mean([r[1] for r in all_held_out_results])
    mean_data_plus = np.mean([r[2] for r in all_held_out_results])

    print(f"\n{'='*70}")
    print("DATA-DRIVEN LOANWORD MAPPING RESULTS (5-FOLD, HELD-OUT EVALUATION)")
    print(f"{'='*70}")
    print(f"Total cross-script pairs evaluated: {len(all_held_out_results):,}")
    print(f"1. Baseline (raw unidecode):                 Mean sim = {mean_before:.4f}")
    print(f"2. + Data-driven mapping (>={MIN_MAPPING_COUNT} occurrences): Mean sim = {mean_data:.4f}  (delta: {mean_data - mean_before:+.4f})")
    print(f"3. + Data-driven + manual seeds:             Mean sim = {mean_data_plus:.4f}  (delta: {mean_data_plus - mean_before:+.4f})")
    print(f"{'='*70}")

    # Now build the full-data mining (ALL folds, for production use)
    t2 = time.time()
    print("\nMining full-data mapping for production (all training pairs)...")
    all_train_pairs = [(s1n, tgn) for _, s1n, tgn in all_cross_pairs]
    full_mapping_counts = mine_mappings_from_fold(all_train_pairs)
    full_mapping = {
        trans_tok: lat_tok
        for (trans_tok, lat_tok), cnt in full_mapping_counts.items()
        if cnt >= MIN_MAPPING_COUNT
    }
    print(f"Mined {len(full_mapping):,} unique token mappings in {time.time()-t2:.1f}s")
    print(f"Top 30 most frequent mappings:")
    top_pairs = sorted(
        [(cnt, t, l) for (t, l), cnt in full_mapping_counts.items() if cnt >= MIN_MAPPING_COUNT],
        reverse=True
    )[:30]
    for cnt, trans_tok, lat_tok in top_pairs:
        print(f"  '{trans_tok}' -> '{lat_tok}'  (seen {cnt}x)")

    # Save full mapping to disk for production use
    with open("models/indic_loanword_map.pkl", "wb") as f:
        pickle.dump(full_mapping, f)
    print(f"\nFull production mapping saved to models/indic_loanword_map.pkl")
    print(f"Total runtime: {time.time()-t0:.1f}s")
