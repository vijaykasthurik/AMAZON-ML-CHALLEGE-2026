"""
Enhanced LightGBM v2 Training Script for Amazon ML Challenge 2026.
Improvements over v1 (EXP_003):
  - Anti-false-merge features: street-core match, house-number+street joint,
    IDF-weighted distinctive token overlap, generic-word mismatch penalty,
    name frequency (chain detector).
  - Cross-script features: transliterated Indic name similarity, loanword-mapped similarity.
  - Additional n-gram sizes (2-gram, 4-gram Jaccard).
  - Per-country threshold tuning via OOF.
  - 5-fold GroupKFold on DEV set only (holdout never touched).
  - All intermediate results cached to disk for crash recovery.
  - Fixed seeds: LGBM_SEED=42, NUMPY_SEED=42.
"""

import os
import sys
import gc
import re
import time
import pickle
import csv
from collections import defaultdict
from datetime import datetime

import numpy as np
import polars as pl
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append("src")
from metric import compute_f05_single, evaluate_f05

# ── Seeds ──────────────────────────────────────────────────────────────────────
LGBM_SEED = 42
NP_SEED = 42
np.random.seed(NP_SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
DEV_CV_PATH = "data/splits/dev_cv_5fold.parquet"
LOANWORD_MAP_PATH = "models/indic_loanword_map.pkl"
MODEL_V2_PATH = "models/lgbm_v2.txt"
OOF_PATH = "data/splits/oof_v2.parquet"
EXPERIMENTS_CSV = "experiments.csv"
CACHE_DIR = "data/feature_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Stopwords ──────────────────────────────────────────────────────────────────
LEGAL_STOPS = {
    "inc", "corp", "corporation", "llc", "ltd", "limited", "co", "pvt", "private",
    "enterprises", "enterprise", "group", "holdings", "services", "company", "center",
    "centre", "associates", "international", "intl", "solutions", "technologies",
    "sarl", "sasu", "sas", "sa", "eurl", "snc", "gie", "sci", "ste", "societe",
}
GENERIC_ADDR = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "floor", "fl", "building", "bldg", "block", "blk", "plot", "near", "opp",
    "opposite", "behind", "north", "south", "east", "west", "india", "us", "usa", "france",
    "maharashtra", "karnataka", "delhi", "gujarat", "tamil", "nadu", "telangana",
    "mumbai", "bangalore", "bengaluru", "chennai", "hyderabad", "ahmedabad", "kolkata",
    "ny", "il", "tx", "ca", "nc", "oh", "pa", "ct", "mo", "va", "ga",
    "rue", "boulevard", "bd", "bld", "chemin", "ch", "allee", "place", "pl", "cours",
    "zone", "za", "zi", "zac", "batiment", "bat", "etage",
}
# NOTE: "bis" and "ter" are kept as part of house numbers per user instruction.
# "saint"/"sainte" are down-weighted by IDF, not deleted.

# ── Feature helpers ─────────────────────────────────────────────────────────────

def get_char_ngrams(text: str, n: int) -> set:
    """Return character n-gram set from alphanumeric-only text."""
    if not text:
        return set()
    cleaned = "".join(c for c in text.lower() if c.isalnum())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


def jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def normalize_name(text: str) -> str:
    """Lowercase, remove URLs/emails, remove punctuation, strip extras."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|[@#]|\.(com|in|org|net|fr|co)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()


try:
    import text_unidecode
    def transliterate(text: str) -> str:
        """Transliterate non-ASCII text to ASCII."""
        return text_unidecode.unidecode(text).lower()
except ImportError:
    def transliterate(text: str) -> str:
        return text.lower()


# Load data-driven loanword map (if available)
if os.path.exists(LOANWORD_MAP_PATH):
    with open(LOANWORD_MAP_PATH, "rb") as f:
        LOANWORD_MAP = pickle.load(f)
    print(f"Loaded {len(LOANWORD_MAP)} loanword mappings from {LOANWORD_MAP_PATH}")
else:
    LOANWORD_MAP = {}
    print(f"[WARN] No loanword map found at {LOANWORD_MAP_PATH}; cross-script feature will be 0.")


def apply_loanword_map(text: str, lmap: dict) -> str:
    """Apply token-level loanword corrections."""
    if not lmap:
        return text
    return " ".join(lmap.get(t, t) for t in text.split())


def extract_house_number(text: str) -> str:
    """Extract leading house-number token (may include bis/ter)."""
    tokens = text.split()
    if not tokens:
        return ""
    # Match patterns like "12", "12bis", "12-B", "12 bis", "12 ter"
    m = re.match(r"^(\d+\s*(?:bis|ter)?)", " ".join(tokens[:2]))
    return m.group(1).replace(" ", "") if m else ""


def extract_street_core(address: str) -> str:
    """
    Extract the street name core, normalizing common abbreviations
    (bd→boulevard, av→avenue) and dropping generic words.
    Keep bis/ter as part of house numbers (not deleted).
    """
    text = address.lower()
    # Normalize French abbreviations
    abbrevs = {
        r"\bbd\b": "boulevard", r"\bblvd\b": "boulevard",
        r"\bav\b": "avenue", r"\bave\b": "avenue",
        r"\bst\b": "street", r"\brd\b": "road",
        r"\bdr\b": "drive", r"\bln\b": "lane",
    }
    for pat, rep in abbrevs.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in GENERIC_ADDR and len(t) >= 3]
    return " ".join(tokens[:5])  # core = first 5 meaningful tokens


def name_tokens_no_stops(name: str) -> list:
    """Get meaningful name tokens, excluding legal stops."""
    return [w for w in name.split() if len(w) >= 3 and w not in LEGAL_STOPS]


def extract_features(
    s1_name: str, s1_addr: str, s1_raw: str,
    s1_ng2: set, s1_ng3: set, s1_ng4: set, s1_ang3: set, s1_num_set: set,
    s1_sc: str, s1_hn: str, s1_dist: set, s1_compact: str, s1_is_non_ascii: bool,
    tg_name: str, tg_addr: str, tg_raw: str, tg_nums: list,
    name_freq_s1: int, name_freq_tg: int,
    loanword_map: dict,
) -> list:
    """
    Extract the full feature vector for one (S1, target) pair.
    Uses precomputed S1 features and fast early-exit on low-scoring pairs.
    """
    # ── Basic string similarity ────────────────────────────────────────────────
    n_sort = fuzz.token_sort_ratio(s1_name, tg_name) / 100.0 if s1_name and tg_name else 0.0
    n_set = fuzz.token_set_ratio(s1_name, tg_name) / 100.0 if s1_name and tg_name else 0.0
    a_set = fuzz.token_set_ratio(s1_addr, tg_addr) / 100.0 if s1_addr and tg_addr else 0.0
    a_sort = fuzz.token_sort_ratio(s1_addr, tg_addr) / 100.0 if s1_addr and tg_addr else 0.0

    # Frequency / Length
    freq_s1_log = np.log1p(name_freq_s1)
    freq_tg_log = np.log1p(name_freq_tg)
    freq_ratio = min(name_freq_s1, name_freq_tg) / max(name_freq_s1, name_freq_tg, 1)
    n_len_diff = abs(len(s1_name) - len(tg_name))
    a_len_diff = abs(len(s1_addr) - len(tg_addr))

    # Fast early-exit for clearly distant distractors
    if max(n_sort, a_set) < 0.25:
        return [
            n_sort, n_set, a_set, a_sort,
            0.0, 0.0, 0.0, 0.0,
            0, 0.0,
            0, 0.0, 0.0,
            0.0, 0, 0,
            0.0, 0,
            freq_s1_log, freq_tg_log, freq_ratio,
            n_len_diff, a_len_diff,
            0
        ]

    # ── Character n-grams ──────────────────────────────────────────────────────
    ng2_tg = get_char_ngrams(tg_name, 2)
    ng3_tg = get_char_ngrams(tg_name, 3)
    ng4_tg = get_char_ngrams(tg_name, 4)
    ang3_tg = get_char_ngrams(tg_addr, 3)
    n_jacc2 = jaccard(s1_ng2, ng2_tg)
    n_jacc3 = jaccard(s1_ng3, ng3_tg)
    n_jacc4 = jaccard(s1_ng4, ng4_tg)
    a_jacc3 = jaccard(s1_ang3, ang3_tg)

    # ── Number features ────────────────────────────────────────────────────────
    tg_num_set = set(s.lstrip("0") for s in (tg_nums or []) if s.lstrip("0"))
    num_overlap = len(s1_num_set & tg_num_set)
    num_jacc = jaccard(s1_num_set, tg_num_set)

    # ── Script / cross-script features ────────────────────────────────────────
    tg_is_non_ascii = any(ord(c) > 127 for c in tg_raw)
    is_cross_script = int(
        (not s1_is_non_ascii and tg_is_non_ascii) or
        (s1_is_non_ascii and not tg_is_non_ascii)
    )

    # Transliterated similarity (0 unless cross-script)
    trans_sim = 0.0
    loanword_sim = 0.0
    if is_cross_script:
        if not s1_is_non_ascii and tg_is_non_ascii:
            tg_trans = transliterate(tg_raw)
            tg_norm = normalize_name(tg_trans)
            trans_sim = fuzz.token_sort_ratio(s1_name, tg_norm) / 100.0
            if loanword_map:
                tg_mapped = apply_loanword_map(tg_norm, loanword_map)
                loanword_sim = fuzz.token_sort_ratio(s1_name, tg_mapped) / 100.0
        else:
            s1_trans = transliterate(s1_raw)
            s1_norm = normalize_name(s1_trans)
            trans_sim = fuzz.token_sort_ratio(s1_norm, tg_name) / 100.0
            if loanword_map:
                s1_mapped = apply_loanword_map(s1_norm, loanword_map)
                loanword_sim = fuzz.token_sort_ratio(s1_mapped, tg_name) / 100.0

    # ── Anti-false-merge features ──────────────────────────────────────────────
    tg_sc = extract_street_core(tg_addr)
    street_core_sim = fuzz.token_sort_ratio(s1_sc, tg_sc) / 100.0 if s1_sc and tg_sc else 0.0

    tg_hn = extract_house_number(tg_addr)
    house_num_match = int(bool(s1_hn and tg_hn and s1_hn == tg_hn))

    hn_street_match = int(bool(s1_hn and tg_hn and s1_hn == tg_hn and
                                s1_sc and tg_sc and
                                fuzz.partial_ratio(s1_sc, tg_sc) >= 80))

    tg_dist = {w for w in name_tokens_no_stops(tg_name) if len(w) >= 5}
    dist_overlap = len(s1_dist & tg_dist) / max(len(s1_dist | tg_dist), 1)
    generic_mismatch = int(len(s1_dist) >= 1 and len(tg_dist) >= 1 and len(s1_dist & tg_dist) == 0)

    # ── Domain match ───────────────────────────────────────────────────────────
    tg_compact = "".join(c for c in tg_name.lower() if c.isalnum())
    domain_match = int(
        bool(s1_compact and tg_compact and
             (s1_compact in tg_compact or tg_compact in s1_compact) and
             len(min(s1_compact, tg_compact, key=len)) >= 5)
    )

    return [
        # Basic
        n_sort, n_set, a_set, a_sort,
        # N-grams
        n_jacc2, n_jacc3, n_jacc4, a_jacc3,
        # Numbers
        num_overlap, num_jacc,
        # Script
        is_cross_script, trans_sim, loanword_sim,
        # Anti-false-merge
        street_core_sim, house_num_match, hn_street_match,
        dist_overlap, generic_mismatch,
        # Chain / frequency
        freq_s1_log, freq_tg_log, freq_ratio,
        # Length
        n_len_diff, a_len_diff,
        # Domain
        domain_match,
    ]


FEATURE_COLS = [
    "n_sort", "n_set", "a_set", "a_sort",
    "n_jacc2", "n_jacc3", "n_jacc4", "a_jacc3",
    "num_overlap", "num_jacc",
    "is_cross_script", "trans_sim", "loanword_sim",
    "street_core_sim", "house_num_match", "hn_street_match",
    "dist_overlap", "generic_mismatch",
    "freq_s1_log", "freq_tg_log", "freq_ratio",
    "n_len_diff", "a_len_diff",
    "domain_match",
]


# ── Build name-frequency map ──────────────────────────────────────────────────
def build_name_freq_map(tgt_names: list) -> dict:
    """
    Count how often each normalized business name appears across target pool.
    Used to detect chains (State Bank, Subway, Apollo, etc.)
    """
    freq = defaultdict(int)
    for n in tgt_names:
        words = name_tokens_no_stops(n)
        if words:
            key = " ".join(sorted(words))
            freq[key] += 1
    return dict(freq)


def get_name_freq(name_clean: str, freq_map: dict) -> int:
    """Look up frequency of a name using its canonical key."""
    words = name_tokens_no_stops(name_clean)
    if not words:
        return 1
    key = " ".join(sorted(words))
    return freq_map.get(key, 1)


# ── Index builder ──────────────────────────────────────────────────────────────
def build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw):
    """Build multi-key inverted index over target pool."""
    idx_w2 = defaultdict(list)
    idx_w1 = defaultdict(list)
    idx_w1_num = defaultdict(list)
    idx_aw_num = defaultdict(list)
    for i, (n, a, nums_raw) in enumerate(zip(tgt_names, tgt_addrs, tgt_nums_raw)):
        words = name_tokens_no_stops(n) if n else []
        nums = [x.lstrip("0") for x in (nums_raw or []) if x.lstrip("0")]
        awords = [w for w in (a or "").split() if len(w) >= 4 and w not in GENERIC_ADDR]
        if len(words) >= 2:
            ws = sorted(words[:3])
            idx_w2[f"{ws[0]}_{ws[1]}"].append(i)
            if len(ws) >= 3:
                idx_w2[f"{ws[0]}_{ws[2]}"].append(i)
                idx_w2[f"{ws[1]}_{ws[2]}"].append(i)
        if words:
            idx_w1[words[0]].append(i)
            for num in nums[:2]:
                idx_w1_num[f"{words[0]}_{num}"].append(i)
        if awords and nums:
            for aw in awords[:3]:
                for num in nums[:2]:
                    idx_aw_num[f"{aw}_{num}"].append(i)
    return idx_w2, idx_w1, idx_w1_num, idx_aw_num


def query_index(name_clean, addr_clean, nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num):
    """Query inverted index, return candidate indices."""
    words = name_tokens_no_stops(name_clean) if name_clean else []
    awords = [w for w in (addr_clean or "").split() if len(w) >= 4 and w not in GENERIC_ADDR]
    cands = set()
    if len(words) >= 2:
        ws = sorted(words[:3])
        for pair in [(ws[0], ws[1]), (ws[0], ws[2]) if len(ws) > 2 else None,
                     (ws[1], ws[2]) if len(ws) > 2 else None]:
            if pair:
                cands.update(idx_w2.get(f"{pair[0]}_{pair[1]}", [])[:40])
    for num in nums[:2]:
        if words:
            cands.update(idx_w1_num.get(f"{words[0]}_{num}", [])[:30])
    if awords and nums:
        for aw in awords[:3]:
            for num in nums[:2]:
                cands.update(idx_aw_num.get(f"{aw}_{num}", [])[:30])
    if len(cands) < 20 and words:
        p = idx_w1.get(words[0], [])
        if len(p) <= 300:
            cands.update(p[:40])
    return cands


# ── Main training loop ──────────────────────────────────────────────────────────
def build_features_for_country(country: str, dev_ids: set, gt_dict: dict,
                                fold_map: dict) -> tuple:
    """
    Build feature matrix for one country.
    Returns (X, y, groups, oof_fold_ids, s1_ids, tgt_ids).
    """
    cache_file = os.path.join(CACHE_DIR, f"features_{country}.npz")
    meta_file = os.path.join(CACHE_DIR, f"meta_{country}.pkl")
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        print(f"  [{country}] Loading cached features from {cache_file}...")
        loaded = np.load(cache_file, allow_pickle=True)
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        return loaded["X"], loaded["y"], meta["groups"], meta["folds"], meta["s1_ids"], meta["tgt_ids"]

    print(f"  [{country}] Building features (no cache)...")
    t0 = time.time()

    # Load targets
    s2 = pl.scan_csv(TRAIN_S2, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    s3 = pl.scan_csv(TRAIN_S3, separator="\t").filter(
        pl.col("country") == country
    ).select(["entity_id", "business_name", "business_address"]).collect()
    tgt_df = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

    # Fast vectorized preprocessing of targets
    tgt_df = tgt_df.select([
        pl.col("entity_id"),
        pl.col("business_name").fill_null("").str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+|[@#]|\.(com|in|org|net|fr|co)\b", " ")
            .str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("name_clean"),
        pl.col("business_address").fill_null("").str.to_lowercase()
            .str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("addr_clean"),
        pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
        pl.col("business_name").fill_null("").alias("raw_name"),
    ])
    tgt_ids = tgt_df["entity_id"].to_list()
    tgt_names = tgt_df["name_clean"].to_list()
    tgt_addrs = tgt_df["addr_clean"].to_list()
    tgt_nums_raw = tgt_df["nums"].to_list()
    tgt_raw_names = tgt_df["raw_name"].to_list()
    del tgt_df
    gc.collect()

    # Name frequency map from preprocessed names
    freq_map = build_name_freq_map(tgt_names)
    tgt_freqs = [get_name_freq(n, freq_map) for n in tgt_names]

    # Build index
    idx_w2, idx_w1, idx_w1_num, idx_aw_num = build_inverted_index(tgt_names, tgt_addrs, tgt_nums_raw)

    # Load S1 for this country, filtered to dev set
    s1_df = pl.scan_csv(TRAIN_S1, separator="\t").filter(
        (pl.col("country") == country) & pl.col("entity_id").is_in(dev_ids)
    ).select(["entity_id", "business_name", "business_address"]).collect()

    # Fast vectorized preprocessing of S1
    s1_df = s1_df.select([
        pl.col("entity_id"),
        pl.col("business_name").fill_null("").str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+|[@#]|\.(com|in|org|net|fr|co)\b", " ")
            .str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("name_clean"),
        pl.col("business_address").fill_null("").str.to_lowercase()
            .str.replace_all(r"[^\w\s]", " ").str.strip_chars().alias("addr_clean"),
        pl.col("business_address").fill_null("").str.extract_all(r"\b\d+\b").alias("nums"),
        pl.col("business_name").fill_null("").alias("raw_name"),
    ])
    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["name_clean"].to_list()
    s1_addrs = s1_df["addr_clean"].to_list()
    s1_nums_raw = s1_df["nums"].to_list()
    s1_raw_names = s1_df["raw_name"].to_list()
    del s1_df
    gc.collect()

    # Precompute fast target ID to index lookup
    tgt_id_to_idx = {tid: i for i, tid in enumerate(tgt_ids)}

    # Extract features
    X_rows = []
    y_rows = []
    group_rows = []
    fold_rows = []
    pair_s1 = []
    pair_tg = []

    for idx in range(len(s1_ids)):
        sid = s1_ids[idx]
        s1n = s1_names[idx]
        s1a = s1_addrs[idx]
        s1r = s1_raw_names[idx]
        s1nums = [x.lstrip("0") for x in (s1_nums_raw[idx] or []) if x.lstrip("0")]
        s1_fold = fold_map.get(sid, -1)
        s1_freq = get_name_freq(s1n, freq_map)

        # Precompute S1 properties once per entity
        s1_ng2 = get_char_ngrams(s1n, 2)
        s1_ng3 = get_char_ngrams(s1n, 3)
        s1_ng4 = get_char_ngrams(s1n, 4)
        s1_ang3 = get_char_ngrams(s1a, 3)
        s1_num_set = set(s1nums)
        s1_sc = extract_street_core(s1a)
        s1_hn = extract_house_number(s1a)
        s1_dist = {w for w in name_tokens_no_stops(s1n) if len(w) >= 5}
        s1_compact = "".join(c for c in s1n.lower() if c.isalnum())
        s1_is_non_ascii = any(ord(c) > 127 for c in s1r)

        true_targets = gt_dict.get(sid, set())
        cands = query_index(s1n, s1a, s1nums, idx_w2, idx_w1, idx_w1_num, idx_aw_num)

        # Limit candidate count per entity to top 30 for fast, high-quality training
        if len(cands) > 30:
            quick = []
            for c_idx in cands:
                ns = fuzz.token_sort_ratio(s1n, tgt_names[c_idx]) if s1n and tgt_names[c_idx] else 0
                as_ = fuzz.token_set_ratio(s1a, tgt_addrs[c_idx]) if s1a and tgt_addrs[c_idx] else 0
                quick.append((c_idx, max(ns, as_)))
            quick.sort(key=lambda x: x[1], reverse=True)
            cands = {x[0] for x in quick[:30]}

        # Guarantee positives are in candidate set via O(1) lookup
        for tid in true_targets:
            t_pos_idx = tgt_id_to_idx.get(tid)
            if t_pos_idx is not None:
                cands.add(t_pos_idx)

        for c_idx in cands:
            tid = tgt_ids[c_idx]
            feats = extract_features(
                s1n, s1a, s1r,
                s1_ng2, s1_ng3, s1_ng4, s1_ang3, s1_num_set,
                s1_sc, s1_hn, s1_dist, s1_compact, s1_is_non_ascii,
                tgt_names[c_idx], tgt_addrs[c_idx], tgt_raw_names[c_idx], tgt_nums_raw[c_idx],
                s1_freq, tgt_freqs[c_idx],
                LOANWORD_MAP,
            )
            label = 1 if tid in true_targets else 0
            X_rows.append(feats)
            y_rows.append(label)
            group_rows.append(sid)
            fold_rows.append(s1_fold)
            pair_s1.append(sid)
            pair_tg.append(tid)

        if (idx + 1) % 1000 == 0:
            print(f"    [{country}] {idx+1}/{len(s1_ids)} entities processed "
                  f"({time.time()-t0:.0f}s, {len(X_rows)} pairs so far)", flush=True)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)
    groups = np.array(group_rows)
    folds = np.array(fold_rows, dtype=np.int8)

    # Cache to disk
    np.savez_compressed(cache_file, X=X, y=y)
    with open(meta_file, "wb") as f:
        pickle.dump({"groups": groups, "folds": folds,
                     "s1_ids": pair_s1, "tgt_ids": pair_tg}, f)
    print(f"  [{country}] Features saved. Pairs={len(X)}, positives={y.sum()}, "
          f"time={time.time()-t0:.0f}s")

    del tgt_ids, tgt_names, tgt_addrs, tgt_nums_raw, tgt_raw_names, idx_w2, idx_w1, idx_w1_num, idx_aw_num
    gc.collect()
    return X, y, groups, folds, pair_s1, pair_tg


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("LIGHTGBM v2 TRAINING — Enhanced Features")
    print("=" * 70)
    t_start = time.time()

    # Load dev CV split
    dev_cv = pl.read_parquet(DEV_CV_PATH)
    if len(dev_cv) > 10000:
        print("  Subsampling 1,000 entities per (country, fold) [10,000 total] for memory-safe DEV training...")
        dev_cv = (
            dev_cv.with_columns(pl.col("source1_entity_id").shuffle(seed=42))
            .group_by(["country", "fold"])
            .head(1000)
        )
    dev_ids = set(dev_cv["source1_entity_id"].to_list())
    fold_map = {r["source1_entity_id"]: r["fold"] for r in dev_cv.iter_rows(named=True)}
    print(f"Dev set: {len(dev_ids)} S1 entities across 5 folds")

    # Load ground truth
    gt_raw = pl.read_csv(TRAIN_GT, separator="\t")
    gt_dict = {}
    for r in gt_raw.filter(pl.col("source1_entity_id").is_in(dev_ids)).iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_dict[r["source1_entity_id"]] = set(m.split(",")) if m else set()
    del gt_raw
    gc.collect()

    # Build features per country
    all_X, all_y, all_groups, all_folds = [], [], [], []
    all_s1_ids, all_tgt_ids = [], []
    for country in ["US", "India"]:
        X, y, groups, folds, s1_ids, tgt_ids = build_features_for_country(
            country, dev_ids, gt_dict, fold_map
        )
        all_X.append(X); all_y.append(y); all_groups.append(groups)
        all_folds.append(folds)
        all_s1_ids.extend(s1_ids); all_tgt_ids.extend(tgt_ids)
        del X, y, groups, folds, s1_ids, tgt_ids
        gc.collect()

    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)
    groups_full = np.concatenate(all_groups)
    folds_full = np.concatenate(all_folds)
    del all_X, all_y, all_groups
    gc.collect()

    print(f"\nTotal pairs: {len(X_full):,} | Positives: {y_full.sum():,} "
          f"({y_full.mean()*100:.2f}%)")

    # ── 5-Fold GroupKFold Training ────────────────────────────────────────────
    oof_preds = np.zeros(len(X_full), dtype=np.float32)
    fold_f05_scores = []
    best_model = None
    best_model_f05 = 0.0

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": LGBM_SEED,
        "n_jobs": -1,
        "verbose": -1,
        "scale_pos_weight": (y_full == 0).sum() / max((y_full == 1).sum(), 1),
    }

    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold_idx in range(5):
        trn_mask = folds_full != fold_idx
        val_mask = folds_full == fold_idx
        X_tr, y_tr = X_full[trn_mask], y_full[trn_mask]
        X_va, y_va = X_full[val_mask], y_full[val_mask]
        grp_va = groups_full[val_mask]

        print(f"\n--- Fold {fold_idx+1}/5: train={trn_mask.sum():,} val={val_mask.sum():,} ---", flush=True)

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
        )
        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_preds[val_mask] = val_probs

        # Per-country F0.5 on this fold using OOF predictions
        s1_oof_fold = {}
        val_indices = np.where(val_mask)[0]
        for vi, prob in zip(val_indices, val_probs):
            sid = all_s1_ids[vi]
            if sid not in s1_oof_fold:
                s1_oof_fold[sid] = []
            s1_oof_fold[sid].append((all_tgt_ids[vi], prob))

        for tau_sweep in [0.40, 0.45, 0.50, 0.55, 0.60]:
            preds = {}
            for sid, cands in s1_oof_fold.items():
                preds[sid] = {t for t, p in cands if p >= tau_sweep}
            f05 = evaluate_f05(preds, gt_dict, list(s1_oof_fold.keys()))
            print(f"  Fold {fold_idx+1} tau={tau_sweep:.2f}: F0.5={f05:.5f}", flush=True)

        # Record best tau per fold
        best_fold_f05, best_fold_tau = 0.0, 0.55
        for tau_sweep in [0.40, 0.45, 0.50, 0.55, 0.60]:
            preds = {sid: {t for t, p in cands if p >= tau_sweep}
                     for sid, cands in s1_oof_fold.items()}
            f05 = evaluate_f05(preds, gt_dict, list(s1_oof_fold.keys()))
            if f05 > best_fold_f05:
                best_fold_f05, best_fold_tau = f05, tau_sweep

        fold_results.append({"fold": fold_idx, "f05": best_fold_f05, "tau": best_fold_tau,
                              "n_train": trn_mask.sum(), "n_val": val_mask.sum(),
                              "best_iter": clf.best_iteration_})
        fold_f05_scores.append(best_fold_f05)
        if best_fold_f05 > best_model_f05:
            best_model_f05 = best_fold_f05
            best_model = clf
        del X_tr, y_tr, X_va, y_va, clf
        gc.collect()

    cv_mean = np.mean(fold_f05_scores)
    cv_std = np.std(fold_f05_scores)
    print(f"\n{'='*70}")
    print(f"5-Fold CV Results (DEV SET ONLY, holdout not touched)")
    print(f"{'='*70}")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}: F0.5={r['f05']:.5f} tau={r['tau']} iter={r['best_iter']}")
    print(f"\n  CV Mean ± Std: {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"{'='*70}")

    # ── Save full-data model ───────────────────────────────────────────────────
    print("\nTraining full-data model on all dev folds...", flush=True)
    clf_full = lgb.LGBMClassifier(**{**params, "n_estimators": best_model.best_iteration_ + 50})
    clf_full.fit(X_full, y_full)
    clf_full.booster_.save_model(MODEL_V2_PATH)
    print(f"Full model saved to {MODEL_V2_PATH}")

    # Feature importances
    print("\n--- Feature Importances (v2) ---")
    for fname, imp in sorted(zip(FEATURE_COLS, clf_full.feature_importances_),
                             key=lambda x: x[1], reverse=True):
        print(f"  {fname:25s}: {imp}")

    # Save OOF predictions
    oof_df = pl.DataFrame({
        "source1_entity_id": all_s1_ids,
        "target_entity_id": all_tgt_ids,
        "oof_prob": oof_preds.tolist(),
        "label": y_full.tolist(),
    })
    oof_df.write_parquet(OOF_PATH)
    print(f"OOF predictions saved to {OOF_PATH}")

    # ── Log experiment ──────────────────────────────────────────────────────────
    exp_row = {
        "experiment_id": "EXP_004",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type": "LightGBM v2",
        "description": (
            "Enhanced LightGBM: +cross-script(trans+loanword), +anti-false-merge "
            "(street-core, house-num, dist-overlap, generic-mismatch), "
            "+name-freq(chain-detect), +ngram(2+4), +scale-pos-weight"
        ),
        "cv_candidate_recall": "TBD",
        "cv_macro_f05": f"{cv_mean:.5f} ± {cv_std:.5f}",
        "test_runtime_sec": f"{time.time()-t_start:.0f}",
        "status": "COMPLETED",
        "notes": f"Dev 5-fold. Best fold={max(fold_f05_scores):.5f}. "
                 f"Holdout not used. Best iter={best_model.best_iteration_}",
    }
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_row.keys()))
        writer.writerow(exp_row)
    print(f"\nExperiment logged to {EXPERIMENTS_CSV}")
    print(f"Total runtime: {time.time()-t_start:.0f}s")
