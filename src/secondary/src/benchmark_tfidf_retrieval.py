"""
Track B3: Char 2-4-gram TF-IDF Retrieval Benchmark (Laptop CPU).
Evaluates character 2-4-gram TF-IDF vectorization and sparse retrieval
on the Calibrated DEV Split (10k S1, 57.5k targets) for India and US.
Measures:
  - Recall@5, @10, @20, @50
  - Average candidates per S1
  - Reduction ratio
  - Runtime and throughput (records/sec)
"""
import os
import sys
import time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def benchmark_tfidf_country(country="India"):
    print(f"\n{'='*70}")
    print(f"TRACK B3: CHAR 2-4-GRAM TF-IDF BENCHMARK — {country}")
    print(f"{'='*70}")
    
    split_dir = f"data/calibrated_dev/{country}_density_5.75"
    s1_df = pl.read_csv(os.path.join(split_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(split_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(split_dir, "test_source3.tsv"), separator="\t")
    gt_df = pl.read_csv(os.path.join(split_dir, "ground_truth.tsv"), separator="\t")
    
    tgt_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    
    gt_dict = {}
    total_positives = 0
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        t_set = set(x.strip() for x in m.split(",")) if m else set()
        gt_dict[r["source1_entity_id"]] = t_set
        total_positives += len(t_set)
        
    s1_ids = s1_df["entity_id"].to_list()
    tgt_ids = tgt_df["entity_id"].to_list()
    
    # Texts: name + address
    s1_texts = [f"{r['business_name'] or ''} {r['business_address'] or ''}".strip().lower() for r in s1_df.iter_rows(named=True)]
    tgt_texts = [f"{r['business_name'] or ''} {r['business_address'] or ''}".strip().lower() for r in tgt_df.iter_rows(named=True)]
    
    print(f"Building char 2-4-gram TF-IDF matrix (10k S1, {len(tgt_texts):,} targets)...")
    t0 = time.time()
    
    # Char 2-4-gram vectorizer with sublinear tf scaling
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    
    # Fit on corpus and transform
    X_tgt = vec.fit_transform(tgt_texts)
    X_s1 = vec.transform(s1_texts)
    t_feat = time.time() - t0
    print(f"TF-IDF matrices built in {t_feat:.1f}s (Vocabulary: {len(vec.vocabulary_):,} features)")
    
    # Query in batches to keep memory tiny
    BATCH_SIZE = 1000
    recalls = {5: 0, 10: 0, 20: 0, 50: 0}
    max_k = 50
    
    t_search = time.time()
    for b_start in range(0, len(s1_ids), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(s1_ids))
        sub_s1 = X_s1[b_start:b_end]
        
        # Batch cosine similarities: (batch_size, n_targets)
        sims = (sub_s1 * X_tgt.T).toarray()
        
        for i_in_batch in range(b_end - b_start):
            idx_global = b_start + i_in_batch
            sid = s1_ids[idx_global]
            true_tgts = gt_dict.get(sid, set())
            if not true_tgts:
                continue
                
            # Top-50 target indices
            row_sims = sims[i_in_batch]
            top_k_indices = np.argpartition(row_sims, -max_k)[-max_k:]
            top_k_sorted = top_k_indices[np.argsort(-row_sims[top_k_indices])]
            retrieved_ids = [tgt_ids[j] for j in top_k_sorted]
            
            for k in recalls:
                recalls[k] += len(true_tgts & set(retrieved_ids[:k]))
                
    search_time = time.time() - t_search
    total_time = t_feat + search_time
    
    print(f"\n--- TF-IDF RESULTS: {country} ---")
    print(f"{'K':<5s} | {'Recall':<10s} | {'Reduction Ratio':<15s}")
    print("-" * 35)
    for k in [5, 10, 20, 50]:
        rec = recalls[k] / total_positives if total_positives > 0 else 0
        red = (1.0 - (k / len(tgt_ids))) * 100
        print(f"{k:<5d} | {rec*100:<9.2f}% | {red:.2f}%")
        
    print(f"\nSearch throughput: {len(s1_ids)/search_time:.1f} S1/sec ({search_time:.1f}s total)")
    print(f"Total time (fit + transform + query): {total_time:.1f}s")
    
    return {
        "country": country,
        "recall@10": recalls[10] / total_positives,
        "recall@20": recalls[20] / total_positives,
        "recall@50": recalls[50] / total_positives,
        "total_time": total_time,
    }

if __name__ == "__main__":
    benchmark_tfidf_country("India")
    benchmark_tfidf_country("US")
