<div align="center">

# 🏢 Amazon ML Challenge 2026: Business Entity Resolution
### High-Precision Multi-Source Record Linkage Pipeline Across Heterogeneous Enterprise Data

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Leaderboard Score](https://img.shields.io/badge/Official%20Score-0.983526-success.svg?logo=target&logoColor=white)](https://unstop.com)
[![Leaderboard Rank](https://img.shields.io/badge/Rank-%23743-orange.svg?logo=trophy&logoColor=white)](https://unstop.com)
[![Metric](https://img.shields.io/badge/Metric-Macro%20F0.5-purple.svg)](https://unstop.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Frameworks](https://img.shields.io/badge/Frameworks-LightGBM%20%7C%20XGBoost%20%7C%20PyTorch%20%7C%20Transformers-red.svg)](https://pytorch.org/)

</div>

---

## 👥 Team: Green Sync Innovators
* **Vijay Kasthuri K** (Team Lead)
* **Sudharsan E**
* **Akash SM**
* **Dinny Paul Navis C**

🏛 **Institution:** PSNA College of Engineering and Technology (PSNACET), Dindigul  
🏆 **Competition:** Amazon ML Challenge 2026 (72-Hour National Hackathon hosted on Unstop)

---

## 📌 Executive Summary

This repository houses the complete, audited, and fully reproducible pipeline developed by **Green Sync Innovators** for the **Amazon ML Challenge 2026**. 

Our solution tackles enterprise-scale **Business Entity Resolution (BER)** across three noisy, fragmented data sources without common keys. By casting the problem into a precision-first framework, our system couples **two-way IDF-weighted sparse candidate blocking** with a **multi-stage hybrid architecture** (LightGBM, 5-Fold Grouped XGBoost, and Cross-Encoder Transformers).

### 🏆 Key Achievements
* **Leaderboard Rank:** **#743** nationwide among thousands of competitive university teams.
* **Official Score:** **`0.983526`** (Macro $F_{0.5}$).
* **Competitive Margin:** Razor-thin gap of less than **0.0085 (0.85%)** from Rank #1 (`0.992082`).
* **Search Space Reduction:** **99.99994881% reduction ratio** against the $\approx 1.73 \times 10^{13}$ Cartesian product space.
* **Scale Evaluated:** Linked **1,732,544 Source 1 entities** against target sources, predicting **5,766,118 verified links** and isolating **101,804 true singletons**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Preprocessing"]
        A1["Raw Source 1 (TSV)"] & A2["Raw Source 2 (TSV)"] & A3["Raw Source 3 (TSV)"]
        B["Tab-separated Fast I/O & Parquet Storage"]
        C["Lexical & Legal Suffix Normalization\n(Corp, Pvt Ltd, LLC, Rd/St standardizer)"]
        D["Native-Script to Latin Transliteration Lexicon\n(Learned without external APIs)"]
        A1 & A2 & A3 --> B --> C --> D
    end

    subgraph BLOCKING["2. Candidate Generation (Blocking)"]
        E["Country Partitioning\n(US, India, Unseen France)"]
        F1["Inverted Index: Character Trigrams"]
        F2["Inverted Index: Composite Word Keys"]
        F3["Two-Way IDF Sparse Retrieval Engine"]
        D --> E --> F1 & F2 & F3
        F1 & F2 & F3 --> G["Candidate Pairs Filtered\n(8.84M Unique Candidate Pairs)"]
    end

    subgraph FEATURE_ENG["3. Feature Engineering Matrix (115+ Features)"]
        H1["Fuzzy Distance: Levenshtein, Jaro-Winkler, Token Sort/Set"]
        H2["Statistical & TF-IDF Cosine Similarities"]
        H3["Structural Matchers: House numbers & signed numeric diffs"]
        H4["Competitive Features: Intra-candidate margin & rank context"]
        G --> H1 & H2 & H3 & H4
    end

    subgraph MODELS["4. Multi-Stage Scoring Pipeline"]
        M1["Baseline: 2-Stage LightGBM\n(Stage 1 Pair + Stage 2 Collective)"]
        M2["v11 Variant: 5-Fold XGBoost +\nms-marco-MiniLM-L6-v2 Cross-Encoder"]
        M3["v12 Variant: 5-Fold XGBoost +\nBAAI/bge-reranker-base Cross-Encoder"]
        M4["Secondary Pipeline: Graph & Phonetic Embeddings"]
        H1 & H2 & H3 & H4 --> M1 & M2 & M3 & M4
    end

    subgraph ENSEMBLE["5. Decision Tuning & Post-Processing"]
        N1["Consensus Decision Gate\n(US & India Precision Pruning)"]
        N2["Zero-Shot France Passthrough\n(Calibrated Baseline Preserved)"]
        N3["Hungarian / Greedy One-Owner Constraint\n(Max 1 Reference Owner per Target Record)"]
        N4["Expected Macro F0.5 Cutoff Optimizer"]
        M1 & M2 & M3 & M4 --> N1 & N2
        N1 & N2 --> N3 --> N4
    end

    subgraph OUTPUT["6. Audited Competition Deliverables"]
        O1["matching_results.tsv\n(1,732,544 rows · 5,766,118 matches)"]
        O2["candidate_pairs.tsv\n(8,841,437 validated candidate pairs)"]
        N4 --> O1 & O2
    end

    classDef primary fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef success fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff;
    classDef highlight fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff;
    class G highlight;
    class N3,N4 primary;
    class O1,O2 success;
```

---

## 🔬 Core Methodology & Innovations

### 1. Robust Preprocessing & Self-Contained Transliteration
* **Legal Suffix Stripping & Standardization:** Homogenized regional enterprise indicators (`Corp`, `Corporation`, `Pvt Ltd`, `Private Limited`, `LLC`, `GmbH`, `SA`, `SAS`).
* **Address Normalization:** Standardized numerical street patterns, landmark markers, and postal code variants.
* **Learned Transliteration Lexicon (`learn_dict.py`):** Learned mapping dictionaries from native Indic/regional scripts directly from ground truth links without querying any third-party APIs or external search engines.

### 2. High-Recall Candidate Blocking (Sparse Two-Way Retrieval)
* Evaluating all combinations of 1.73M Source 1 records and 9.97M target records would require over **17 trillion pairwise comparisons**.
* Implemented **country-isolated two-way sparse retrieval**:
  * Inverted indexes over character n-grams, composite address keys, and phonetic representations.
  * Reduced the comparison space by **99.99994881%**, arriving at an average of only **5.103 candidates per record** while retaining **97.89%** of ground truth links.

### 3. Multi-Stage Hybrid ML Architecture
* **Stage-1 Pair Scoring (LightGBM & XGBoost):** Trained on 115 pairwise features including fuzzy string metrics, token rarity, numeric house-number differences, and missing-field indicators.
* **Stage-2 Collective Graph Context:** Modelled intra-candidate competition (e.g. difference in score between the 1st and 2nd best match for a record, candidate rank, and margin).
* **Transformer Cross-Encoders:**
  * Utilized `cross-encoder/ms-marco-MiniLM-L6-v2` (22M parameters) and `BAAI/bge-reranker-base` (278M parameters) to evaluate semantically ambiguous candidate pairs.
  * Both models strictly complied with the hackathon's parameter ceiling (< 8B parameters) and open-source licensing constraints.

### 4. Precision-First Macro $F_{0.5}$ Optimization
* The official evaluation metric was **Macro $F_{0.5}$**, where precision is weighted twice as heavily as recall:
  
  $$\text{Macro } F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}}$$

* **One-Owner Assignment:** A single target entity cannot belong to conflicting businesses; enforced greedy bipartite matching so each target record maps to exactly one Source 1 parent.
* **Ensemble Consensus Rules:**
  * Base predictions pruned when $v_{11} < 0.05$ or when rejected by secondary models with mean score $< 0.30$.
  * New links added only when both $v_{11}$ and $v_{12}$ demonstrated extreme confidence ($\ge 0.95$).
* **Zero-Shot Handling for France:** Synthetic validation generation (`synth.py`) was employed to guarantee robust behavior on unseen countries without manual label leakage.

---

## 📂 Repository Layout

```text
AMAZON-ML-CHALLEGE-2026/
├── .gitignore                         # Comprehensive ignore rules (data, large archives, binaries)
├── LICENSE                            # MIT License
├── MODEL_LICENSES.md                  # Open-source compliance report for all used models
├── README.md                          # Repository documentation & architecture guide
├── requirements.txt                   # Production Python 3.11 dependencies
├── build_final_package.py             # Packaging and delivery assembly script
├── prepare_submission_audit.py        # Independent audit, checksum verification, and stats generator
├── refine_submission_draft.py         # Submission post-processing pipeline
│
├── docs/
│   └── SOLUTION_REPORT.md             # Complete technical report submitted with the deliverables
│
├── environments/
│   ├── baseline-requirements.txt      # Pinned environment for LightGBM CPU baseline
│   ├── scoring-requirements.txt       # Pinned environment for Transformer & XGBoost models
│   └── BASELINE_REPRODUCE.md          # Step-by-step instructions for baseline verification
│
├── models/
│   └── baseline/                      # Audited LightGBM trained tree models (<12MB each)
│       ├── decoder.json               # Metric tuning thresholds
│       ├── lexicon.json               # Learned transliteration dictionary
│       ├── lgb_s1_A.txt               # Fold A Stage-1 LightGBM model
│       ├── lgb_s1_B.txt               # Fold B Stage-1 LightGBM model
│       ├── lgb_s2.txt                 # Stage-2 Collective LightGBM model
│       └── stage1.json                # Feature specification indices
│
├── provenance/
│   └── SHA256SUMS.json                # Cryptographic verification hashes of submitted TSVs
│
└── src/
    ├── ensemble_final.py              # Main entry point: Consensus ensembling & post-processing
    ├── validate_submission.py         # Official competition schema and ID validator
    ├── baseline/                      # Normalization, blocking, and LightGBM pipeline
    ├── scoring/                       # Sparse retrieval, pair features, and XGBoost modules
    ├── secondary/                     # Secondary graph and phonetic pipeline components
    └── kaggle_launchers/              # Exact Kaggle execution launchers with deterministic seeds
```

---

## 🚀 Reproduction & Setup Guide

### 1. Environment Installation
Clone the repository and install the verified Python 3.11 environment:

```bash
git clone https://github.com/vijaykasthurik/AMAZON-ML-CHALLEGE-2026.git
cd AMAZON-ML-CHALLEGE-2026
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup
Ensure the official dataset is organized in the following format:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Generate Final Ensemble
Execute the final post-processing and ensembling engine:

```bash
python src/ensemble_final.py \
  --baseline work/file1/matching_results.tsv \
  --secondary work/file2/matching_results.tsv \
  --v11-scores work/v11/test_scores.parquet \
  --v12-scores work/v12/test_scores.parquet \
  --test-source1 /path/to/dataset/test/test_source1.tsv \
  --output-dir reproduced_output
```

### 4. Validate Submission
Run the official competition compliance test suite:

```bash
python src/validate_submission.py \
  --matching reproduced_output/matching_results.tsv \
  --candidate reproduced_output/candidate_pairs.tsv \
  --test-dir /path/to/dataset/test \
  --check-ids
```

---

## 🔒 Fair Play & Compliance

* **Zero External Data:** No third-party APIs, search engines, geocoders, commercial entity databases, or external web scraping were used.
* **Deterministic Seeds:** All fold assignments, random state generators, and sampling rates used fixed seeds (`seed=42`).
* **License Adherence:** All foundation models (`ms-marco-MiniLM-L6-v2`, `bge-reranker-base`, `LightGBM`, `XGBoost`) adhere to permissive Apache-2.0, MIT, or BSD licensing terms.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
