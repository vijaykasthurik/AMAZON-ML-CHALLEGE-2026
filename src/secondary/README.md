# Business Entity Resolution — Amazon ML Challenge 2026
## Exact Reproduction Guide

### Environment & Python Version
- **Python Version**: Python 3.10, 3.11, or 3.12
- **Hardware**: Standard 8-core CPU, 16GB+ RAM (no GPU required for reproduction).

### Quick Setup
`ash
pip install -r code/business_entity_resolution/requirements.txt
`

### Execution Commands (From Raw Data to Final Predictions)
Ensure the competition dataset is in dataset/ (or student_resource/dataset/).

1. **Option A: Direct Fast Inference (Using Pretrained Checkpoints, ~15 mins)**:
   `ash
   python code/business_entity_resolution/src/infer_ensemble_optimized.py
   `
   This loads models/lgbm_v2.txt and models/catboost_v2.cbm, performs multi-key blocking, extracts 24 pairwise features, applies ensemble inference (0.70 LGBM + 0.30 CatBoost), and outputs:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv

2. **Option B: Full End-to-End Retraining from Scratch**:
   `ash
   python code/business_entity_resolution/run_pipeline.py
   `
   Or step-by-step:
   `ash
   # Step 1: Mine data-driven Indic transliteration loanwords
   python code/business_entity_resolution/mine_indic_loanwords.py

   # Step 2: Train LightGBM v2 on 24 pairwise features with 5-fold GroupKFold
   python code/business_entity_resolution/train_lgbm_v2.py

   # Step 3: Run production ensemble inference
   python code/business_entity_resolution/src/infer_ensemble_optimized.py
   `

### Architecture, Features & Thresholds
- **Blocking**: Multi-key inverted index (sorted word-pair, word+number, addr-word+number, transliterated Indic names).
- **Features (24)**: RapidFuzz token-sort/set ratios, char 2/3/4-gram Jaccard, cross-script transliteration, Indic loanword map, house-number joint key, street-core match, distinctive token Jaccard, name frequency chain detection.
- **Reranker**: LightGBM v2 (70%) + CatBoost v2 (30%).
- **Thresholds**: France: 0.85, US: 0.90, India: 0.90 (tuned via OOF macro F0.5).
- **Exclusivity**: Strict global 1-to-1 target exclusivity (one_owner).
