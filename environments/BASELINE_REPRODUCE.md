# file1 – code for `matching_results.tsv` (SHA-256 bf9d90aaddb06c5c…)

This package regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the official dataset only. No notebooks, Kaggle or Colab were used, and no
external data, APIs, web lookups or pretrained models. The only learned models
are LightGBM classifiers trained from scratch on the provided training pairs.

## Contents

| path | what it is |
|---|---|
| `code/business_entity_resolution/src/` | complete source (all helper modules included) |
| `code/business_entity_resolution/requirements.txt` | pinned dependencies |
| `code/business_entity_resolution/README.md` | code map |
| `models/` | the exact trained models used for this file: `lgb_s1_A.txt`, `lgb_s1_B.txt` (stage-1 fold models), `lgb_s2.txt` (stage 2), `stage1.json` (stage-1 feature list, validation scores), `decoder.json` (stage-2 feature list, chosen decoder `ef_0.1`) |
| `artifacts/lexicon.json` | native-script → Latin word lexicon learned from the training pairs by `preprocess.py` (regenerated automatically if absent) |
| `output/matching_results.tsv` | the submitted prediction file |
| `output/candidate_pairs.tsv` | the exact candidate set scored by the models (after all pruning) |
| `SHA256SUMS.txt` | fingerprints of the two output files |
| `Documentation_template.md` | methodology write-up |

## Environment

- Windows 11, Python 3.10.11, CPU only (12 cores, 16 GB RAM); no GPU.
- `pip install -r code/business_entity_resolution/requirements.txt`
  (polars 1.44.2, pyarrow 25.0.1, numpy 1.26.4, scipy 1.15.3, numba 0.66.0,
  llvmlite 0.48.0, lightgbm 4.7.0, rapidfuzz 3.14.5).

## Paths

Set before running (defaults shown are the ones we used):

- `BER_DATA_DIR` – folder containing `train/` and `test/` of the official dataset
  (default `D:/aws/datasets/student_resource/dataset`)
- `BER_WORK_DIR` – cache / model folder, needs ~15 GB (default `D:/aws/work`)
- `BER_OUTPUT_DIR` – where the TSVs are written (default `<BER_WORK_DIR>/output`)

## Run order (from `code/business_entity_resolution/src/`)

```
python preprocess.py          # lexicon + normalised parquet for the 6 source files
python candidates.py train    # blocking on train (+ recall report)
python candidates.py test     # blocking on test
python train.py               # stage-1 fold models A/B + out-of-fold p1 (seed 42, 30% of queries per fold)
python stage2.py              # stage-2 model; picks the decoder on held-out training businesses
python infer.py               # scores all test candidates, writes output/matching_results.tsv + candidate_pairs.tsv
python check_submission.py    # format checks + organisers' validator
```

To score with the shipped models instead of retraining, copy `models/*` into
`<BER_WORK_DIR>/models/` after the blocking steps and run only `python infer.py`
(it also needs `train.py`'s `prepare` step, which `infer.py` calls itself).

Seeds are fixed (`config.SEED = 42`, LightGBM `seed=42`, validation/fold split by
fixed-seed hashes of the ids). LightGBM multithreading can cause tiny
floating-point differences between machines; the decisions are stable to these.

## Pipeline summary

1. **Normalisation** (`preprocess.py`, `translit.py`): NFKC, zero-width removal,
   accent stripping, Indian-script words via a lexicon learned from training
   pairs with a rule-based romaniser as fallback; legal forms canonicalised;
   aliases (dba/aka/t/a), websites, digit-for-letter typos; street types,
   ordinals, states/regions as admin tokens.
2. **Blocking** (`blocking.py`, `keys.py`, `candidates.py`): per country, each
   S2/S3 record retrieves S1 records from two TF-IDF inverted indexes (character
   3-grams and composite word keys), six top-10 lists, exact rescoring, a pair
   kept if in the exact top 3 of any list (`cands_*.parquet`); the models use
   pairs in the top 2 (`K_MODEL = 2`) – this final set is `candidate_pairs.tsv`.
3. **Stage 1** (`features.py`, `align.py`, `train.py`): 115 pair features
   (string similarities, word alignment, arithmetic house-number gaps incl.
   signed difference, legal-form differences, word rarity, acronyms,
   retrieval scores and candidate-context features). Two LightGBM fold models.
4. **Stage 2** (`stage2.py`): LightGBM on out-of-fold p1 + candidate-graph
   features + 12 stage-1 features.
5. **Decoding** (`decode.py`): each S2/S3 record keeps only its best S1
   candidate; per S1 business the subset maximising expected F0.5 is kept
   (`ef_0.1`).

## Validation (measured)

Held-out 4% of training S1 businesses (88,408 entities, all their candidate
queries excluded from fitting), exact macro F0.5 incl. singletons:
stage 1 0.98653, **stage 2 0.98744**. Blocking recall on train: 97.89% of true
links in the model's candidate set.
