# Model and library licenses

- LightGBM: MIT. The baseline models were trained from scratch on the provided data.
- XGBoost: Apache-2.0. The pair and collective models were trained on the provided data.
- `cross-encoder/ms-marco-MiniLM-L6-v2`: Apache-2.0; approximately 22 million parameters.
- `BAAI/bge-reranker-base`: MIT; approximately 278 million parameters.
- Polars: MIT; PyArrow: Apache-2.0; NumPy/SciPy/scikit-learn: BSD-family licenses;
  RapidFuzz: MIT; PyTorch: BSD-style; Transformers: Apache-2.0.

Both transformer checkpoints are below the challenge’s 8-billion-parameter limit.
They are used as text-pair scorers and are fine-tuned only on challenge training pairs.
