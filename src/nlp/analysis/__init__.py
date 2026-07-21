"""Model-selection analyses for Chinese text data: task difficulty, label quality,
per-class keywords and PII/normalization scanning.

All modules here run CPU-only on sklearn/scipy by default; heavier vector
sources (sentence embeddings) are optional and lazily imported.
"""
