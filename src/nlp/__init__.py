"""NLP sub-package for Traditional Chinese text analysis.

Provides a text EDA module, a pluggable model benchmark harness and
GPU/device utilities. Heavy dependencies (torch, transformers, spacy) are
imported lazily inside the modules that need them so the pure-Python core
stays importable everywhere.
"""
