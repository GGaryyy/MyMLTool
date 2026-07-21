"""Text EDA for Traditional-Chinese text classification datasets.

:func:`run_eda` is pure computation (no file writing — see
:mod:`src.nlp.report` for writers). It produces char/token length
distributions with the over-512-token ratio for BERT planning, class
balance, vocabulary statistics on segmenter tokens, data-quality issues
(exact/near duplicates, label conflicts, empty/short texts), PII and
normalization scans, optional split-leakage checks and human-readable
warnings.

Near-duplicate heuristic: documents are compared by ``NEAR_DUP_SHINGLE``-char
shingle Jaccard. Above ``NEAR_DUP_CAP`` documents an all-pairs scan is too
slow, so docs are bucketed by length rounded to ``NEAR_DUP_LENGTH_BUCKET``
chars and only compared within a bucket. A Jaccard >= 0.9 pair necessarily
has similar lengths, but true near-dups straddling a bucket boundary are
missed — an accepted trade-off to stay O(manageable).

Privacy: top tokens / n-grams are passed through the PII masks before they
are stored, so a report can never surface a raw 身分證 number or phone
number through the vocabulary tables.
"""

import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from src.nlp.analysis.pii import (
    ADDRESS_PATTERN,
    PII_PATTERNS,
    NormalizationReport,
    PiiReport,
    check_normalization,
    mask_snippet,
    scan_pii,
)
from src.nlp.config import RunConfig
from src.nlp.labels import build_label_space, class_distribution
from src.nlp.segment import get_segmenter

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BERT_TOKEN_LIMIT = 512
NEAR_DUP_SHINGLE = 8   # char-shingle size
NEAR_DUP_JACCARD = 0.9
NEAR_DUP_CAP = 1000
NEAR_DUP_LENGTH_BUCKET = 50
SHORT_TEXT_CHARS = 10
TOP_TOKENS_N = 30
TOP_NGRAMS_N = 20
MAX_CONFLICT_EXAMPLES = 20
MAX_LEAKED_EXAMPLES = 20
MINORITY_FRACTION = 0.1
IMBALANCE_WARN_RATIO = 5.0
OVER_LIMIT_WARN_RATIO = 0.05


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class LengthStats:
    """Char- and token-level length distribution summary.

    ``char_lengths`` keeps the raw per-document char counts so the report
    writer can draw the length histogram without re-reading the corpus.
    """

    mean: float
    median: float
    p95: float
    max: int
    token_mean: float
    token_median: float
    token_p95: float
    token_max: int
    over_512_ratio: float
    char_lengths: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassBalance:
    """Class counts plus imbalance diagnostics."""

    counts: dict
    imbalance_ratio: float
    entropy: float
    minority_classes: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VocabStats:
    """Vocabulary statistics over segmenter tokens (PII-masked)."""

    vocab_size: int
    ttr: float
    hapax_ratio: float
    top_tokens: list
    top_bigrams: list
    top_trigrams: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityIssues:
    """Duplicate / conflict / degenerate-text diagnostics.

    ``exact_duplicate_groups`` counts distinct text values occurring more
    than once; ``exact_duplicate_docs`` counts every document belonging to
    such a group. ``conflict_examples`` holds row-index groups only (no
    text, no PII).
    """

    exact_duplicate_groups: int
    exact_duplicate_docs: int
    near_duplicate_pairs: int
    label_conflicts: int
    conflict_examples: list
    empty_texts: int
    short_texts: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeakageReport:
    """Exact-text overlap between data splits (row indices only)."""

    pairs: dict
    leaked_examples: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextEdaReport:
    """Bundle of every EDA result, JSON-ready via :meth:`to_dict`."""

    n_docs: int
    task_type: str
    length: LengthStats
    balance: ClassBalance
    vocab: VocabStats
    quality: QualityIssues
    pii: PiiReport
    normalization: NormalizationReport
    leakage: Optional[LeakageReport]
    warnings: list

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_eda(texts: Sequence[str], raw_labels: Sequence, config: RunConfig,
            splits: Optional[dict] = None) -> TextEdaReport:
    """Run the full text EDA and return a :class:`TextEdaReport`.

    ``splits`` optionally maps split names to row-index lists; when given,
    exact-text leakage is computed between every pair of splits.
    """
    if not isinstance(config, RunConfig):
        raise TypeError(f"config must be a RunConfig, got {type(config).__name__}")
    text_list, label_list = _validated_inputs(texts, raw_labels)

    label_space, y = build_label_space(
        label_list, config.data.task_type, separator=config.data.label_separator
    )
    balance = _class_balance(class_distribution(label_space, y))

    segmenter = get_segmenter(config.segment.engine)
    token_lists = segmenter.tokenize_batch(text_list)
    length = _length_stats(text_list, token_lists)
    vocab = _vocab_stats(token_lists)
    quality = _quality_issues(text_list, label_list)
    pii_report = scan_pii(text_list)
    normalization = check_normalization(text_list)
    leakage = _leakage_report(text_list, splits) if splits is not None else None

    return TextEdaReport(
        n_docs=len(text_list),
        task_type=config.data.task_type,
        length=length,
        balance=balance,
        vocab=vocab,
        quality=quality,
        pii=pii_report,
        normalization=normalization,
        leakage=leakage,
        warnings=_build_warnings(balance, length, quality, pii_report),
    )


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _validated_inputs(texts: Sequence[str], raw_labels: Sequence) -> tuple:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of documents, not a single str")
    text_list = list(texts)
    label_list = list(raw_labels)
    if len(text_list) != len(label_list):
        raise ValueError(
            f"texts and raw_labels length mismatch: {len(text_list)} vs {len(label_list)}"
        )
    if not text_list:
        raise ValueError("Cannot run EDA on an empty text list")
    for i, text in enumerate(text_list):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
    return text_list, label_list


# --------------------------------------------------------------------------- #
# Statistics builders
# --------------------------------------------------------------------------- #
def _length_stats(texts: list, token_lists: list) -> LengthStats:
    char_lengths = [len(text) for text in texts]
    token_counts = [len(tokens) for tokens in token_lists]
    over = sum(1 for count in token_counts if count > BERT_TOKEN_LIMIT)
    return LengthStats(
        mean=float(np.mean(char_lengths)),
        median=float(np.median(char_lengths)),
        p95=float(np.percentile(char_lengths, 95)),
        max=int(max(char_lengths)),
        token_mean=float(np.mean(token_counts)),
        token_median=float(np.median(token_counts)),
        token_p95=float(np.percentile(token_counts, 95)),
        token_max=int(max(token_counts)),
        over_512_ratio=over / len(token_counts),
        char_lengths=char_lengths,
    )


def _class_balance(counts: dict) -> ClassBalance:
    values = list(counts.values())
    max_count = max(values)
    min_count = min(values)
    # Classes derive from the data so min_count >= 1; inf-guard regardless.
    ratio = float(max_count) / min_count if min_count > 0 else float("inf")
    minority = [c for c, n in counts.items() if n < MINORITY_FRACTION * max_count]
    return ClassBalance(
        counts=dict(counts),
        imbalance_ratio=ratio,
        entropy=_normalized_entropy(values),
        minority_classes=minority,
    )


def _normalized_entropy(values: list) -> float:
    """Shannon entropy of the count distribution, normalized to [0, 1].

    Base is ``log(n_classes)``; a single class is defined as 1.0 (perfectly
    balanced for its class count).
    """
    if len(values) <= 1:
        return 1.0
    total = float(sum(values))
    probs = [v / total for v in values if v > 0]
    raw = -sum(p * math.log(p) for p in probs)
    return raw / math.log(len(values))


def _vocab_stats(token_lists: list) -> VocabStats:
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    for tokens in token_lists:
        unigrams.update(tokens)
        # N-grams join tokens without a separator: segmenter tokens are CJK
        # chars / ASCII runs, so the joined form reads naturally in Chinese.
        bigrams.update("".join(g) for g in zip(tokens, tokens[1:]))
        trigrams.update("".join(g) for g in zip(tokens, tokens[1:], tokens[2:]))
    total_tokens = sum(unigrams.values())
    vocab_size = len(unigrams)
    hapax = sum(1 for count in unigrams.values() if count == 1)
    return VocabStats(
        vocab_size=vocab_size,
        ttr=vocab_size / total_tokens if total_tokens else 0.0,
        hapax_ratio=hapax / vocab_size if vocab_size else 0.0,
        top_tokens=_top_masked(unigrams, TOP_TOKENS_N),
        top_bigrams=_top_masked(bigrams, TOP_NGRAMS_N),
        top_trigrams=_top_masked(trigrams, TOP_NGRAMS_N),
    )


def _top_masked(counter: Counter, n: int) -> list:
    """Top-n (token, count) pairs, count-desc then token-asc, PII-masked."""
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:n]
    return [(_mask_pii_text(token), count) for token, count in ordered]


def _mask_pii_text(text: str) -> str:
    """Mask any PII-pattern match inside a vocabulary string."""
    for pattern in (*PII_PATTERNS.values(), ADDRESS_PATTERN):
        text = pattern.sub(lambda m: mask_snippet(m.group(0)), text)
    return text


# --------------------------------------------------------------------------- #
# Quality issues
# --------------------------------------------------------------------------- #
def _quality_issues(texts: list, raw_labels: list) -> QualityIssues:
    rows_by_text: dict = {}
    for i, text in enumerate(texts):
        rows_by_text.setdefault(text, []).append(i)

    dup_groups = [rows for rows in rows_by_text.values() if len(rows) > 1]
    conflicts = [
        sorted(rows)
        for rows in dup_groups
        if len({str(raw_labels[i]) for i in rows}) > 1
    ]
    return QualityIssues(
        exact_duplicate_groups=len(dup_groups),
        exact_duplicate_docs=sum(len(rows) for rows in dup_groups),
        near_duplicate_pairs=_near_duplicate_pairs(texts),
        label_conflicts=len(conflicts),
        conflict_examples=conflicts[:MAX_CONFLICT_EXAMPLES],
        empty_texts=sum(1 for text in texts if not text.strip()),
        short_texts=sum(1 for text in texts if len(text) < SHORT_TEXT_CHARS),
    )


def _shingles(text: str) -> frozenset:
    if len(text) < NEAR_DUP_SHINGLE:
        return frozenset()
    return frozenset(
        text[i:i + NEAR_DUP_SHINGLE] for i in range(len(text) - NEAR_DUP_SHINGLE + 1)
    )


def _near_duplicate_pairs(texts: list) -> int:
    """Count pairs with char-shingle Jaccard >= ``NEAR_DUP_JACCARD``.

    Exact-equal pairs are excluded (reported as exact duplicates instead).
    See the module docstring for the >``NEAR_DUP_CAP`` bucketing heuristic.
    A cheap size prefilter skips pairs whose shingle-set sizes already bound
    Jaccard below the threshold (J <= min/max of the two set sizes).
    """
    shingle_sets = [_shingles(text) for text in texts]
    if len(texts) > NEAR_DUP_CAP:
        buckets: dict = {}
        for i, text in enumerate(texts):
            buckets.setdefault(round(len(text) / NEAR_DUP_LENGTH_BUCKET), []).append(i)
        groups = list(buckets.values())
    else:
        groups = [list(range(len(texts)))]

    pairs = 0
    for group in groups:
        for a_pos, i in enumerate(group):
            set_i = shingle_sets[i]
            if not set_i:
                continue
            for j in group[a_pos + 1:]:
                set_j = shingle_sets[j]
                if not set_j or texts[i] == texts[j]:
                    continue
                small = min(len(set_i), len(set_j))
                large = max(len(set_i), len(set_j))
                if small / large < NEAR_DUP_JACCARD:
                    continue
                inter = len(set_i & set_j)
                union = small + large - inter
                if union and inter / union >= NEAR_DUP_JACCARD:
                    pairs += 1
    return pairs


# --------------------------------------------------------------------------- #
# Split leakage
# --------------------------------------------------------------------------- #
def _leakage_report(texts: list, splits: dict) -> LeakageReport:
    if not isinstance(splits, dict):
        raise TypeError(
            f"splits must be a dict of split-name -> row indices, got {type(splits).__name__}"
        )
    normalized = {str(name): _validated_rows(name, rows, len(texts))
                  for name, rows in splits.items()}

    pairs: dict = {}
    examples: list = []
    for name_a, name_b in combinations(normalized, 2):
        first_a = _first_row_by_text(texts, normalized[name_a])
        first_b = _first_row_by_text(texts, normalized[name_b])
        shared = [text for text in first_a if text in first_b]
        key = f"{name_a}~{name_b}"
        pairs[key] = len(shared)
        for text in shared:
            if len(examples) >= MAX_LEAKED_EXAMPLES:
                break
            examples.append({"pair": key, "row_a": first_a[text], "row_b": first_b[text]})
    return LeakageReport(pairs=pairs, leaked_examples=examples)


def _validated_rows(name, rows, n_docs: int) -> list:
    try:
        raw = list(rows)
    except TypeError as exc:
        raise ValueError(f"Split '{name}' indices must be an iterable of ints") from exc
    cleaned = []
    for row in raw:
        try:
            idx = int(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Split '{name}' has a non-integer row index: {row!r}") from exc
        if not 0 <= idx < n_docs:
            raise ValueError(f"Split '{name}' row index {idx} out of range [0, {n_docs})")
        cleaned.append(idx)
    return cleaned


def _first_row_by_text(texts: list, rows: list) -> dict:
    first: dict = {}
    for row in rows:
        first.setdefault(texts[row], row)
    return first


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
def _build_warnings(balance: ClassBalance, length: LengthStats,
                    quality: QualityIssues, pii_report: PiiReport) -> list:
    flags = []
    if balance.imbalance_ratio > IMBALANCE_WARN_RATIO:
        flags.append(
            f"類別不平衡（imbalance_ratio={balance.imbalance_ratio:.1f} > "
            f"{IMBALANCE_WARN_RATIO:g}）：建議設定 class_weight=balanced，"
            "並以 f1_macro 作為主要評估指標。"
        )
    if length.over_512_ratio > OVER_LIMIT_WARN_RATIO:
        flags.append(
            f"{length.over_512_ratio:.1%} 的文件 token 數超過 BERT 上限 "
            f"{BERT_TOKEN_LIMIT}：建議 BERT 系列模型改用滑動視窗"
            "（sliding window）切塊處理長文。"
        )
    if pii_report.docs_with_pii > 0:
        flags.append(
            f"{pii_report.docs_with_pii} 份文件疑似含個資（PII）：報告內已遮罩，"
            "建議先完成去識別化（de-identification）再進行訓練與資料釋出。"
        )
    if quality.label_conflicts > 0:
        flags.append(
            f"{quality.label_conflicts} 組文本完全相同但標籤不同（label conflict）："
            "建議人工複核標註品質。"
        )
    if quality.exact_duplicate_docs > 0:
        flags.append(
            f"{quality.exact_duplicate_docs} 份文件完全重複"
            f"（{quality.exact_duplicate_groups} 組）：建議切分資料前先去重，"
            "避免訓練/測試洩漏使指標虛高。"
        )
    return flags
