"""Deterministic synthetic 公文 generator for tests and demos.

ALL content is FICTIONAL. Agency names, reference numbers, contact names and
every PII-looking string (身分證 number ``A123456789``, phone
``02-2345-6789``, mobile ``0912-345-678``) are FAKE values injected on
purpose so cleaning / EDA code has defects to find. No real document,
agency decision or person is represented.

Documents follow the Taiwan government document style (機關 + 文別 +
發文字號 + 主旨 + 說明) with per-topic vocabulary, so the six ``TOPICS`` are
separable by a text classifier. Generation draws from
``numpy.random.default_rng(seed)`` exclusively — the same arguments always
produce the identical DataFrame. Each base document embeds a unique
發文字號 serial, so exact text collisions only exist when defect injection
creates them deliberately.

Note: ``n_docs >= 20`` is recommended so class-distribution and
defect-injection statistics are meaningful; smaller values are for smoke
tests only. This module is importable with only numpy + pandas installed.
"""

import numpy as np
import pandas as pd

from src.nlp.config import DEFAULT_LABEL_SEPARATOR

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODES = ("balanced", "imbalanced", "multilabel")
TOPICS = ("人事", "預算", "資訊安全", "採購", "法規", "教育訓練")
IMBALANCED_PROBS = (0.44, 0.25, 0.15, 0.08, 0.05, 0.03)
RECOMMENDED_MIN_DOCS = 20

AGENCIES = (
    "臺中市政府資訊局",
    "南區服務中心",
    "新北市政府人事處",
    "臺南市政府財政局",
    "高雄市政府法制局",
    "桃園市政府採購管理處",
    "基隆市政府教育處",
    "花蓮縣政府行政暨研考處",
)
DOC_TYPES = ("函", "公告", "開會通知單", "簽")
TOPIC_VOCAB = {
    "人事": ("人員陞遷", "差勤管理", "銓敘部函釋", "考績作業", "職務代理", "退休撫卹"),
    "預算": ("歲出預算編列", "經費核銷", "追加減預算", "會計月報", "預算執行率", "統籌分配稅款"),
    "資訊安全": ("資通安全管理法", "弱點掃描", "端點防護", "資安事件通報", "社交工程演練", "資訊資產盤點"),
    "採購": ("政府採購法", "公開招標", "底價核定", "履約管理", "驗收作業", "最有利標評選"),
    "法規": ("行政程序法", "法規命令發布", "自治條例修正", "函釋彙編", "法制作業注意事項", "法規鬆綁建議"),
    "教育訓練": ("公務人員訓練計畫", "數位學習課程", "終身學習時數", "講習班報名", "訓練需求調查", "在職進修申請"),
}
SUBJECT_TEMPLATES = (
    "檢送{phrase}相關資料一份",
    "為辦理{phrase}事宜",
    "有關{phrase}一案",
    "函轉{phrase}相關規定一案",
    "公告本機關{phrase}實施計畫",
)
BODY_TEMPLATES = (
    "依{phrase}相關規定辦理。",
    "旨揭{phrase}業經本機關核定，請依說明段配合辦理。",
    "請於文到七日內查復{phrase}辦理情形。",
    "檢附{phrase}一覽表乙份，請惠予配合。",
    "本案{phrase}所需經費由年度相關預算項下支應。",
    "請轉知所屬確實依{phrase}規定辦理並列管追蹤。",
)
CLOSINGS = ("請查照。", "請鑒核。", "請照辦。", "請查照轉知。")

LONG_DOC_MIN_CHARS = 600
LONG_DOC_FRACTION = 0.05
DUP_FRACTION = 0.02
CONFLICT_FRACTION = 0.01
PII_FRACTION = 0.02
FULLWIDTH_FRACTION = 0.02
NEAR_EMPTY_TEXTS = ("函", "主旨：")
# FAKE PII: fictional person, identifiers invalid by construction.
PII_CLAUSE = "聯絡人王小明，身分證字號A123456789，電話02-2345-6789，行動電話0912-345-678。"
# Full-width digits/letters plus OCR-noise characters (〇 vs 0, ｌ vs 1).
FULLWIDTH_CLAUSE = "案件編號１２３ＡＢ－〇ｌ號，全形字元供正規化檢核。"

_ZH_DIGITS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")


def _zh_ordinal(n: int) -> str:
    """Chinese numeral for 說明 clause numbering (一, 二, ... 十一, ...)."""
    if n < 10:
        return _ZH_DIGITS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        text = "十" if tens == 1 else _ZH_DIGITS[tens] + "十"
        return text + (_ZH_DIGITS[ones] if ones else "")
    return str(n)


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _sample_topics(rng: np.random.Generator, mode: str) -> list[str]:
    """Draw the topic(s) for one document; multilabel draws 1-3 distinct."""
    if mode == "multilabel":
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(TOPICS), size=k, replace=False)
        return sorted(TOPICS[int(i)] for i in idx)
    if mode == "imbalanced":
        return [str(rng.choice(TOPICS, p=IMBALANCED_PROBS))]
    return [str(rng.choice(TOPICS))]


def _render_clause(rng: np.random.Generator, vocab: list[str]) -> str:
    return _pick(rng, BODY_TEMPLATES).format(phrase=_pick(rng, vocab))


def _assemble(agency: str, doc_type: str, serial: str, subject: str,
              closing: str, clauses: list[str]) -> str:
    body = "".join(f"{_zh_ordinal(i)}、{clause}" for i, clause in enumerate(clauses, start=1))
    return f"{agency}{doc_type}\n發文字號：{serial}\n主旨：{subject}，{closing}\n說明：{body}"


def _compose_doc(rng: np.random.Generator, topics: list[str], serial: str,
                 make_long: bool) -> str:
    """Build one document; long docs keep appending 說明 clauses past 600 chars."""
    vocab = [phrase for topic in topics for phrase in TOPIC_VOCAB[topic]]
    agency = _pick(rng, AGENCIES)
    doc_type = _pick(rng, DOC_TYPES)
    subject = _pick(rng, SUBJECT_TEMPLATES).format(phrase=_pick(rng, vocab))
    closing = _pick(rng, CLOSINGS)
    clauses = [_render_clause(rng, vocab) for _ in range(int(rng.integers(1, 7)))]
    text = _assemble(agency, doc_type, serial, subject, closing, clauses)
    while make_long and len(text) <= LONG_DOC_MIN_CHARS:
        clauses.append(_render_clause(rng, vocab))
        text = _assemble(agency, doc_type, serial, subject, closing, clauses)
    return text


def _scaled_count(fraction: float, n_docs: int) -> int:
    """Defect count for one class: scales with n_docs, at least 1."""
    return max(1, int(round(fraction * n_docs)))


def _partition(pool: np.ndarray, sizes: list[int]) -> list[list[int]]:
    """Split a permuted index pool into consecutive disjoint groups.

    Returns one group per requested size plus the leftover indices as the
    final group. Groups shrink (possibly to empty) when the pool runs out.
    """
    groups, cursor = [], 0
    for size in sizes:
        groups.append([int(j) for j in pool[cursor:cursor + size]])
        cursor += size
    groups.append([int(j) for j in pool[cursor:]])
    return groups


def _different_label(rng: np.random.Generator, original: str, mode: str) -> str:
    """A valid label for ``mode`` that differs from ``original``."""
    if mode == "multilabel":
        for _ in range(8):
            candidate = DEFAULT_LABEL_SEPARATOR.join(_sample_topics(rng, mode))
            if candidate != original:
                return candidate
        # Extremely unlikely fallback: combos hold at most 3 of 6 topics,
        # so a leftover topic always exists.
        original_topics = set(original.split(DEFAULT_LABEL_SEPARATOR))
        return next(t for t in TOPICS if t not in original_topics)
    others = [t for t in TOPICS if t != original]
    return _pick(rng, others)


def _inject_defects(rng: np.random.Generator, texts: list[str],
                    labels: list[str], mode: str) -> None:
    """Overwrite/extend rows in place with the documented defect classes.

    Duplicate and conflict rows copy their text from rows that receive no
    other defect, so the injected pairs stay intact in the final frame.
    """
    n = len(texts)
    if n < 2:
        return
    pool = rng.permutation(np.arange(1, n))
    sizes = [
        _scaled_count(DUP_FRACTION, n),
        _scaled_count(CONFLICT_FRACTION, n),
        len(NEAR_EMPTY_TEXTS),
        _scaled_count(PII_FRACTION, n),
        _scaled_count(FULLWIDTH_FRACTION, n),
    ]
    dup_rows, conflict_rows, stub_rows, pii_rows, fullwidth_rows, clean = _partition(pool, sizes)
    sources = [0] + clean  # rows guaranteed to stay defect-free

    for j in dup_rows:
        src = sources[int(rng.integers(len(sources)))]
        texts[j], labels[j] = texts[src], labels[src]
    for j in conflict_rows:
        src = sources[int(rng.integers(len(sources)))]
        texts[j] = texts[src]
        labels[j] = _different_label(rng, labels[src], mode)
    for j, stub in zip(stub_rows, NEAR_EMPTY_TEXTS):
        texts[j] = stub
    for j in pii_rows:
        texts[j] = texts[j] + PII_CLAUSE
    for j in fullwidth_rows:
        texts[j] = texts[j] + FULLWIDTH_CLAUSE


def generate_synthetic_gov_docs(mode: str = "balanced", n_docs: int = 200,
                                seed: int = 0, inject_defects: bool = True) -> pd.DataFrame:
    """Generate a fictional 公文 dataset as ``DataFrame[["text", "label"]]``.

    Columns are exactly ``["text", "label"]`` — text first, label LAST,
    matching the repo convention. Multilabel labels are ``"|"``-joined
    sorted topic combinations (1-3 topics per document). Roughly 5% of
    documents (always at least one) exceed 600 characters so the
    over-512-token EDA check has signal.

    ``inject_defects=True`` deterministically adds, scaled with ``n_docs``
    (each defect class present at least once for ``n_docs >= 50``):

    - ~2% exact duplicate rows (text and label copied from a clean row)
    - ~1% label conflicts (same text as a clean row, different label)
    - 2 near-empty texts (``"函"`` / ``"主旨："``)
    - ~2% docs with FAKE fictional PII (A123456789 / 02-2345-6789 /
      0912-345-678)
    - ~2% docs with full-width digits/letters (１２３ＡＢ) and OCR-noise
      characters (〇, ｌ) for normalization checks
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    if n_docs < 1:
        raise ValueError(f"n_docs must be >= 1, got {n_docs}")

    rng = np.random.default_rng(seed)
    n_long = min(n_docs, _scaled_count(LONG_DOC_FRACTION, n_docs))
    long_indices = {int(i) for i in rng.choice(n_docs, size=n_long, replace=False)}

    texts: list[str] = []
    labels: list[str] = []
    for i in range(n_docs):
        topics = _sample_topics(rng, mode)
        serial = f"府授字第114{i:05d}號"
        texts.append(_compose_doc(rng, topics, serial, make_long=i in long_indices))
        labels.append(DEFAULT_LABEL_SEPARATOR.join(topics))

    if inject_defects:
        _inject_defects(rng, texts, labels, mode)
    return pd.DataFrame({"text": texts, "label": labels})


def texts_and_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a generated frame into plain ``(texts, labels)`` string lists."""
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'text' and 'label' columns")
    return [str(t) for t in df["text"]], [str(lab) for lab in df["label"]]
