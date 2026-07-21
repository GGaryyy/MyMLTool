"""Taiwan-focused PII scanning and text-normalization checks for Chinese text EDA.

Detects Taiwan-style PII (身分證字號, 行動電話, 市話, email, 地址 heuristic)
plus normalization hazards: full-width digits/letters, OCR look-alike
characters and documents mixing half-width and full-width digits. Matched
snippets are never stored raw — :func:`mask_snippet` keeps only the first
and last character, so downstream reports contain no usable PII.

Pure stdlib (``re`` / ``dataclasses``); safe to import anywhere.
"""

import re
from dataclasses import asdict, dataclass
from typing import Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MAX_FINDINGS = 200
MASK_CHAR = "*"
ADDRESS_KIND = "address"

# national_id: uppercase region letter + gender digit 1/2 + eight digits.
# mobile: 09xx-xxx-xxx, dashes optional.
# landline: 0 + 1-2 digit area code + 7-8 digits with optional dashes; the
#   (?!9) lookahead keeps dash-less mobile numbers (09xxxxxxxx) from being
#   double-counted as landlines — 09 prefixes are mobile-only in Taiwan.
PII_PATTERNS: dict[str, re.Pattern] = {
    "national_id": re.compile(r"[A-Z][12]\d{8}"),
    "mobile": re.compile(r"09\d{2}-?\d{3}-?\d{3}"),
    "landline": re.compile(r"0(?!9)\d{1,2}-?\d{3,4}-?\d{4}"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
# Heuristic street address: an administrative-division marker, a road/street
# token within 12 chars, then a house number ending in 號.
ADDRESS_PATTERN = re.compile(
    r"(縣|市|區|鄉|鎮)[^，。\s]{0,12}?(路|街|大道|巷)[^，。\s]{0,8}?號"
)
FULLWIDTH_PATTERN = re.compile(r"[０-９Ａ-Ｚａ-ｚ]")
# OCR look-alikes for 0 / 1 / O; extend as new confusions are observed.
OCR_SUSPECT_CHARS = ("〇", "ｌ", "Ｏ", "○")

_ASCII_DIGIT_RE = re.compile(r"[0-9]")
_FULLWIDTH_DIGIT_RE = re.compile(r"[０-９]")


@dataclass
class PiiFinding:
    """One masked PII hit: which document row, which kind, masked snippet."""

    row: int
    kind: str
    masked: str


@dataclass
class PiiReport:
    """Corpus-level PII scan result with capped, masked findings."""

    total_docs: int
    docs_with_pii: int
    counts: dict
    findings: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NormalizationReport:
    """Full-width / OCR-noise / mixed-width-digit statistics."""

    docs_with_fullwidth: int
    fullwidth_char_count: int
    docs_with_ocr_suspects: int
    ocr_suspect_counts: dict
    mixed_width_digit_docs: int

    def to_dict(self) -> dict:
        return asdict(self)


def mask_snippet(s: str) -> str:
    """Mask a matched snippet: keep first + last char, middle becomes '*'.

    Snippets of length <= 2 are fully masked.
    """
    if not isinstance(s, str):
        raise TypeError(f"snippet must be str, got {type(s).__name__}")
    if len(s) <= 2:
        return MASK_CHAR * len(s)
    return s[0] + MASK_CHAR * (len(s) - 2) + s[-1]


def scan_pii(texts: Sequence[str]) -> PiiReport:
    """Scan documents for Taiwan-style PII; findings are masked and capped.

    ``counts`` holds per-kind match totals (never capped); ``findings`` keeps
    the first ``MAX_FINDINGS`` masked hits for reporting.
    """
    items = _validated_texts(texts)
    patterns = dict(PII_PATTERNS)
    patterns[ADDRESS_KIND] = ADDRESS_PATTERN
    counts = {kind: 0 for kind in patterns}
    findings: list = []
    docs_with_pii = 0
    for row, text in enumerate(items):
        doc_hits = 0
        for kind, pattern in patterns.items():
            for match in pattern.finditer(text):
                counts[kind] += 1
                doc_hits += 1
                if len(findings) < MAX_FINDINGS:
                    findings.append(
                        PiiFinding(row=row, kind=kind, masked=mask_snippet(match.group(0)))
                    )
        if doc_hits:
            docs_with_pii += 1
    return PiiReport(
        total_docs=len(items),
        docs_with_pii=docs_with_pii,
        counts=counts,
        findings=findings,
    )


def check_normalization(texts: Sequence[str]) -> NormalizationReport:
    """Report full-width usage, OCR-suspect characters and width mixing.

    ``mixed_width_digit_docs`` counts documents containing BOTH an ASCII
    digit and a full-width digit — the classic un-normalized OCR/typing mix.
    """
    items = _validated_texts(texts)
    docs_with_fullwidth = 0
    fullwidth_char_count = 0
    docs_with_ocr = 0
    ocr_counts = {ch: 0 for ch in OCR_SUSPECT_CHARS}
    mixed_docs = 0
    for text in items:
        fullwidth_hits = FULLWIDTH_PATTERN.findall(text)
        if fullwidth_hits:
            docs_with_fullwidth += 1
            fullwidth_char_count += len(fullwidth_hits)
        doc_ocr = 0
        for ch in OCR_SUSPECT_CHARS:
            n = text.count(ch)
            ocr_counts[ch] += n
            doc_ocr += n
        if doc_ocr:
            docs_with_ocr += 1
        if _ASCII_DIGIT_RE.search(text) and _FULLWIDTH_DIGIT_RE.search(text):
            mixed_docs += 1
    return NormalizationReport(
        docs_with_fullwidth=docs_with_fullwidth,
        fullwidth_char_count=fullwidth_char_count,
        docs_with_ocr_suspects=docs_with_ocr,
        ocr_suspect_counts=ocr_counts,
        mixed_width_digit_docs=mixed_docs,
    )


def _validated_texts(texts: Sequence[str]) -> list:
    """Materialize ``texts`` as a list of str, or raise ``TypeError``."""
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of documents, not a single str")
    try:
        items = list(texts)
    except TypeError as exc:
        raise TypeError(
            f"texts must be a sequence of str, got {type(texts).__name__}"
        ) from exc
    for i, text in enumerate(items):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
    return items
