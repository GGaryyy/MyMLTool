"""Chinese segmenter abstraction for text EDA token statistics.

Provides a uniform :class:`Segmenter` interface over several tokenization
engines so exploratory statistics (token counts, length distributions,
over-512-token checks) can run in any environment:

- ``char``: pure-Python CJK-aware character segmentation, zero dependencies.
- ``spacy``: spaCy blank Chinese pipeline (character segmentation). Falls
  back to ``char`` with a :class:`UserWarning` when spacy is not installed.
- ``bert``: WordPiece tokens from a Hugging Face tokenizer, matching what
  the pretrained model families actually consume.
- ``ckip``: CKIP word segmentation (Academia Sinica, Taiwan); GPL-3.0
  licensed, opt-in only, unavailable unless explicitly installed.

Compliance: jieba / pkuseg (China-origin) are prohibited in this project and
are never imported or offered as engines. Heavy libraries (spacy /
transformers / ckip_transformers) are imported lazily inside the classes
that need them, never at module top.
"""

import importlib.util
import re
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from src.nlp.config import VALID_SEGMENT_ENGINES

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_BERT_TOKENIZER = "google-bert/bert-base-chinese"

# CJK Unified Ideographs (U+4E00-U+9FFF) plus Extension A (U+3400-U+4DBF).
_CJK_CHAR_CLASS = "一-鿿㐀-䶿"
# Alternation order matters: one CJK ideograph, else one contiguous ASCII
# letter/digit run ("GPU", "114", "RTX4070"), else any single non-space
# symbol. Whitespace never matches any alternative, so it is dropped.
_CHAR_TOKEN_RE = re.compile(rf"[{_CJK_CHAR_CLASS}]|[A-Za-z0-9]+|\S")


class Segmenter(ABC):
    """Uniform tokenizer interface consumed by the EDA statistics."""

    engine: str = "base"

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split a single document into a flat token list."""

    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """Tokenize many documents with the default one-by-one loop."""
        return [self.tokenize(text) for text in texts]


class CharSegmenter(Segmenter):
    """Deterministic CJK-aware character segmentation, zero dependencies."""

    engine = "char"

    def tokenize(self, text: str) -> list[str]:
        return _CHAR_TOKEN_RE.findall(text)


class SpacyCharSegmenter(Segmenter):
    """spaCy blank Chinese pipeline; tokens are single characters.

    COMPLIANCE: spaCy's Chinese pipeline defaults to character segmentation.
    Do NOT pass any segmenter config here — jieba / pkuseg are the
    China-origin opt-ins this project must avoid.
    """

    engine = "spacy"

    def __init__(self):
        # Lazy import keeps src.nlp importable in minimal environments.
        from spacy.lang.zh import Chinese

        self._nlp = Chinese()

    def tokenize(self, text: str) -> list[str]:
        return [token.text for token in self._nlp(text) if not token.is_space]


class BertWordpieceSegmenter(Segmenter):
    """WordPiece tokens from a Hugging Face BERT-style tokenizer."""

    engine = "bert"

    def __init__(self, pretrained_path: Optional[str] = None):
        try:
            from transformers import AutoTokenizer  # lazy heavy import
        except ImportError as exc:
            raise ImportError(
                "Segment engine 'bert' needs the transformers package; "
                "install the NLP extras from requirements-nlp.txt"
            ) from exc
        self.model_id = pretrained_path or DEFAULT_BERT_TOKENIZER
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def tokenize(self, text: str) -> list[str]:
        return list(self._tokenizer.tokenize(text))


class CkipSegmenter(Segmenter):
    """CKIP word segmentation driver — strictly opt-in (GPL-3.0)."""

    engine = "ckip"

    def __init__(self, pretrained_path: Optional[str] = None):
        if importlib.util.find_spec("ckip_transformers") is None:
            raise ImportError(
                "Segment engine 'ckip' is unavailable: ckip-transformers is "
                "GPL-3.0 licensed (Academia Sinica, Taiwan), opt-in only, "
                "requires legal review before use, and is not installed in "
                "the default images."
            )
        self._pretrained_path = pretrained_path
        self._driver = None

    def tokenize(self, text: str) -> list[str]:
        return self.tokenize_batch([text])[0]

    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        driver = self._load_driver()
        return [list(tokens) for tokens in driver(list(texts), show_progress=False)]

    def _load_driver(self):
        if self._driver is None:
            from ckip_transformers.nlp import CkipWordSegmenter  # lazy heavy import

            if self._pretrained_path:
                self._driver = CkipWordSegmenter(model_name=self._pretrained_path)
            else:
                self._driver = CkipWordSegmenter()
        return self._driver


def get_segmenter(engine: str = "spacy", pretrained_path: Optional[str] = None) -> Segmenter:
    """Build the requested :class:`Segmenter`.

    ``spacy`` degrades to :class:`CharSegmenter` with a ``UserWarning`` when
    spacy is missing, so the EDA still runs in minimal environments. ``bert``
    and ``ckip`` raise ``ImportError`` instead, because their token semantics
    cannot be reproduced without the real dependency.
    """
    if engine not in VALID_SEGMENT_ENGINES:
        raise ValueError(f"engine must be one of {VALID_SEGMENT_ENGINES}, got '{engine}'")
    if engine == "char":
        return CharSegmenter()
    if engine == "spacy":
        try:
            return SpacyCharSegmenter()
        except ImportError:
            warnings.warn(
                "spacy is not installed; falling back to pure-char "
                "segmentation with identical semantics for Chinese text "
                "(install spacy via requirements-nlp.txt to remove this warning)",
                UserWarning,
                stacklevel=2,
            )
            return CharSegmenter()
    if engine == "bert":
        return BertWordpieceSegmenter(pretrained_path=pretrained_path)
    return CkipSegmenter(pretrained_path=pretrained_path)
