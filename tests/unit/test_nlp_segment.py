"""Unit tests for src.nlp.segment — engines, fallbacks, compliance guards.

spacy / transformers tests are importorskip-gated so absence == skip; the
missing-dependency paths are exercised via sys.modules monkeypatching and
pass regardless of what is installed.
"""

import importlib.util
import sys

import pytest

from src.nlp.config import VALID_SEGMENT_ENGINES
from src.nlp.segment import (
    DEFAULT_BERT_TOKENIZER,
    BertWordpieceSegmenter,
    CharSegmenter,
    SpacyCharSegmenter,
    get_segmenter,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# char engine
# --------------------------------------------------------------------------- #
def test_char_splits_cjk_per_character():
    assert CharSegmenter().tokenize("資訊安全法") == ["資", "訊", "安", "全", "法"]


def test_char_groups_contiguous_ascii_runs():
    assert CharSegmenter().tokenize("RTX4070顯卡") == ["RTX4070", "顯", "卡"]


def test_char_ascii_runs_split_on_symbols():
    assert CharSegmenter().tokenize("GPU-114") == ["GPU", "-", "114"]


def test_char_drops_whitespace():
    assert CharSegmenter().tokenize("資 訊\t安\n全  GPU") == ["資", "訊", "安", "全", "GPU"]


def test_char_punctuation_is_single_tokens():
    assert CharSegmenter().tokenize("主旨：測試。") == ["主", "旨", "：", "測", "試", "。"]


def test_char_fullwidth_digits_are_single_tokens():
    # Only ASCII runs group; full-width digits stay one token each.
    assert CharSegmenter().tokenize("１２３") == ["１", "２", "３"]


def test_char_empty_string_gives_empty_list():
    assert CharSegmenter().tokenize("") == []


def test_tokenize_batch_matches_single_calls():
    seg = CharSegmenter()
    texts = ["資訊安全", "GPU 114", ""]
    assert seg.tokenize_batch(texts) == [seg.tokenize(t) for t in texts]


def test_get_segmenter_char_returns_char_segmenter():
    seg = get_segmenter("char")
    assert isinstance(seg, CharSegmenter)
    assert seg.engine == "char"


# --------------------------------------------------------------------------- #
# engine validation / compliance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", ["jieba", "pkuseg", "bogus", ""])
def test_unknown_engine_raises_value_error(engine):
    with pytest.raises(ValueError, match="engine must be one of"):
        get_segmenter(engine)


def test_valid_engines_exclude_china_origin_tokenizers():
    assert "jieba" not in VALID_SEGMENT_ENGINES
    assert "pkuseg" not in VALID_SEGMENT_ENGINES


# --------------------------------------------------------------------------- #
# spacy engine
# --------------------------------------------------------------------------- #
def test_spacy_engine_matches_char_engine_and_has_no_jieba():
    pytest.importorskip("spacy")
    seg = get_segmenter("spacy")
    assert isinstance(seg, SpacyCharSegmenter)

    sample = "主旨：檢送資訊安全管理法規定，請查照。"
    assert seg.tokenize(sample) == CharSegmenter().tokenize(sample)

    # Compliance: blank Chinese pipeline must stay on character segmentation;
    # jieba / pkuseg must not be configured anywhere in the tokenizer.
    tokenizer = seg._nlp.tokenizer
    segmenter_name = str(getattr(tokenizer, "segmenter", "char")).lower()
    assert "char" in segmenter_name
    assert "jieba" not in segmenter_name
    assert "pkuseg" not in segmenter_name


def test_spacy_missing_falls_back_to_char_with_warning(monkeypatch):
    # Simulate an environment without spacy: a None entry makes any
    # "import spacy" raise ImportError, even if spacy is installed.
    for name in ("spacy", "spacy.lang", "spacy.lang.zh"):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.warns(UserWarning, match="falling back"):
        seg = get_segmenter("spacy")
    assert isinstance(seg, CharSegmenter)
    assert seg.tokenize("資訊安全") == ["資", "訊", "安", "全"]


# --------------------------------------------------------------------------- #
# bert engine
# --------------------------------------------------------------------------- #
def test_default_bert_tokenizer_constant():
    assert DEFAULT_BERT_TOKENIZER == "google-bert/bert-base-chinese"


def test_bert_engine_with_tiny_local_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers_impl = pytest.importorskip("tokenizers.implementations")

    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                   "資", "訊", "安", "全", "法"]) + "\n",
        encoding="utf-8",
    )
    # transformers 5.x no longer builds a fast tokenizer from a bare
    # vocab_file, so build tokenizer.json via the tokenizers library and
    # save a directory AutoTokenizer can load offline (zero network).
    wordpiece = tokenizers_impl.BertWordPieceTokenizer(vocab=str(vocab_file))
    tokenizer_json = tmp_path / "tokenizer.json"
    wordpiece.save(str(tokenizer_json))

    tokenizer = transformers.BertTokenizerFast(tokenizer_file=str(tokenizer_json))
    save_dir = tmp_path / "tiny-bert-tokenizer"
    tokenizer.save_pretrained(str(save_dir))

    seg = get_segmenter("bert", pretrained_path=str(save_dir))
    assert isinstance(seg, BertWordpieceSegmenter)
    assert seg.model_id == str(save_dir)
    assert seg.tokenize("資訊安全法") == ["資", "訊", "安", "全", "法"]
    assert seg.tokenize_batch(["資訊", "安全"]) == [["資", "訊"], ["安", "全"]]


def test_bert_engine_missing_transformers_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="requirements-nlp"):
        get_segmenter("bert")


# --------------------------------------------------------------------------- #
# ckip engine
# --------------------------------------------------------------------------- #
def test_ckip_absent_raises_import_error_mentioning_gpl():
    if importlib.util.find_spec("ckip_transformers") is not None:
        pytest.skip("ckip_transformers is installed; absence path not testable")
    with pytest.raises(ImportError, match="GPL-3.0") as excinfo:
        get_segmenter("ckip")
    message = str(excinfo.value)
    assert "Academia Sinica" in message
    assert "opt-in" in message
    assert "legal review" in message
    assert "not installed" in message
