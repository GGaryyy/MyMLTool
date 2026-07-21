"""Unit tests for the deep-learning model families.

TextCNN / BiLSTM train on CPU with tiny synthetic corpora. BERT uses a
locally-built random-weight checkpoint (zero network). sent_embed injects a
fake encoder. SetFit needs a real backbone download, so it is network-gated.
"""

import os

import numpy as np
import pytest

from src.nlp.config import DeviceConfig, ModelConfig, TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.labels import build_label_space
from src.nlp.metrics import compute_metrics
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit


def _dataset(mode=TASK_MULTICLASS, n=72, seed=0):
    synth_mode = "multilabel" if mode == TASK_MULTILABEL else "balanced"
    df = generate_synthetic_gov_docs(synth_mode, n_docs=n, seed=seed)
    texts, raw = texts_and_labels(df)
    space, y = build_label_space(raw, mode)
    cut = int(n * 0.75)
    return texts[:cut], y[:cut], texts[cut:], y[cut:], space


@pytest.fixture
def tiny_bert_dir(tmp_path):
    """Build a tiny random-weight BERT + tokenizer locally (no download)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    df = generate_synthetic_gov_docs("balanced", n_docs=60, seed=1)
    texts, _ = texts_and_labels(df)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(texts), encoding="utf-8")

    wp = BertWordPieceTokenizer()
    wp.train([str(corpus)], vocab_size=1000, min_frequency=1,
             special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    wp.save_model(str(tok_dir))
    tok = BertTokenizerFast(vocab_file=str(tok_dir / "vocab.txt"))

    model_dir = tmp_path / "tiny_bert"
    model_dir.mkdir()
    tok.save_pretrained(str(model_dir))
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    BertForSequenceClassification(cfg).save_pretrained(str(model_dir))
    return str(model_dir)


class _FakeEncoder:
    """Deterministic per-text embedding; no model download."""

    def __init__(self, dim=48, signal=None):
        self.dim = dim
        self.signal = signal  # optional label-correlated signal for separability

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            r = np.random.default_rng(abs(hash(t)) % (10 ** 6))
            out[i] = r.standard_normal(self.dim)
        return out


# --------------------------------------------------------------------------- #
# TextCNN / BiLSTM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["textcnn", "bilstm_attn"])
def test_lightweight_dl_multiclass(name):
    from src.nlp.models.registry import create_model

    texts_tr, y_tr, texts_te, y_te, space = _dataset()
    model = create_model(name)
    model.build(space, ModelConfig(name=name, epochs=3, batch_size=16, max_length=256,
                                   learning_rate=1e-3, class_weight="balanced"),
                DeviceConfig(device="cpu", precision="fp32"))
    report = model.fit(texts_tr, y_tr, texts_te, y_te)
    assert report.device == "cpu"
    assert report.n_epochs == 3
    assert len(report.history) == 3

    proba = model.predict_proba(texts_te)
    assert proba.shape == (len(texts_te), space.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = model.predict(texts_te)
    assert pred.shape == (len(texts_te),)
    # synthetic topics are separable; learning should beat random even briefly
    assert compute_metrics(y_te, pred, space)["f1_macro"] > 0.2


@pytest.mark.parametrize("name", ["textcnn", "bilstm_attn"])
def test_lightweight_dl_multilabel(name):
    from src.nlp.models.registry import create_model

    texts_tr, y_tr, texts_te, y_te, space = _dataset(mode=TASK_MULTILABEL, n=80)
    model = create_model(name)
    model.build(space, ModelConfig(name=name, epochs=2, batch_size=16, max_length=256,
                                   learning_rate=1e-3),
                DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    proba = model.predict_proba(texts_te)
    assert proba.shape == (len(texts_te), space.n_classes)
    assert ((proba >= 0) & (proba <= 1)).all()
    pred = model.predict(texts_te)
    assert set(np.unique(pred)).issubset({0, 1})
    assert pred.shape == (len(texts_te), space.n_classes)


def test_lightweight_dl_save_load_roundtrip(tmp_path):
    from src.nlp.models.registry import create_model

    texts_tr, y_tr, texts_te, y_te, space = _dataset(n=60)
    model = create_model("textcnn")
    mc = ModelConfig(name="textcnn", epochs=2, batch_size=16, max_length=200, learning_rate=1e-3)
    model.build(space, mc, DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    before = model.predict_proba(texts_te)

    path = str(tmp_path / "textcnn.pt")
    model.save(path)
    restored = create_model("textcnn")
    restored.build(space, mc, DeviceConfig(device="cpu"))
    restored.load(path)
    after = restored.predict_proba(texts_te)
    np.testing.assert_allclose(before, after, atol=1e-4)


def test_predict_before_fit_raises():
    from src.nlp.models.registry import create_model

    _, _, _, _, space = _dataset(n=40)
    model = create_model("textcnn")
    model.build(space, ModelConfig(name="textcnn"), DeviceConfig(device="cpu"))
    with pytest.raises(RuntimeError):
        model.predict_proba(["主旨：測試"])


# --------------------------------------------------------------------------- #
# sent_embed (fake encoder)
# --------------------------------------------------------------------------- #
def test_sent_embed_multiclass_with_fake_encoder():
    from src.nlp.models.sent_embed import SentEmbedClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(n=60)
    model = SentEmbedClassifier()
    model.build(space, ModelConfig(name="sent_embed"), DeviceConfig(device="cpu"))
    model.set_encoder(_FakeEncoder())
    report = model.fit(texts_tr, y_tr, texts_te, y_te)
    assert report.family == "pretrained"
    proba = model.predict_proba(texts_te)
    assert proba.shape == (len(texts_te), space.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_sent_embed_multilabel_with_fake_encoder():
    from src.nlp.models.sent_embed import SentEmbedClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(mode=TASK_MULTILABEL, n=80)
    model = SentEmbedClassifier()
    model.build(space, ModelConfig(name="sent_embed"), DeviceConfig(device="cpu"))
    model.set_encoder(_FakeEncoder())
    model.fit(texts_tr, y_tr)
    pred = model.predict(texts_te)
    assert pred.shape == (len(texts_te), space.n_classes)
    assert set(np.unique(pred)).issubset({0, 1})


def test_sent_embed_missing_dependency(monkeypatch):
    import sys

    from src.nlp.models.sent_embed import SentEmbedClassifier

    _, _, _, _, space = _dataset(n=40)
    model = SentEmbedClassifier()
    model.build(space, ModelConfig(name="sent_embed"), DeviceConfig(device="cpu"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        model._ensure_encoder()


# --------------------------------------------------------------------------- #
# BERT (tiny local checkpoint, no network)
# --------------------------------------------------------------------------- #
def test_bert_finetune_multiclass_local(tiny_bert_dir):
    from src.nlp.models.bert_finetune import BertFinetuneClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(n=48)
    model = BertFinetuneClassifier()
    model.build(space, ModelConfig(name="bert", pretrained_path=tiny_bert_dir, epochs=1,
                                   batch_size=8, max_length=128, learning_rate=5e-4),
                DeviceConfig(device="cpu", precision="fp32"))
    report = model.fit(texts_tr, y_tr, texts_te, y_te)
    assert report.device == "cpu"
    assert report.notes.get("max_length")  # 128 < 512 -> truncation note
    proba = model.predict_proba(texts_te)
    assert proba.shape == (len(texts_te), space.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_bert_save_load_roundtrip(tiny_bert_dir, tmp_path):
    from src.nlp.models.bert_finetune import BertFinetuneClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(n=40)
    mc = ModelConfig(name="bert", pretrained_path=tiny_bert_dir, epochs=1, batch_size=8,
                     max_length=96, learning_rate=5e-4)
    model = BertFinetuneClassifier()
    model.build(space, mc, DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    before = model.predict_proba(texts_te)

    save_dir = str(tmp_path / "saved_bert")
    model.save(save_dir)
    restored = BertFinetuneClassifier()
    restored.build(space, ModelConfig(name="bert", pretrained_path=save_dir, max_length=96),
                   DeviceConfig(device="cpu"))
    restored.load(save_dir)
    after = restored.predict_proba(texts_te)
    np.testing.assert_allclose(before, after, atol=1e-4)


def test_bert_multilabel_local(tiny_bert_dir):
    from src.nlp.models.bert_finetune import BertFinetuneClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(mode=TASK_MULTILABEL, n=48)
    model = BertFinetuneClassifier()
    model.build(space, ModelConfig(name="bert", pretrained_path=tiny_bert_dir, epochs=1,
                                   batch_size=8, max_length=96, learning_rate=5e-4),
                DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    pred = model.predict(texts_te)
    assert pred.shape == (len(texts_te), space.n_classes)
    assert set(np.unique(pred)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# SetFit (network-gated: needs a real backbone download)
# --------------------------------------------------------------------------- #
def test_setfit_importable():
    """The module and class import without setfit installed (lazy import)."""
    from src.nlp.models.setfit_clf import SetFitClassifier

    model = SetFitClassifier()
    assert model.name == "setfit"
    assert model.family == "pretrained"


@pytest.mark.network
@pytest.mark.slow
def test_setfit_train_predict_downloads():
    pytest.importorskip("setfit")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("offline mode: SetFit backbone download unavailable")
    from src.nlp.models.registry import create_model

    texts_tr, y_tr, texts_te, y_te, space = _dataset(n=48)
    model = create_model("setfit")
    model.build(space, ModelConfig(name="setfit", epochs=1, batch_size=8,
                                   params={"num_iterations": 5}),
                DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    proba = model.predict_proba(texts_te)
    assert proba.shape[0] == len(texts_te)
