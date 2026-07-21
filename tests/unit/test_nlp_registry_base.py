"""Unit tests for the model registry and the TextClassifier base logic."""

import numpy as np
import pytest

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import LabelSpace
from src.nlp.models.base import FitReport, TextClassifier
from src.nlp.models.registry import MODEL_REGISTRY, create_model, list_models

pytestmark = pytest.mark.unit

EXPECTED_MODELS = sorted([
    "tfidf_logreg", "tfidf_linearsvm", "tfidf_nb", "tfidf_tree", "tfidf_lightgbm",
    "textcnn", "bilstm_attn", "bert", "sent_embed", "setfit",
])


def test_list_models_matches_expected():
    assert list_models() == EXPECTED_MODELS
    assert set(MODEL_REGISTRY) == set(EXPECTED_MODELS)


def test_create_model_returns_textclassifier():
    model = create_model("tfidf_logreg")
    assert isinstance(model, TextClassifier)
    assert model.name == "tfidf_logreg"


def test_create_unknown_model_lists_available():
    with pytest.raises(ValueError) as excinfo:
        create_model("no_such_model")
    message = str(excinfo.value)
    assert "no_such_model" in message
    assert "tfidf_logreg" in message  # lists available names


class _DummyClassifier(TextClassifier):
    """Returns fixed probabilities so base decision rules can be tested."""

    name = "dummy"
    family = "baseline"

    def __init__(self, proba):
        super().__init__()
        self._proba = np.asarray(proba, dtype=float)

    def fit(self, texts, y, val_texts=None, val_y=None):
        return FitReport(model_name=self.name, family=self.family)

    def predict_proba(self, texts):
        return self._proba


def test_base_predict_multiclass_argmax():
    space = LabelSpace(classes=["a", "b", "c"], is_multilabel=False)
    proba = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]
    model = _DummyClassifier(proba)
    model.build(space, ModelConfig(name="dummy"))
    np.testing.assert_array_equal(model.predict(["x", "y"]), np.array([1, 0]))


@pytest.mark.parametrize("threshold,expected", [
    (0.3, [[1, 1, 0]]),
    (0.7, [[0, 1, 0]]),
])
def test_base_predict_multilabel_threshold(threshold, expected):
    space = LabelSpace(classes=["a", "b", "c"], is_multilabel=True)
    model = _DummyClassifier([[0.4, 0.8, 0.1]])
    model.build(space, ModelConfig(name="dummy", threshold=threshold))
    np.testing.assert_array_equal(model.predict(["x"]), np.array(expected))


def test_build_type_errors():
    space = LabelSpace(classes=["a", "b"], is_multilabel=False)
    model = _DummyClassifier([[0.5, 0.5]])
    with pytest.raises(TypeError):
        model.build("not a label space", ModelConfig(name="dummy"))
    with pytest.raises(TypeError):
        model.build(space, "not a model config")


def test_predict_before_build_raises():
    model = _DummyClassifier([[0.5, 0.5]])
    with pytest.raises(RuntimeError):
        model.predict(["x"])


def test_default_save_load_not_implemented():
    space = LabelSpace(classes=["a", "b"], is_multilabel=False)
    model = _DummyClassifier([[0.5, 0.5]])
    model.build(space, ModelConfig(name="dummy"))
    with pytest.raises(NotImplementedError):
        model.save("x")
    with pytest.raises(NotImplementedError):
        model.load("x")
