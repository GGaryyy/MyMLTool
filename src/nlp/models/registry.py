"""Static model registry: config names -> lazily-imported classifier classes.

Kept as a plain dict (module path, class name, constructor kwargs) so that
listing models never imports torch / transformers; the import happens only
when :func:`create_model` instantiates the chosen entry.
"""

import importlib

from src.nlp.models.base import TextClassifier

MODEL_REGISTRY = {
    "tfidf_logreg":    ("src.nlp.models.tfidf_linear", "TfidfLinearClassifier", {"variant": "logreg"}),
    "tfidf_linearsvm": ("src.nlp.models.tfidf_linear", "TfidfLinearClassifier", {"variant": "linearsvm"}),
    "tfidf_nb":        ("src.nlp.models.tfidf_linear", "TfidfLinearClassifier", {"variant": "nb"}),
    "tfidf_tree":      ("src.nlp.models.tfidf_linear", "TfidfLinearClassifier", {"variant": "tree"}),
    "tfidf_lightgbm":  ("src.nlp.models.tfidf_gbm", "TfidfGbmClassifier", {}),
    "textcnn":         ("src.nlp.models.textcnn", "TextCnnClassifier", {}),
    "bilstm_attn":     ("src.nlp.models.bilstm_attn", "BiLstmAttnClassifier", {}),
    "bert":            ("src.nlp.models.bert_finetune", "BertFinetuneClassifier", {}),
    "sent_embed":      ("src.nlp.models.sent_embed", "SentEmbedClassifier", {}),
    "setfit":          ("src.nlp.models.setfit_clf", "SetFitClassifier", {}),
}


def list_models() -> list:
    """Sorted names accepted by ``ModelConfig.name``."""
    return sorted(MODEL_REGISTRY)


def create_model(name: str) -> TextClassifier:
    """Instantiate the classifier registered under ``name``.

    Raises ``ValueError`` for unknown names and ``ImportError`` (with the
    original cause) when the family's optional dependency is missing.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")

    module_path, class_name, kwargs = MODEL_REGISTRY[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Model '{name}' needs an optional dependency that is not installed "
            f"(see requirements-nlp.txt): {exc}"
        ) from exc

    cls = getattr(module, class_name)
    model = cls(**kwargs)
    if not isinstance(model, TextClassifier):
        raise TypeError(f"Registry entry '{name}' did not produce a TextClassifier")
    return model
