"""Benchmark harness: run every configured model family on one dataset.

Runs are isolated — one model blowing up records an errored
:class:`ModelRun` instead of aborting the benchmark; only
``KeyboardInterrupt`` / ``SystemExit`` propagate. Results serialize to the
stable JSON schema documented on :meth:`BenchmarkResult.to_dict`.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.nlp.config import DeviceConfig, ModelConfig, RunConfig, TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.datasets import TextDataset, load_text_dataset
from src.nlp.device import assert_wheel_compatible, detect_device, seed_everything
from src.nlp.metrics import compute_metrics, summarize_for_ranking
from src.nlp.models.base import FitReport
from src.nlp.models.registry import create_model

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
UNKNOWN_FAMILY = "unknown"  # recorded when create_model itself fails

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelRun:
    """Outcome of one model entry: fit report + metrics, or an error string."""

    name: str
    family: str
    fit_report: Optional[FitReport] = None
    metrics: Optional[dict] = None
    ranking_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to the fixed model-entry schema.

        ``{"name","family","train_seconds","n_epochs","ranking_score",
        "error","metrics","history","notes"}`` — errored runs always emit
        ``train_seconds=0.0, n_epochs=0, metrics=None, ranking_score=None,
        history=[], notes={}`` regardless of any partial state.
        """
        failed = self.error is not None or self.fit_report is None
        return {
            "name": self.name,
            "family": self.family,
            "train_seconds": 0.0 if failed else float(self.fit_report.train_seconds),
            "n_epochs": 0 if failed else int(self.fit_report.n_epochs),
            "ranking_score": None if self.error is not None else self.ranking_score,
            "error": self.error,
            "metrics": None if self.error is not None else self.metrics,
            "history": [] if failed else list(self.fit_report.history),
            "notes": {} if failed else dict(self.fit_report.notes),
        }


@dataclass
class BenchmarkResult:
    """Full benchmark outcome: run context, per-model runs and the ranking."""

    task_type: str
    seed: int
    device: str
    precision: str
    n_train: int
    n_val: int
    n_test: int
    label_space_summary: dict
    runs: list
    ranking: list

    def to_dict(self) -> dict:
        """Serialize to the stable top-level schema.

        ``{"task_type","seed","device","precision","n_train","n_val",
        "n_test","label_space":{"classes":[...],"is_multilabel":bool},
        "models":[...],"ranking":[...]}``
        """
        return {
            "task_type": self.task_type,
            "seed": self.seed,
            "device": self.device,
            "precision": self.precision,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "label_space": dict(self.label_space_summary),
            "models": [run.to_dict() for run in self.runs],
            "ranking": list(self.ranking),
        }


def run_benchmark(config: RunConfig, dataset: Optional[TextDataset] = None) -> BenchmarkResult:
    """Benchmark every ``config.models`` entry on the (loaded) dataset.

    ``dataset=None`` loads ``config.data.csv_path`` via
    :func:`src.nlp.datasets.load_text_dataset`. Each model is built, fitted
    on train (with validation feedback when a val split exists), evaluated
    on TEST and ranked by :func:`summarize_for_ranking` (macro-F1,
    descending, name-ascending tie-break). A CUDA-resolved device is
    hard-gated through :func:`assert_wheel_compatible` before any model
    runs. Raises ``ValueError`` when ``config.models`` is empty.
    """
    if not config.models:
        raise ValueError("config.models is empty; add at least one model entry to benchmark")

    seed_everything(config.seed)
    info = detect_device(config.device)
    if info.resolved_device == "cuda":
        assert_wheel_compatible(info)

    if dataset is None:
        dataset = load_text_dataset(config)
    label_space = dataset.label_space
    task_type = TASK_MULTILABEL if label_space.is_multilabel else TASK_MULTICLASS

    LOGGER.info(
        "Benchmark start: %d model(s), task=%s, device=%s/%s, n_train=%d n_val=%d n_test=%d",
        len(config.models), task_type, info.resolved_device, info.precision,
        len(dataset.texts_train), len(dataset.texts_val), len(dataset.texts_test),
    )

    runs = [
        _run_single_model(model_config, dataset, config.device)
        for model_config in config.models
    ]

    ranked = sorted(
        (run for run in runs if run.error is None),
        key=lambda run: (-run.ranking_score, run.name),
    )
    ranking = [run.name for run in ranked]
    LOGGER.info("Benchmark done: ranking=%s", ranking)

    return BenchmarkResult(
        task_type=task_type,
        seed=config.seed,
        device=info.resolved_device,
        precision=info.precision,
        n_train=len(dataset.texts_train),
        n_val=len(dataset.texts_val),
        n_test=len(dataset.texts_test),
        label_space_summary={
            "classes": list(label_space.classes),
            "is_multilabel": bool(label_space.is_multilabel),
        },
        runs=runs,
        ranking=ranking,
    )


def _run_single_model(model_config: ModelConfig, dataset: TextDataset,
                      device_config: DeviceConfig) -> ModelRun:
    """Run one model entry; every failure is captured, never propagated.

    ``KeyboardInterrupt`` / ``SystemExit`` are re-raised — a user abort must
    not be recorded as a model failure.
    """
    name = model_config.name
    family = UNKNOWN_FAMILY
    label_space = dataset.label_space
    LOGGER.info("Model '%s': starting", name)
    try:
        model = create_model(name)
        family = model.family
        model.build(label_space, model_config, device_config)
        fit_report = model.fit(
            dataset.texts_train, dataset.y_train,
            val_texts=dataset.texts_val, val_y=dataset.y_val,
        )
        y_pred = model.predict(dataset.texts_test)
        y_proba = model.predict_proba(dataset.texts_test)
        metrics = compute_metrics(dataset.y_test, y_pred, label_space, y_proba)
        ranking_score = summarize_for_ranking(metrics, label_space.is_multilabel)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        LOGGER.info("Model '%s': FAILED with %s: %s", name, type(exc).__name__, exc)
        return ModelRun(name=name, family=family, error=f"{type(exc).__name__}: {exc}")

    LOGGER.info(
        "Model '%s': done in %.2fs, ranking_score=%.4f",
        name, fit_report.train_seconds, ranking_score,
    )
    return ModelRun(
        name=name,
        family=family,
        fit_report=fit_report,
        metrics=metrics,
        ranking_score=ranking_score,
    )
