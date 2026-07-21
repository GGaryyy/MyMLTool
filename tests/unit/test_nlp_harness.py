"""Unit tests for the benchmark harness (baseline models only, fast)."""

import json

import numpy as np
import pytest

from src.nlp import harness as harness_mod
from src.nlp.config import DataConfig, DeviceConfig, ModelConfig, RunConfig, TASK_MULTILABEL
from src.nlp.harness import BenchmarkResult, run_benchmark
from src.nlp.synth import generate_synthetic_gov_docs

pytestmark = pytest.mark.unit

TOP_KEYS = {"task_type", "seed", "device", "precision", "n_train", "n_val", "n_test",
            "label_space", "models", "ranking"}
MODEL_KEYS = {"name", "family", "train_seconds", "n_epochs", "ranking_score", "error",
              "metrics", "history", "notes"}


def _write_csv(tmp_path, mode="balanced", n=100, seed=0):
    df = generate_synthetic_gov_docs(mode, n_docs=n, seed=seed)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _config(csv_path, models, task_type="multiclass", output_dir="output/nlp"):
    return RunConfig(
        data=DataConfig(csv_path=csv_path, text_col="text", label_col="label",
                        task_type=task_type, test_size=0.2, val_size=0.1),
        device=DeviceConfig(device="cpu"),
        models=models,
        output_dir=output_dir,
    )


def test_benchmark_baselines_rank_and_schema(tmp_path):
    csv = _write_csv(tmp_path, n=100)
    config = _config(csv, [ModelConfig(name="tfidf_logreg"), ModelConfig(name="tfidf_nb")])
    result = run_benchmark(config)

    assert isinstance(result, BenchmarkResult)
    assert len(result.ranking) == 2
    for run in result.runs:
        assert run.error is None
        assert "f1_macro" in run.metrics

    payload = result.to_dict()
    assert set(payload) == TOP_KEYS
    for entry in payload["models"]:
        assert set(entry) == MODEL_KEYS
    json.dumps(payload)  # must be JSON-serializable


def test_benchmark_is_deterministic(tmp_path):
    csv = _write_csv(tmp_path, n=100)
    models = [ModelConfig(name="tfidf_logreg")]
    r1 = run_benchmark(_config(csv, models)).to_dict()
    r2 = run_benchmark(_config(csv, models)).to_dict()
    assert r1["ranking"] == r2["ranking"]
    assert r1["models"][0]["metrics"]["f1_macro"] == r2["models"][0]["metrics"]["f1_macro"]


def test_benchmark_error_isolation(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path, n=90)
    real_create = harness_mod.create_model

    def flaky_create(name):
        if name == "boom":
            raise RuntimeError("intentional failure")
        return real_create(name)

    monkeypatch.setattr(harness_mod, "create_model", flaky_create)
    config = _config(csv, [ModelConfig(name="tfidf_logreg"), ModelConfig(name="boom")])
    result = run_benchmark(config)

    by_name = {run.name: run for run in result.runs}
    assert by_name["tfidf_logreg"].error is None
    assert by_name["boom"].error is not None
    assert "RuntimeError" in by_name["boom"].error
    assert result.ranking == ["tfidf_logreg"]  # errored model excluded from ranking


def test_benchmark_empty_models_raises(tmp_path):
    csv = _write_csv(tmp_path)
    with pytest.raises(ValueError):
        run_benchmark(_config(csv, []))


def test_benchmark_multilabel_end_to_end(tmp_path):
    csv = _write_csv(tmp_path, mode="multilabel", n=90)
    config = _config(csv, [ModelConfig(name="tfidf_logreg")], task_type=TASK_MULTILABEL)
    result = run_benchmark(config)
    assert result.task_type == TASK_MULTILABEL
    metrics = result.runs[0].metrics
    assert "subset_accuracy" in metrics
