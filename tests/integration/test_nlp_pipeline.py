"""Integration tests: config -> dataset -> EDA / feature-select / benchmark.

Exercises the modules together on synthetic 公文 data written to a temp CSV,
plus the CLI subcommands end to end. CPU + baseline models only (no torch
training, no lightgbm/setfit) so the suite stays fast and dependency-light.
"""

import json

import pandas as pd
import pytest

from src.nlp.cli import main as cli_main
from src.nlp.config import config_from_dict
from src.nlp.datasets import load_text_dataset
from src.nlp.eda import run_eda
from src.nlp.harness import run_benchmark

pytestmark = pytest.mark.integration


def _write_config(tmp_path, csv_path, task_type="multiclass", models=None, metadata_cols=None):
    cfg = {
        "data": {"csv_path": str(csv_path), "text_col": "text", "label_col": "label",
                 "task_type": task_type, "test_size": 0.2, "val_size": 0.1},
        "segment": {"engine": "char"},
        "device": {"device": "cpu"},
        "output_dir": str(tmp_path / "out"),
        "seed": 0,
    }
    if models is not None:
        cfg["models"] = models
    if metadata_cols is not None:
        cfg["data"]["metadata_cols"] = metadata_cols
    return cfg


@pytest.fixture
def synth_csv(tmp_path):
    from src.nlp.synth import generate_synthetic_gov_docs

    df = generate_synthetic_gov_docs("imbalanced", n_docs=120, seed=0)
    path = tmp_path / "gongwen.csv"
    df.to_csv(path, index=False)
    return path


def test_prepare_eda_benchmark_chain(tmp_path, synth_csv):
    cfg_dict = _write_config(tmp_path, synth_csv,
                             models=[{"name": "tfidf_logreg", "class_weight": "balanced"}])
    config = config_from_dict(cfg_dict)

    dataset = load_text_dataset(config)
    assert len(dataset.texts_train) > 0
    assert dataset.label_space.n_classes >= 2

    texts = dataset.texts_train + dataset.texts_val + dataset.texts_test
    labels = dataset.label_space.decode(
        __import__("numpy").concatenate([dataset.y_train, dataset.y_val, dataset.y_test])
    )
    eda = run_eda(texts, labels, config)
    assert eda.n_docs == len(texts)
    assert eda.balance.imbalance_ratio > 1.0  # imbalanced synth

    result = run_benchmark(config, dataset=dataset)
    assert result.ranking == ["tfidf_logreg"]
    assert result.runs[0].metrics["f1_macro"] >= 0.0


def test_multilabel_chain(tmp_path):
    from src.nlp.synth import generate_synthetic_gov_docs

    df = generate_synthetic_gov_docs("multilabel", n_docs=100, seed=1)
    csv = tmp_path / "ml.csv"
    df.to_csv(csv, index=False)
    config = config_from_dict(_write_config(tmp_path, csv, task_type="multilabel",
                                            models=[{"name": "tfidf_linearsvm"}]))
    result = run_benchmark(config)
    assert result.task_type == "multilabel"
    assert "subset_accuracy" in result.runs[0].metrics


def test_cli_eda_and_benchmark(tmp_path, synth_csv, capsys):
    import yaml

    cfg = _write_config(tmp_path, synth_csv,
                        models=[{"name": "tfidf_logreg"}, {"name": "tfidf_nb"}])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    assert cli_main(["eda", "--config", str(cfg_path)]) == 0
    assert cli_main(["benchmark", "--config", str(cfg_path)]) == 0

    out = tmp_path / "out"
    assert (out / "eda" / "eda_report.md").exists()
    assert (out / "eda" / "eda_report.json").exists()
    bench_json = out / "benchmark" / "benchmark_report.json"
    assert bench_json.exists()
    payload = json.loads(bench_json.read_text(encoding="utf-8"))
    assert len(payload["models"]) == 2


def test_cli_feature_select_with_metadata(tmp_path, capsys):
    from src.nlp.synth import generate_synthetic_gov_docs
    import numpy as np
    import yaml

    df = generate_synthetic_gov_docs("balanced", n_docs=120, seed=2)
    rng = np.random.default_rng(0)
    # a metadata column perfectly correlated with the label + a noise column
    df["來源機關"] = df["label"].astype(str) + "局"
    df["雜訊"] = rng.integers(0, 3, size=len(df))
    csv = tmp_path / "meta.csv"
    df.to_csv(csv, index=False)

    cfg = _write_config(tmp_path, csv, metadata_cols=["來源機關", "雜訊"])
    cfg_path = tmp_path / "fs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    assert cli_main(["feature-select", "--config", str(cfg_path)]) == 0
    report = tmp_path / "out" / "feature_selection" / "feature_selection_report.md"
    assert report.exists()
    assert "總體建議" in report.read_text(encoding="utf-8")
