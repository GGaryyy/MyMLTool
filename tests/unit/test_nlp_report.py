"""Unit tests for src.nlp.report — EDA / benchmark markdown, JSON, plots."""

import json
import re
from pathlib import Path

import pytest

from src.nlp.config import config_from_dict
from src.nlp.eda import run_eda
from src.nlp.report import write_benchmark_report, write_eda_report
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

NATIONAL_ID_RE = re.compile(r"[A-Z][12]\d{8}")


@pytest.fixture(scope="module")
def eda_report():
    df = generate_synthetic_gov_docs(mode="imbalanced", n_docs=120, seed=0)
    texts, labels = texts_and_labels(df)
    config = config_from_dict({
        "data": {"task_type": "multiclass"},
        "segment": {"engine": "char"},
    })
    return run_eda(texts, labels, config)


def _fake_model(name, family, score, error=None):
    metrics = {} if error else {"f1_macro": score, "accuracy": score + 0.02}
    return {
        "name": name,
        "family": family,
        "train_seconds": 12.5,
        "n_epochs": 3,
        "ranking_score": score,
        "error": error,
        "metrics": metrics,
        "history": [{"epoch": 1, "loss": 0.5}],
        "notes": {"note": "synthetic"},
    }


def _fake_benchmark_result():
    return {
        "task_type": "multiclass",
        "seed": 0,
        "device": "cpu",
        "precision": "fp32",
        "n_train": 210,
        "n_val": 30,
        "n_test": 60,
        "label_space": {"classes": ["人事", "預算", "資訊安全"], "is_multilabel": False},
        "models": [
            _fake_model("tfidf_logreg", "sklearn", 0.81),
            _fake_model("bert_base", "transformer", 0.88),
            _fake_model("broken_model", "transformer", None, error="CUDA out of memory"),
        ],
        "ranking": ["bert_base", "tfidf_logreg"],
    }


# --------------------------------------------------------------------------- #
# write_eda_report
# --------------------------------------------------------------------------- #
def test_eda_report_files_created_and_json_round_trips(tmp_path, eda_report):
    result = write_eda_report(eda_report, str(tmp_path))
    md_path = Path(result["markdown"])
    json_path = Path(result["json"])
    assert md_path == tmp_path / "eda" / "eda_report.md"
    assert json_path == tmp_path / "eda" / "eda_report.json"
    assert md_path.is_file() and json_path.is_file()

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["n_docs"] == 120
    assert parsed["task_type"] == "multiclass"


def test_eda_markdown_sections_present_and_pii_masked(tmp_path, eda_report):
    # Precondition: the synthetic corpus really contains fake PII.
    assert eda_report.pii.docs_with_pii > 0
    result = write_eda_report(eda_report, str(tmp_path))
    md = Path(result["markdown"]).read_text(encoding="utf-8")
    for section in ("概要", "長度分布", "類別分布", "詞彙", "品質問題",
                    "PII 與正規化", "切分洩漏", "建議"):
        assert section in md
    assert NATIONAL_ID_RE.search(md) is None

    raw_json = Path(result["json"]).read_text(encoding="utf-8")
    assert NATIONAL_ID_RE.search(raw_json) is None


def test_eda_plots_written_nonempty(tmp_path, eda_report):
    result = write_eda_report(eda_report, str(tmp_path))
    assert result["plots"]
    for plot in result["plots"]:
        path = Path(plot)
        assert path.parent == tmp_path / "eda" / "plots"
        assert path.is_file()
        assert path.stat().st_size > 0


def test_eda_report_nested_out_dir_autocreated(tmp_path, eda_report):
    out_dir = tmp_path / "deep" / "nested" / "run01"
    result = write_eda_report(eda_report, str(out_dir))
    assert Path(result["markdown"]).is_file()
    assert Path(result["json"]).is_file()


def test_eda_report_rejects_wrong_type(tmp_path):
    with pytest.raises(TypeError, match="TextEdaReport"):
        write_eda_report({"n_docs": 1}, str(tmp_path))


# --------------------------------------------------------------------------- #
# write_benchmark_report
# --------------------------------------------------------------------------- #
def test_benchmark_report_files_created(tmp_path):
    result = write_benchmark_report(_fake_benchmark_result(), str(tmp_path))
    md_path = Path(result["markdown"])
    json_path = Path(result["json"])
    assert md_path == tmp_path / "benchmark" / "benchmark_report.md"
    assert json_path == tmp_path / "benchmark" / "benchmark_report.json"
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(parsed["models"]) == 3


def test_benchmark_markdown_lists_error_and_ranking(tmp_path):
    result = write_benchmark_report(_fake_benchmark_result(), str(tmp_path))
    md = Path(result["markdown"]).read_text(encoding="utf-8")
    assert "執行摘要" in md
    assert "每模型詳細" in md
    assert "排名與建議" in md
    assert "CUDA out of memory" in md
    assert "broken_model" in md
    assert "bert_base" in md
    # best non-errored model is the recommendation
    assert "建議優先採用 **bert_base**" in md


def test_benchmark_ranking_plot_written_skipping_errored(tmp_path):
    result = write_benchmark_report(_fake_benchmark_result(), str(tmp_path))
    assert result["plots"]
    plot = Path(result["plots"][0])
    assert plot.parent == tmp_path / "benchmark" / "plots"
    assert plot.is_file()
    assert plot.stat().st_size > 0


def test_benchmark_all_models_errored_still_writes_md_without_plot(tmp_path):
    result_dict = _fake_benchmark_result()
    for model in result_dict["models"]:
        model["error"] = "boom"
        model["ranking_score"] = None
    result = write_benchmark_report(result_dict, str(tmp_path))
    md = Path(result["markdown"]).read_text(encoding="utf-8")
    assert "所有模型皆執行失敗" in md
    assert result["plots"] == []


def test_benchmark_empty_models_raises(tmp_path):
    payload = _fake_benchmark_result()
    payload["models"] = []
    with pytest.raises(ValueError, match="no models"):
        write_benchmark_report(payload, str(tmp_path))
    with pytest.raises(ValueError, match="no models"):
        write_benchmark_report({}, str(tmp_path))


def test_benchmark_out_dir_nested_autocreated(tmp_path):
    out_dir = tmp_path / "x" / "y"
    result = write_benchmark_report(_fake_benchmark_result(), str(out_dir))
    assert Path(result["markdown"]).is_file()
    assert Path(result["json"]).is_file()
