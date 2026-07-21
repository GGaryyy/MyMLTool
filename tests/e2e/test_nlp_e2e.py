"""End-to-end tests: full CLI workflow a user/operator runs.

diagnose -> eda -> benchmark against a synthetic Chinese text CSV, verifying the
report artifacts land under output_dir. Includes one real lightweight-DL
model (TextCNN, 1 epoch, CPU) so the deep-learning path is exercised end to
end, kept tiny for speed.
"""

import json

import pytest
import yaml

from src.nlp.cli import main as cli_main

pytestmark = pytest.mark.e2e


@pytest.fixture
def workspace(tmp_path):
    from src.nlp.synth import generate_synthetic_gov_docs

    df = generate_synthetic_gov_docs("balanced", n_docs=90, seed=0)
    csv = tmp_path / "data" / "train.csv"
    csv.parent.mkdir()
    df.to_csv(csv, index=False)
    out = tmp_path / "output"
    return tmp_path, csv, out


def _config(csv, out, models):
    return {
        "data": {"csv_path": str(csv), "text_col": "text", "label_col": "label",
                 "task_type": "multiclass", "test_size": 0.2, "val_size": 0.1},
        "segment": {"engine": "char"},
        "device": {"device": "cpu", "precision": "fp32"},
        "output_dir": str(out),
        "seed": 0,
        "models": models,
    }


def test_diagnose_runs(capsys):
    assert cli_main(["diagnose"]) == 0
    out = capsys.readouterr().out
    assert "wheel/device compatibility" in out


def test_full_workflow_baseline_and_dl(workspace):
    tmp_path, csv, out = workspace
    cfg = _config(csv, out, [
        {"name": "tfidf_logreg"},
        {"name": "textcnn", "epochs": 1, "batch_size": 16, "max_length": 200,
         "learning_rate": 0.001},
    ])
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    assert cli_main(["eda", "--config", str(cfg_path)]) == 0
    assert cli_main(["benchmark", "--config", str(cfg_path)]) == 0

    assert (out / "eda" / "eda_report.md").exists()
    bench = json.loads((out / "benchmark" / "benchmark_report.json").read_text(encoding="utf-8"))
    names = {m["name"] for m in bench["models"]}
    assert names == {"tfidf_logreg", "textcnn"}
    # both models trained without error
    assert all(m["error"] is None for m in bench["models"])
    assert len(bench["ranking"]) == 2
