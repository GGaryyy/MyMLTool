# MyMLTool

A machine-learning toolkit with two parts:

1. **Tabular utilities** (`src/data_prep.py`, `src/feature_shift.py`) — reusable data-prep and feature-shift/drift detection for classic ML.
2. **`src/nlp/` — Traditional-Chinese (繁中) NLP text-classification pipeline** — a data-agnostic pipeline that analyses a labelled text dataset (EDA) and benchmarks candidate algorithms so you can pick the best model and preprocessing from evidence. PyTorch-based, GPU-aware, deliverable as a Docker image.

This README focuses on the NLP pipeline. For the tabular utilities see `docs/flow/system_flow.md`.

---

## What the NLP pipeline does

Given a CSV with a **text column** and a **label column**, it produces:

- an **EDA report** — text length, class distribution & imbalance, vocabulary, duplicates, train/val/test leakage, label-quality checks, PII scan, and normalization checks;
- a **feature-selection recommendation** — which terms/n-grams matter (χ²/MI/ANOVA/L1/tree), a recommended vocabulary size, and (if you have structured metadata columns) their relevance and correlation;
- a **benchmark report** — several candidate models trained on the same split, ranked by macro-F1, so you can choose.

Supports **single-label multiclass**, **imbalanced multiclass**, and **multi-label** tasks.

### Candidate models

| Family | Models | Device |
|--------|--------|--------|
| Baseline | TF-IDF + {LogReg, LinearSVM, NaiveBayes, Tree}, TF-IDF/SVD + LightGBM | CPU |
| Lightweight DL | TextCNN, BiLSTM + Attention (character embeddings) | CPU / GPU |
| Pretrained | Chinese-BERT fine-tune (`google-bert/bert-base-chinese`), frozen sentence-embedding + linear head, SetFit | CPU / GPU |

Segmentation uses spaCy **character** tokenization (no third-party word segmenter dependency).

---

## Requirements

- **Python 3.12**.
- Core deps: `pip install -r requirements.txt` (numpy, pandas, scikit-learn, scipy, pyyaml, matplotlib). The pure-Python core (config, EDA, metrics, TF-IDF baselines, analysis) runs on this alone.
- Deep-learning deps (PyTorch/transformers/spaCy/…): `requirements-nlp.txt`. **Install the matching torch wheel first** — see [docs/nlp/INSTALL.md](docs/nlp/INSTALL.md):

  | Platform / GPU | torch install |
  |----------------|---------------|
  | x86_64 + RTX 4070 (Ada) / 5070 Ti (Blackwell) | `pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128` |
  | GB10 / Grace-Blackwell (ARM64) | NVIDIA aarch64 wheel / NGC container |
  | CPU-only (CI) | `pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu` |

  then `pip install -r requirements-nlp.txt`.

---

## Quick start

### 1. Prepare your data

A **CSV with a header**, at minimum a text column and a label column:

| Task | Label column looks like | `task_type` |
|------|-------------------------|-------------|
| Single-label | `classA` | `multiclass` |
| Multi-label | `classA\|classB` (pipe-separated) | `multilabel` |

Save it as UTF-8, e.g. `data/train.csv`.

### 2. Write a config

Copy [`configs/classification.example.yaml`](configs/classification.example.yaml) and edit the column names. Config structure (top-level keys are **not** indented):

```yaml
data:                      # where the data is and how to read it
  csv_path: data/train.csv
  text_col: text           # column name (or 0-based integer position)
  label_col: label         # column name (or -1 = last column, the default)
  task_type: multiclass    # multiclass | multilabel
  test_size: 0.2
  val_size: 0.1
  metadata_cols: []        # optional structured columns for feature-select

segment:
  engine: spacy            # character segmentation

device:
  device: auto             # auto | cuda | cpu
  precision: auto          # auto (Blackwell/Ada -> bf16) | bf16 | fp16 | fp32

output_dir: output/nlp     # <-- where reports are written (top-level field)
seed: 0

models:                    # which models to benchmark
  - {name: tfidf_logreg, class_weight: balanced}
  - {name: textcnn, epochs: 5}
  - {name: bert, epochs: 3, max_length: 512}
```

> **`output_dir`** is the top-level field that controls where results go. YAML indentation matters: top-level keys (`data`, `segment`, `device`, `output_dir`, `seed`, `models`) are flush-left; use spaces, never tabs.

### 3. Run

```bash
cd <repo>
# ① Check the GPU is detected (Ada/Blackwell + bf16 + compatibility OK)
python -m src.nlp.cli diagnose

# ② Analyse the data
python -m src.nlp.cli eda --config configs/classification.example.yaml

# ③ Benchmark the models (BERT downloads the backbone on first run)
python -m src.nlp.cli benchmark --config configs/classification.example.yaml

# (optional) feature-selection recommendation, incl. metadata correlation
python -m src.nlp.cli feature-select --config configs/classification.example.yaml
```

(In this repo the environment lives in a WSL venv; prefix commands with
`wsl -e bash -lc "cd /mnt/.../MyMLTool && .venv/bin/python -m src.nlp.cli ..."`.)

### 4. Read the results

Everything lands under `output_dir` (default `output/nlp/`):

```
output/nlp/
├── eda/eda_report.md                 # length, imbalance, duplicates, leakage, PII, suggestions
├── feature_selection/…               # term & metadata feature recommendations
└── benchmark/benchmark_report.md     # per-model F1 table + ranking + recommendation
```

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `diagnose` | GPU / wheel-compatibility report (run first on a new machine) |
| `eda --config <f>` | Text EDA report |
| `feature-select --config <f>` | Term + metadata feature-selection recommendations |
| `benchmark --config <f>` | Train & rank the configured models |
| `download-models --dest <dir>` | Pre-download compliant models for offline use |

Example configs: `configs/{eda,benchmark,feature_select,classification}.example.yaml`.

---

## Docker (delivery)

A single CUDA 12.8 image covers RTX 4070 and 5070 Ti; GB10/ARM64 is a build-arg extension. See [docs/nlp/DEPLOYMENT.md](docs/nlp/DEPLOYMENT.md).

```bash
docker build -t mymltool-nlp:latest .

# offline: pre-download models, then transfer the image
python scripts/download_models.py --dest ./models
docker save -o mymltool-nlp.tar mymltool-nlp:latest    # docker load on target

# run (mount data / config / output / models)
docker compose run --rm nlp diagnose
docker compose run --rm nlp benchmark --config /config/classification.example.yaml
```

`data/`, `models/`, `output/`, and `*.tar` are git-ignored so datasets, model
files, generated reports, and delivery images are never committed.

---

## Compliance notes

Built for regulated/enterprise deployment: **no China-origin packages** (jieba/pkuseg excluded; spaCy character segmentation instead), every dependency and model tracked with its license and origin in [docs/nlp/LICENSES.md](docs/nlp/LICENSES.md), GPL/AGPL components kept out of the default image, PII masked in reports, `yaml.safe_load`, and safe (`weights_only=True`) checkpoint loading.

---

## Development

```bash
# full test suite (GPU/network/LightGBM tests skip-gate automatically)
python -m pytest

# a single category
python -m pytest -m unit          # or integration | stress | e2e | security
```

Layout: source in `src/`, tests mirror it under `tests/{unit,integration,stress,e2e,security}/`, docs in `docs/`, generated artifacts in `output/` (git-ignored). See `docs/workflow/workflow.md` and `docs/plan_nlp_pipeline.md`.
