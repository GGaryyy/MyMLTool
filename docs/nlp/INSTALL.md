# 安裝指南 — 公文 NLP Pipeline

本頁說明如何在不同目標平台安裝 PyTorch 與 NLP 依賴。授權合規見 [LICENSES.md](LICENSES.md)，
容器部署見 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 核心依賴（CPU、CI）

```bash
pip install -r requirements.txt
```

僅含 numpy / pandas / scikit-learn / scipy / pyyaml / matplotlib 與測試工具。
`src/nlp` 的純 Python 核心（config、labels、metrics、synth、segment(char)、EDA、
TF-IDF baseline、analysis）在此環境即可完整運作與測試。

## PyTorch 安裝矩陣

深度模型（TextCNN / BiLSTM / BERT / 句嵌入 / SetFit）需要 PyTorch。
**務必先依平台裝對 torch wheel，再裝其餘 NLP 依賴。**

| 目標平台 | GPU / 架構 | 指令 |
|----------|-----------|------|
| x86_64 + RTX 4070 / 5070 Ti | Ada sm_89 / Blackwell sm_120 | `pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128` |
| GB10 / Grace-Blackwell | ARM64 aarch64, sm_121 | 使用 NVIDIA aarch64 wheel 或 NGC PyTorch 容器（見下） |
| CPU-only（CI / 無 GPU） | — | `pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu` |

> **為何是 CUDA 12.8**：RTX 5070 Ti（Blackwell, sm_120）需要 CUDA 12.8+ 的 kernel；
> 舊 wheel 雖能 `import torch` 且 `cuda.is_available()==True`，卻會在訓練時噴
> `no kernel image is available`。本專案 `src/nlp/device.py` 會在訓練前主動比對
> `torch.cuda.get_arch_list()` 與裝置 compute capability，提前擋下此類不相容。

裝完 torch 後：

```bash
pip install -r requirements-nlp.txt
```

### GB10 / aarch64 補充

GB10（DGX Spark）為 ARM64 平台，x86_64 wheel 不適用。建議二選一：

1. **NGC 容器**：以 `nvcr.io/nvidia/pytorch:<tag>` 為 base（已含 aarch64 + Blackwell CUDA）。
2. **NVIDIA aarch64 wheel**：依 NVIDIA 官方對應版本安裝 torch，再 `pip install -r requirements-nlp.txt`。

本階段 GB10 為**預先設計**：Docker 以 build-arg 預留 multi-arch，實際 arm64 映像待硬體到位再建。

## 驗證安裝

```bash
# 裝置與 wheel 相容性診斷（無 GPU 也可跑，會顯示 cpu fallback）
python -m src.nlp.cli diagnose

# 或直接呼叫
python -c "from src.nlp.device import print_diagnostics; print_diagnostics()"
```

預期在 RTX 4070 上輸出 `architecture: ada`、`precision: bf16`、
`wheel/device compatibility: OK`。

## 斷詞引擎（spaCy）

`segment.engine: spacy` 使用 spaCy 空白中文管線的**字元切分**（合規，不觸及 jieba/pkuseg）。
spaCy 已列於 `requirements-nlp.txt`。若環境未安裝 spaCy，pipeline 會自動退回純字元切分
（語義相同），EDA 仍可執行。

## 開發環境（本 repo）

本 repo 的 `.venv` 為 WSL (Ubuntu 24.04) venv。測試於 WSL 執行：

```bash
wsl -e bash -lc "cd /mnt/d/IT/githubProbject/MyMLTool && .venv/bin/python -m pytest -q"
```
