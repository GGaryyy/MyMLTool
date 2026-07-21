# 部署指南 — 中文文本 NLP Pipeline (Docker 交付)

交付給客戶設備 / 現場工程師的容器安裝與使用說明。安裝細節見 [INSTALL.md](INSTALL.md)，
授權合規見 [LICENSES.md](LICENSES.md)。

目標設備：RTX 4070 / RTX 5070 Ti（主力），GB10 Grace-Blackwell（預留）。

---

## 1. 前置需求檢查

| 項目 | 需求 | 檢查指令 |
|------|------|----------|
| NVIDIA 驅動 | ≥ 570（Blackwell/5070 Ti 需求；4070 亦相容） | `nvidia-smi` |
| Docker | ≥ 24 | `docker --version` |
| NVIDIA Container Toolkit | 已安裝並設定 | `nvidia-ctk --version` |
| GPU 直通測試 | 容器內看得到 GPU | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |

- **RTX 4070**：driver 570+ 即可，Ada 架構，bf16 支援。
- **RTX 5070 Ti**：Blackwell sm_120，**務必** driver 570+ 與 CUDA 12.8 映像，否則訓練報
  `no kernel image is available`。
- **Windows 主機**：安裝 Docker Desktop 並啟用 WSL2 backend；GPU 直通需 WSL2 + 最新 NVIDIA Windows 驅動。

---

## 2. 映像載入

**線上環境**（可連網）：
```bash
docker build -t mymltool-nlp:latest .
```

**離線交付**（客戶air-gapped設備）：
```bash
# 交付方 (有網路):
docker build -t mymltool-nlp:latest .
python scripts/download_models.py --dest ./models      # 預下載合規模型
docker save -o mymltool-nlp.tar mymltool-nlp:latest

# 客戶設備:
docker load -i mymltool-nlp.tar
```

`models/` 資料夾與 `mymltool-nlp.tar` 一併交付，掛載到容器 `/models`。

---

## 3. 裝機驗證（第一步必做）

```bash
docker run --rm --gpus all mymltool-nlp:latest diagnose
```

**RTX 4070 預期輸出**（節錄）：
```
device name        : NVIDIA GeForce RTX 4070
compute capability : 8.9
architecture       : ada
bf16 supported     : True
precision          : bf16
wheel/device compatibility: OK
```

**RTX 5070 Ti 預期**：`architecture : blackwell`、`compute capability : 12.0`、
`wheel/device compatibility: OK`。

若出現 `MISMATCH` → torch wheel 缺該卡 kernel，請確認用的是 CUDA 12.8 映像（見疑難排解）。

---

## 4. 執行分析

掛載資料/設定/輸出目錄。以 docker compose 最簡便：

```bash
# EDA 資料分析
docker compose run --rm nlp eda --config /config/eda.example.yaml

# 演算法選型 benchmark
docker compose run --rm nlp benchmark --config /config/benchmark.example.yaml
```

或用純 docker：
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/configs:/config:ro" \
  -v "$(pwd)/output:/output" \
  -v "$(pwd)/models:/models" \
  -e HF_HUB_OFFLINE=1 \
  mymltool-nlp:latest benchmark --config /config/benchmark.example.yaml
```

資料 CSV 放 `data/`，設定改 `configs/*.yaml`（`csv_path` 指向 `/data/你的檔.csv`）。

---

## 5. 產出位置

所有報告與圖表寫入掛載的 `/output`（對應主機 `./output/`）：

```
output/nlp/
├── eda/
│   ├── eda_report.md        # 長度分布/類別不平衡/PII/正規化/標註品質/建議
│   ├── eda_report.json
│   └── plots/*.png
└── benchmark/
    ├── benchmark_report.md  # 各模型 F1 對照與選型建議
    ├── benchmark_report.json
    └── plots/*.png
```

---

## 6. 疑難排解

| 症狀 | 原因 | 解法 |
|------|------|------|
| `diagnose` 顯示 `wheel/device compatibility: MISMATCH` | torch wheel 缺該 GPU 架構 kernel | 確認使用 CUDA 12.8 base image（5070 Ti/GB10 必需） |
| `no kernel image is available` | 同上，且未被 diagnose 擋下 | 重建映像用 `pytorch:2.7+-cuda12.8` base |
| 容器看不到 GPU | NVIDIA Container Toolkit 未設定 | 安裝並 `sudo nvidia-ctk runtime configure` |
| `CUDA out of memory` | batch_size 過大 | 調小 config 的 `batch_size` / `max_length` |
| 離線環境找不到模型 | 未預下載或路徑錯 | 跑 `download_models.py`，確認掛 `/models` 且 `HF_HUB_OFFLINE=1` |
| driver 太舊 | driver < 570 | 更新 NVIDIA 驅動 |
| BERT 對超長文本截斷 | 文本 > 512 token | config 開啟滑窗（見 benchmark 範例），或參考 EDA 的 >512 比例 |

---

## 7. GB10 / ARM64（未來支援）

GB10（DGX Spark，ARM64 aarch64）目前為**預留設計**：

- Dockerfile 以 `ARG BASE_IMAGE` 預留，屆時改用 NVIDIA 官方 aarch64 CUDA base 或 NGC PyTorch 容器（含 Blackwell + aarch64 kernel）。
- 建置：`docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:<aarch64-tag> ...`
- `device.py` 已能辨識 GB10（cc 12.1 + aarch64 → `blackwell-gb10`），程式面無需改動。
- 實際 arm64 映像待硬體到位後建置與驗證。
