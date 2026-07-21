# 中文文本 (繁中) NLP 分類分析與選型 Pipeline — 實作計畫

> 核准日期：2026-07-21（含深度 re-review 修訂）。原始計畫與討論記錄見 git 歷史。

## Context

現有 `MyMLTool` 是通用 ML 模板（`Classification/`、`Regression/` 教學腳本 + `src/data_prep.py`、`src/feature_shift.py`），完全沒有 NLP、PyTorch、GPU 相關程式碼。

需求：工具要用在**中文文本領域（繁體中文 NLP 文本分類）**，但**目前尚無資料**。因此預先建 pipeline，資料到位即可：
1. 對中文文本做**資料分析（EDA）**，判斷資料特性（長度、類別分布、不平衡、標註品質…）。
2. **benchmark 多種候選演算法**，用分析結果決定前處理與模型選型。

## 已確認決策

- **標籤結構**：三種都支援 → 單標籤多類別、不平衡多類別、多標籤（分析後選）。
- **交付物**：EDA 分析模組 + 可插拔訓練/評估 benchmark 框架。
- **合規（法規遵循／企業部署要求）**：
  - **禁用中國來源套件/模型**：jieba、pkuseg、hfl 系列（哈工大+科大訊飛）全部剔除。
  - 斷詞用 **spaCy**（MIT）`Chinese(segmenter="char")` — 必須明確指定 char，因 spaCy 中文模型底層預設呼叫 jieba/pkuseg。
  - 每個依賴與預訓練模型記入 `docs/nlp/LICENSES.md` 含**來源國別**。
  - GPL（ckip-transformers，台灣中研院）opt-in、**不裝入預設交付映像**；AGPL（cleanlab）禁用 → confident-learning 自實作。
- **候選演算法**（10 個，經 review 定案）：
  - 傳統：TF-IDF + LogReg / LinearSVM / Naive Bayes / Tree；TF-IDF/SVD + LightGBM。
  - 輕量深度：TextCNN、BiLSTM+Attention（PyTorch）。
  - 句嵌入(凍結) + 線性分類器（sentence-transformers）；SetFit（少樣本對比微調）。
  - BERT 微調：預設 `google-bert/bert-base-chinese`（Apache-2.0），備選 `xlm-roberta-base`（MIT）；CKIP opt-in。
  - 不含 LLM/few-shot。
- **長文本**：EDA 回報超過 512 token 比例；BERT 模組支援截斷與滑窗。
- **GPU**：主力 RTX 4070（Ada, sm_89）+ 5070 Ti（Blackwell, sm_120）；GB10 Grace-Blackwell（ARM64, sm_121）預留。torch ≥2.7 / CUDA 12.8+。
- **部署**：Docker 映像交付客戶設備，單一 x86_64 cu128 映像涵蓋兩張主力卡；離線交付（`docker save/load` + `HF_HUB_OFFLINE`）；附 `DEPLOYMENT.md`。
- **開發環境**：repo `.venv` 為 WSL venv（Ubuntu 24.04），測試於 WSL 執行；WSL 內可見 RTX 4070。

**設計主軸**：`torch`/`transformers`/`spacy` 一律 lazy import；純 Python 核心（config/eda/metrics/labels/synth/TF-IDF）不依賴 torch。程式產出全部到 `output/nlp/`。

## 模組佈局

```
src/nlp/
  __init__.py
  config.py       # YAML -> dataclass；task_type/model/segmenter/device/離線路徑
  device.py       # GPU 偵測與精度策略（Ada + Blackwell + GB10/ARM64）
  segment.py      # 斷詞抽象：spacy(char) / char / bert / ckip(opt-in)
  labels.py       # LabelSpace：單/多標籤編碼、is_multilabel、class 統計
  synth.py        # 合成繁中文本假資料（balanced/imbalanced/multilabel + 注入瑕疵）
  eda.py          # 文本 EDA -> TextEdaReport + 圖表
  metrics.py      # 依 task_type 分派指標（含 balanced acc、macro PR-AUC）
  datasets.py     # 文字 CSV 載入、train/val/test 分層三向切分、多標籤 | 分隔
  harness.py      # benchmark 驅動器：seed_everything + 同一切分跑所有模型
  report.py       # EDA/benchmark/feature-select 報告輸出（md+json）到 output/nlp/
  cli.py          # CLI：eda / feature-select / benchmark / diagnose / download-models
  models/
    base.py         # TextClassifier ABC + FitReport；predict 由 label_space 統一分派
    registry.py     # 名稱->模組/類別 靜態註冊表（lazy import）
    tfidf_linear.py # logreg/linearsvm/nb/tree 四變體（無 torch）
    tfidf_gbm.py    # TF-IDF/SVD + LightGBM（無 torch）
    sent_embed.py   # 凍結句嵌入 + LogReg
    setfit_clf.py   # SetFit
    textcnn.py      # PyTorch TextCNN
    bilstm_attn.py  # PyTorch BiLSTM+Attention
    bert_finetune.py# HF 微調（AMP/bf16、離線路徑、滑窗）
  analysis/
    difficulty.py   # PCA/t-SNE、silhouette、linear-probe、learning curve（預設 TF-IDF+SVD 向量，無 torch）
    label_quality.py# confident-learning 自實作（避 cleanlab AGPL）
    keywords.py     # chi²/MI 每類關鍵詞
    pii.py          # 台灣 PII regex + 全形半形/OCR 雜訊檢查
    feature_selection.py # 特徵篩選建議：詞彙(χ²/MI/ANOVA/L1/樹)+特徵數曲線+n-gram推薦+冗餘；
                    #   metadata 相關性(Cramér's V/Pearson/VIF)+建議保留/剔除（純 sklearn/scipy）
```

**Feature Selection（分析與推薦報告，資料含 metadata 時）**：`analysis/feature_selection.py`
產出「該用哪種前處理」的量化依據——詞彙層 χ²/MI/ANOVA/L1/樹重要度五法比較 + 保留特徵數
vs macro-F1 曲線（推薦 `max_features`）+ n-gram range 推薦 + 冗餘詞對；metadata 層對標籤
相關性 + 特徵間冗餘（Cramér's V / Pearson / 手算 VIF）+ 保守的保留/剔除建議。詞彙層 FS 主要
服務 TF-IDF baseline，BERT/句嵌入不受影響。DataConfig 新增 `metadata_cols`。

沿用既有慣例：模組 docstring、頂部常數區塊、小函式、dataclass 結果容器、type hints、明確例外、絕對匯入。

## 共同模型介面

- `TextClassifier` ABC：`build(label_space, model_config, device_config)` / `fit(texts, y, val_texts, val_y) -> FitReport` / `predict_proba` / `predict`（base 依 `label_space.is_multilabel` 統一 argmax vs threshold）/ `save` / `load`。
- **不平衡緩解**：config `class_weight`（none/balanced）→ sklearn `class_weight`、torch weighted CE。
- `harness.run_benchmark(config)`：seed_everything → 同一組切分跑所有模型 → `BenchmarkResult`。

## device.py 重點

- `DeviceInfo`：cuda 可用性、裝置名、compute capability、架構判別（Ada 8.9/Blackwell 12.0/GB10 12.1+aarch64）、`torch.version.cuda`、bf16 支援。
- **防呆**：device cc 與 `torch.cuda.get_arch_list()` 比對，wheel 缺 sm_120/121 時開跑前明確報錯。
- 精度 auto：CUDA→bf16（不支援退 fp16）、CPU→fp32；`torch.compile` 保守啟用 + eager fallback。
- 全程可 CPU fallback；`print_diagnostics()` 供裝機驗證。

## EDA 涵蓋

長度分布（char+token, p95, >512 比例）、類別分布與不平衡指標、詞彙統計（vocab/TTR/hapax/top n-gram）、重複偵測、跨切分洩漏、標註品質（同文異標/空文本）、PII 掃描（台灣身分證/電話/地址）、正規化檢查（全形半形/OCR 雜訊）。輸出 `output/nlp/eda/`（md+json+PNG, Agg backend）。

選型階段（資料到位後）：difficulty（可分性+learning curve）、label_quality（疑似標錯清單）、keywords（判別詞表）。

## 測試計畫

- `pytest.ini` 新增 markers：`gpu`、`slow`、`network`。
- unit/integration/stress/e2e/security 五類照舊；DL 測試極小（~40 篇、≤2 epoch、CPU 可跑）。
- **零網路**：BERT 測試用本地建構微型隨機權重 BERT（小 BertConfig+臨時 vocab）；需真模型的 sentence-transformers/SetFit 測試以 `network` marker gate。
- GPU 測試 `skipif` 無 CUDA。

## 依賴與 License

`requirements.txt`（輕量核心）+pyyaml、matplotlib；`requirements-nlp.txt`（重量級）torch≥2.7(BSD-3)、transformers≥4.44(Apache-2.0)、sentence-transformers≥3.0(Apache-2.0)、setfit≥1.0(Apache-2.0)、lightgbm≥4.3(MIT)、spacy≥3.7(MIT)。torch 安裝矩陣記於 `docs/nlp/INSTALL.md`（x86_64→cu128、GB10→aarch64 wheel、CI→cpu）。

模型合規清單（`docs/nlp/LICENSES.md`，含來源國別）：google-bert/bert-base-chinese（Google, Apache-2.0, 預設）、xlm-roberta-base（Meta, MIT）、paraphrase-multilingual-MiniLM-L12-v2（底座 Microsoft, Apache-2.0）、ckiplab/*（台灣中研院, GPL-3.0, opt-in）。

## Docker 部署

- Base：`pytorch/pytorch:2.7.*-cuda12.8-cudnn9-runtime`；GB10/ARM64 以 build-arg 預留。
- Entrypoint `python -m src.nlp.cli`：`diagnose`/`eda`/`benchmark`/`download-models`。
- Volumes：`/data`(ro)、`/config`、`/output`、`/models`（`HF_HOME` + `HF_HUB_OFFLINE=1`）。
- 離線交付：build+download-models → `docker save` → 客戶 `docker load` → `docker run --gpus all`。
- 前置：driver ≥570、NVIDIA Container Toolkit、Windows 需 WSL2。
- 預設映像不含 GPL 套件；`configs/*.example.yaml` 隨附。
- 文件：`docs/nlp/DEPLOYMENT.md`（前置檢查/載入/diagnose 驗證/跑分析/疑難排解/GB10 預留）。

## 分階段實作

| 階段 | 內容 | 相依 |
|------|------|------|
| 0 | 核心：config、labels、metrics、synth、segment | 無 |
| 1 | eda + report + 圖表（含 PII/正規化/512 檢查） | 0 |
| 2 | models/base、registry、tfidf_linear、tfidf_gbm、datasets、harness | 0 |
| 3 | device.py + 診斷 | 無 |
| 4 | textcnn、bilstm_attn | 2,3 |
| 5 | bert_finetune + sent_embed + setfit_clf | 2,3 |
| 6 | analysis/（TF-IDF+SVD 路徑） | 2 |
| 7 | benchmark 串接 + cli.py + e2e | 2,4,5 |
| 8 | Docker + DEPLOYMENT.md | 7 |
| 9 | 文件 + 三份報告 | 全部 |

## 驗證方式

1. EDA：合成 imbalanced 資料 → `output/nlp/eda/` 有完整報告與 PNG。
2. Benchmark（CPU）：TF-IDF + TextCNN ≤2 epoch → 報告含 F1 對照表。
3. 多標籤路徑：task_type=multilabel → subset acc/hamming/per-label F1。
4. GPU 診斷：WSL + RTX 4070 實測 `diagnose` 正確辨識 Ada/bf16；無 GPU fallback CPU。
5. `pytest` 全綠（GPU/network 測試自動 skip），報告覆蓋率。
6. Docker：build → diagnose → 容器內合成資料 EDA+mini-benchmark → save/load 演練。
7. 依 CLAUDE.md 產出三份報告 + 更新 CHANGELOG/flow/workflow。
