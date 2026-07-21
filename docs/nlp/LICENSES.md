# 依賴與模型授權清單 — 公文 NLP Pipeline

> 政府採購合規用途。本清單須隨依賴變動同步更新，並附**來源國別**欄位。
> 核心合規原則：**禁用中國大陸來源套件/模型**；GPL 僅 opt-in 且不入預設交付映像；AGPL 禁用。

## Python 套件

| 套件 | 版本 | 授權 | 來源/維護方 | 國別 | 狀態 |
|------|------|------|-------------|------|------|
| numpy | ≥1.26 | BSD-3 | NumPy 社群 | 國際 | ✅ 採用 |
| pandas | ≥2.1 | BSD-3 | NumFOCUS | 國際 | ✅ 採用 |
| scikit-learn | ≥1.4 | BSD-3 | scikit-learn | 國際 | ✅ 採用 |
| scipy | ≥1.12 | BSD-3 | SciPy 社群 | 國際 | ✅ 採用 |
| pyyaml | ≥6.0 | MIT | PyYAML | 國際 | ✅ 採用 |
| matplotlib | ≥3.8 | PSF-based (BSD 相容) | Matplotlib | 國際 | ✅ 採用 |
| torch | ≥2.7 | BSD-3 | Meta / PyTorch Foundation | 美國 | ✅ 採用 |
| transformers | ≥4.44 | Apache-2.0 | Hugging Face | 美/法 | ✅ 採用 |
| tokenizers | (隨 transformers) | Apache-2.0 | Hugging Face | 美/法 | ✅ 採用 |
| sentence-transformers | ≥3.0 | Apache-2.0 | UKP Lab (TU Darmstadt) | 德國 | ✅ 採用 |
| setfit | ≥1.0 | Apache-2.0 | Hugging Face / Intel | 美國 | ✅ 採用 |
| lightgbm | ≥4.3 | MIT | Microsoft | 美國 | ✅ 採用 |
| spaCy | ≥3.7 | MIT | Explosion AI | 德國 | ✅ 採用 |

### 明確排除

| 套件/模型 | 授權 | 原因 |
|-----------|------|------|
| jieba | MIT | **中國大陸來源** — 政策禁用 |
| pkuseg | MIT | **中國大陸來源**（北京大學）— 政策禁用 |
| cleanlab | AGPL-3.0 | **AGPL 授權風險** — 改由 `analysis/label_quality.py` 自實作 confident-learning |
| hfl/chinese-* 模型 | Apache-2.0 | 模型權重**中國大陸來源**（哈工大 + 科大訊飛）— 政策禁用 |

> ⚠️ **spaCy 中文注意**：spaCy 官方中文模型 `zh_core_web_*` 底層預設呼叫 jieba/pkuseg。
> 本專案 `segment.py` 一律使用 `spacy.lang.zh.Chinese()` 空白管線（字元切分），
> **絕不**載入官方中文模型或設定 jieba/pkuseg segmenter。

## 預訓練模型

| 模型 | 授權 | 來源/維護方 | 國別 | 狀態 |
|------|------|-------------|------|------|
| google-bert/bert-base-chinese | Apache-2.0 | Google | 美國 | ✅ **BERT 預設** |
| FacebookAI/xlm-roberta-base | MIT | Meta | 美國 | ✅ 備選 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | UKP Lab；底座 microsoft/Multilingual-MiniLM | 德/美 | ✅ 句嵌入預設 |
| ckiplab/bert-base-chinese | **GPL-3.0** | 中央研究院 (Academia Sinica) | **台灣** | ⚠️ opt-in，需法務核可，**不入預設映像** |

### GPL 隔離說明

`ckip-transformers` 與 `ckiplab/*` 模型雖為**台灣**中研院來源（非中國，合乎來源政策），
但採 **GPL-3.0**。將 GPL 元件裝入交付給客戶的 Docker 映像，可能使整體散布構成
GPL 衍生/組合作品。因此：

1. 預設 `requirements-nlp.txt` **不含** ckip-transformers。
2. 預設 Docker 映像**不安裝** CKIP；提供獨立 build-arg 變體，需法務核可後才建置。
3. `segment.py` 的 `ckip` engine 與 CKIP BERT 模型皆為 opt-in，未安裝時明確報錯並提示上述限制。

## 授權盤點指令

新增任何依賴前，先確認來源國別與授權：

```bash
# 檢視已安裝套件授權
pip-licenses --format=markdown --with-urls
# 或針對單一套件
pip show <package>   # 看 License 與 Home-page/Author
```

判定流程：中國大陸來源 → 直接排除；GPL/AGPL → 標記法務審查、不入預設映像；
BSD/MIT/Apache → 可採用並登錄本表。
