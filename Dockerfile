# 中文文本 NLP Pipeline — 交付客戶設備的分析工具映像
#
# 主力平台: x86_64 + RTX 4070 (Ada, sm_89) / RTX 5070 Ti (Blackwell, sm_120)。
# 單一 CUDA 12.8 映像同時涵蓋兩張卡。GB10 (ARM64) 為預留設計，見 docs/nlp/DEPLOYMENT.md。
#
# 合規: 預設不安裝任何中國來源套件，也不安裝 GPL 的 ckip-transformers。
#
# 建置 (有網路環境):
#   docker build -t mymltool-nlp:latest .
# 離線交付:
#   docker save -o mymltool-nlp.tar mymltool-nlp:latest   # 傳輸到客戶設備
#   docker load -i mymltool-nlp.tar                        # 客戶設備載入

ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MyMLTool 中文文本 NLP Pipeline"
LABEL org.opencontainers.image.description="Traditional-Chinese text classification analysis and benchmark tool"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=0 \
    MPLBACKEND=Agg

WORKDIR /app

# Noto CJK 字型：matplotlib 圖表的繁中標籤才不會變成缺字方框。
USER root
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

# 依賴先裝，善用 layer cache。torch 已在 base image 內。
COPY requirements.txt requirements-nlp.txt ./
# requirements-nlp.txt 內的 torch 行由 base image 提供，用 --no-deps 避免覆寫；
# 其餘 NLP 依賴照裝。
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir transformers>=4.44 sentence-transformers>=3.0 \
        setfit>=1.0 lightgbm>=4.3 spacy>=3.7

# 應用程式碼
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 非 root 使用者
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser && \
    mkdir -p /data /config /output /models && \
    chown -R appuser:appuser /app /data /config /output /models
USER appuser

# 掛載點: /data(唯讀資料) /config(YAML) /output(報告) /models(離線模型)
VOLUME ["/data", "/config", "/output", "/models"]

ENTRYPOINT ["python", "-m", "src.nlp.cli"]
CMD ["diagnose"]
