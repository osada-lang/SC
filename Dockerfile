# NVIDIA CUDA 12.1 + Ubuntu 22.04 ベースイメージ
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# システムパッケージとPython 3.10のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# pythonコマンドのリンク設定
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace

# pipのアップグレード
RUN python -m pip install --upgrade pip setuptools wheel

# サーバー依存パッケージのインストール
COPY requirements-server.txt /workspace/
RUN pip install --no-cache-dir onnxruntime-gpu==1.16.3 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || \
    pip install --no-cache-dir onnxruntime-gpu==1.16.3 || \
    pip install --no-cache-dir onnxruntime
RUN pip install --no-cache-dir -r requirements-server.txt

# モデル・ソースコードのコピー
COPY models/ /workspace/models/
COPY target.jpg /workspace/target.jpg
COPY server/ /workspace/server/

EXPOSE 8080

WORKDIR /workspace/server

CMD ["python", "app.py"]
