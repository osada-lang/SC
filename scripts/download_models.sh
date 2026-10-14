#!/bin/bash
set -e

mkdir -p models

echo "=== AIモデルの自動ダウンロードを開始します ==="

# 1. Inswapper 128
if [ ! -f "models/inswapper_128.onnx" ]; then
    echo "Downloading inswapper_128.onnx..."
    curl -L -o models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
else
    echo "inswapper_128.onnx already exists."
fi

# 2. Face Parser (BiSeNet)
if [ ! -f "models/face_parser.onnx" ]; then
    echo "Downloading face_parser.onnx..."
    curl -L -o models/face_parser.onnx https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/bisenet_resnet18.onnx || true
else
    echo "face_parser.onnx already exists."
fi

# 3. GFPGAN v1.4
if [ ! -f "models/GFPGANv1.4.onnx" ]; then
    echo "Downloading GFPGANv1.4.onnx..."
    curl -L -o models/GFPGANv1.4.onnx https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/GFPGANv1.4.onnx || \
    curl -L -o models/GFPGANv1.4.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/GFPGANv1.4.onnx || true
else
    echo "GFPGANv1.4.onnx already exists."
fi

echo "=== モデルの準備が完了しました ==="
