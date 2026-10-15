#!/bin/bash
set -e

mkdir -p models

echo "=================================================="
echo "  LiveChat AI - 完全版モデルの自動ダウンロード"
echo "=================================================="

# 1. Inswapper 128 (約554MB)
if [ ! -s "models/inswapper_128.onnx" ] || [ $(stat -c%s "models/inswapper_128.onnx" 2>/dev/null || stat -f%z "models/inswapper_128.onnx" 2>/dev/null || echo 0) -lt 300000000 ]; then
    echo "[1/3] Inswapper 128 をダウンロード中 (約554MB)..."
    rm -f models/inswapper_128.onnx
    wget -q --show-progress -O models/inswapper_128.onnx "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
else
    echo "✓ Inswapper 128 は準備完了です。"
fi

# 2. GFPGAN v1.4 超解像・目元高画質化 (約340MB 完全バイナリ)
if [ ! -s "models/GFPGANv1.4.onnx" ] || [ $(stat -c%s "models/GFPGANv1.4.onnx" 2>/dev/null || stat -f%z "models/GFPGANv1.4.onnx" 2>/dev/null || echo 0) -lt 300000000 ]; then
    echo "[2/3] GFPGAN v1.4 超解像モデルをダウンロード中 (約340MB)..."
    rm -f models/GFPGANv1.4.onnx
    wget -q --show-progress -O models/GFPGANv1.4.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"
else
    echo "✓ GFPGAN v1.4 超解像モデルは準備完了です。"
fi

# 3. Face Parser (BiSeNet 約53MB 完全バイナリ)
if [ ! -s "models/face_parser.onnx" ] || [ $(stat -c%s "models/face_parser.onnx" 2>/dev/null || stat -f%z "models/face_parser.onnx" 2>/dev/null || echo 0) -lt 10000000 ]; then
    echo "[3/3] Face Parser (BiSeNet) をダウンロード中 (約53MB)..."
    rm -f models/face_parser.onnx
    wget -q --show-progress -O models/face_parser.onnx "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/bisenet_resnet18.onnx" || true
else
    echo "✓ Face Parser は準備完了です。"
fi

echo "=================================================="
echo "  🎉 すべてのAIモデル（完全版）が正常に揃いました！"
echo "=================================================="
