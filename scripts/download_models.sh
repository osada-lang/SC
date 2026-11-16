#!/bin/bash
set -e

mkdir -p models

echo "=================================================="
echo "  LiveChat AI - 完全版モデルの自動ダウンロード"
echo "=================================================="

# 1. Ghost 256 (カメレオンAI同系列 本命モデル 約739MB)
if [ ! -s "models/ghost_2_256.onnx" ] || [ $(stat -c%s "models/ghost_2_256.onnx" 2>/dev/null || stat -f%z "models/ghost_2_256.onnx" 2>/dev/null || echo 0) -lt 500000000 ]; then
    echo "[1/4] Ghost 256 (カメレオンAI同系列モデル) をダウンロード中 (約739MB)..."
    rm -f models/ghost_2_256.onnx
    wget -q --show-progress -O models/ghost_2_256.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_2_256.onnx"
else
    echo "✓ Ghost 256 は準備完了です。"
fi

# 2. ArcFace to Ghost Embedding Converter (約21MB)
if [ ! -s "models/arcface_converter_ghost.onnx" ] || [ $(stat -c%s "models/arcface_converter_ghost.onnx" 2>/dev/null || stat -f%z "models/arcface_converter_ghost.onnx" 2>/dev/null || echo 0) -lt 15000000 ]; then
    echo "[2/4] ArcFace-to-Ghost 特徴変換コンバーターをダウンロード中 (約21MB)..."
    rm -f models/arcface_converter_ghost.onnx
    wget -q --show-progress -O models/arcface_converter_ghost.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_converter_ghost.onnx"
else
    echo "✓ ArcFace to Ghost Converter は準備完了です。"
fi

# 3. GFPGAN v1.4 超解像・目元高画質化 (約340MB 完全バイナリ)
if [ ! -s "models/GFPGANv1.4.onnx" ] || [ $(stat -c%s "models/GFPGANv1.4.onnx" 2>/dev/null || stat -f%z "models/GFPGANv1.4.onnx" 2>/dev/null || echo 0) -lt 300000000 ]; then
    echo "[3/4] GFPGAN v1.4 超解像モデルをダウンロード中 (約340MB)..."
    rm -f models/GFPGANv1.4.onnx
    wget -q --show-progress -O models/GFPGANv1.4.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"
else
    echo "✓ GFPGAN v1.4 超解像モデルは準備完了です。"
fi

# 4. Inswapper 128 (バックアップ互換用 約554MB)
if [ ! -s "models/inswapper_128.onnx" ] || [ $(stat -c%s "models/inswapper_128.onnx" 2>/dev/null || stat -f%z "models/inswapper_128.onnx" 2>/dev/null || echo 0) -lt 300000000 ]; then
    echo "[4/4] Inswapper 128 をダウンロード中 (約554MB)..."
    rm -f models/inswapper_128.onnx
    wget -q --show-progress -O models/inswapper_128.onnx "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
else
    echo "✓ Inswapper 128 は準備完了です。"
fi

echo "=================================================="
echo "  🎉 すべてのAIモデル（完全版）が正常に揃いました！"
echo "=================================================="
