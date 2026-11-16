#!/bin/bash
set -e

mkdir -p models

echo "=================================================="
echo "  LiveChat AI - 完全版モデルの自動ダウンロード"
echo "=================================================="

# 1. SimSwap 512 (ネイティブ512px高解像度モデル 約239MB)
if [ ! -s "models/simswap_unofficial_512.onnx" ] || [ $(stat -c%s "models/simswap_unofficial_512.onnx" 2>/dev/null || stat -f%z "models/simswap_unofficial_512.onnx" 2>/dev/null || echo 0) -lt 200000000 ]; then
    echo "[1/4] SimSwap 512 (512px高解像度モデル) をダウンロード中 (約239MB)..."
    rm -f models/simswap_unofficial_512.onnx
    wget -q --show-progress -O models/simswap_unofficial_512.onnx "https://huggingface.co/LPDoctor/simswap_official/resolve/main/simswap_unofficial_512.onnx"
else
    echo "✓ SimSwap 512 は準備完了です。"
fi

# 2. ArcFace to SimSwap Embedding Converter (約21MB)
if [ ! -s "models/arcface_converter_simswap.onnx" ] || [ $(stat -c%s "models/arcface_converter_simswap.onnx" 2>/dev/null || stat -f%z "models/arcface_converter_simswap.onnx" 2>/dev/null || echo 0) -lt 15000000 ]; then
    echo "[2/4] ArcFace-to-SimSwap 特徴変換コンバーターをダウンロード中 (約21MB)..."
    rm -f models/arcface_converter_simswap.onnx
    wget -q --show-progress -O models/arcface_converter_simswap.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_converter_simswap.onnx"
else
    echo "✓ ArcFace Converter は準備完了です。"
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
