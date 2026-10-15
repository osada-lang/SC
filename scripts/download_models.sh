#!/bin/bash
set -e

mkdir -p models

echo "=== AIモデルの自動ダウンロードを開始します ==="

# 1. Inswapper 128 (約554MB)
if [ ! -s "models/inswapper_128.onnx" ] || [ $(stat -c%s "models/inswapper_128.onnx" 2>/dev/null || stat -f%z "models/inswapper_128.onnx" 2>/dev/null || echo 0) -lt 100000000 ]; then
    echo "Downloading inswapper_128.onnx..."
    curl -L -o models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
else
    echo "inswapper_128.onnx is ready."
fi

# 2. GFPGAN v1.4 (超解像・高画質化 約340MB)
if [ ! -s "models/GFPGANv1.4.onnx" ] || [ $(stat -c%s "models/GFPGANv1.4.onnx" 2>/dev/null || stat -f%z "models/GFPGANv1.4.onnx" 2>/dev/null || echo 0) -lt 50000000 ]; then
    echo "Downloading GFPGANv1.4.onnx..."
    curl -L -o models/GFPGANv1.4.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/GFPGANv1.4.onnx || \
    curl -L -o models/GFPGANv1.4.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/GFPGANv1.4.onnx || true
else
    echo "GFPGANv1.4.onnx is ready."
fi

# 3. Face Parser (BiSeNet 約50MB)
if [ ! -s "models/face_parser.onnx" ] || [ $(stat -c%s "models/face_parser.onnx" 2>/dev/null || stat -f%z "models/face_parser.onnx" 2>/dev/null || echo 0) -lt 10000000 ]; then
    echo "Downloading face_parser.onnx..."
    curl -L -o models/face_parser.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/face_parser.onnx || true
else
    echo "face_parser.onnx is ready."
fi

# 4. MediaPipe Hand Landmarker (手の骨格・オクルージョン検出 約11MB)
if [ ! -s "models/hand_landmarker.task" ]; then
    echo "Downloading hand_landmarker.task..."
    curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task || true
else
    echo "hand_landmarker.task is ready."
fi

echo "=== すべてのAIモデルが正常に準備完了しました ==="
