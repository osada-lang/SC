#!/bin/bash
# LiveChat AI Studio - ワンクリック起動ランチャー

SERVER_URL="${1:-https://9hjxxdwunpfrc-8080.proxy.runpod.net}"

echo "=================================================="
echo "  LiveChat AI Studio 専用デスクトップアプリを起動します"
echo "  接続先サーバー: $SERVER_URL"
echo "=================================================="

# 必要なライブラリの確認
pip3 install -q pywebview requests 2>/dev/null || true

# デスクトップアプリの起動
python3 scripts/desktop_app.py --server "$SERVER_URL"
