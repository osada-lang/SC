#!/bin/bash
# ========================================================
#  LiveChat AI Studio - ワンクリック起動ランチャー (Mac用)
# ========================================================

SERVER_IP="${1:-54.249.95.222}"
SERVER_URL="http://${SERVER_IP}:8080"

echo "=================================================="
echo "  LiveChat AI Studio を起動しています..."
echo "  接続先サーバー: $SERVER_URL"
echo "=================================================="

# カメラ自動許可フラグ付きで専用アプリを起動
open -na "Google Chrome" --args \
  --app="$SERVER_URL" \
  --window-size=1360,860 \
  --user-data-dir="/tmp/livechat_app_profile" \
  --unsafely-treat-insecure-origin-as-secure="$SERVER_URL"

exit 0
