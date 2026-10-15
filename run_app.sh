#!/bin/bash
# LiveChat AI Studio - 専用デスクトップアプリ（App Mode ランチャー）

SERVER_URL="${1:-https://13-231-80-112.sslip.io}"

echo "=================================================="
echo "  LiveChat AI Studio 専用デスクトップアプリを起動します"
echo "  安全な接続先 (HTTPS): $SERVER_URL"
echo "=================================================="

if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="$SERVER_URL" --window-size=1360,860 --user-data-dir="/tmp/livechat_app_profile"
else
    open -a Safari "$SERVER_URL"
fi
