#!/bin/bash
# LiveChat AI Studio - 専用デスクトップアプリ（App Mode ランチャー）

SERVER_URL="${1:-https://9hjxxdwunpfrc-8080.proxy.runpod.net}"

echo "=================================================="
echo "  LiveChat AI Studio 専用デスクトップアプリを起動します"
echo "  接続先サーバー: $SERVER_URL"
echo "=================================================="

# Google Chrome が存在する場合は最速の App Mode（アドレスバーなし独立ウィンドウ）で起動
if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="$SERVER_URL" --window-size=1360,860 --user-data-dir="/tmp/livechat_chrome_profile"
elif [ -d "/Applications/Brave Browser.app" ]; then
    open -na "Brave Browser" --args --app="$SERVER_URL" --window-size=1360,860
elif [ -d "/Applications/Microsoft Edge.app" ]; then
    open -na "Microsoft Edge" --args --app="$SERVER_URL" --window-size=1360,860
else
    # フォールバック: Safari で開く
    open -a Safari "$SERVER_URL"
fi
