#!/bin/bash
# ========================================================
#  LiveChat AI Studio - 全自動サーバー連動ランチャー (Mac用)
# ========================================================

APP_DIR="$HOME/dev/SC"

if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    python3 "$APP_DIR/scripts/launch_app.py" "$@"
else
    echo "エラー: $APP_DIR フォルダが見つかりません。"
    read -p "Press enter to exit..."
fi

exit 0
