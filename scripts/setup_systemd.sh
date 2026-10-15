#!/bin/bash
set -e

echo "=== LiveChat AI サーバー常駐・自動起動サービスを登録します ==="

sudo cp scripts/livechat-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable livechat-ai.service
sudo systemctl restart livechat-ai.service

echo "=== サービス登録完了: サーバー起動時に自動起動します ==="
