#!/bin/bash
set -e

IP="13.231.80.112"
DOMAIN="13-231-80-112.sslip.io"

echo "=================================================="
echo "  LiveChat AI Studio - 完全HTTPS (SSL) 自動構築"
echo "  ドメイン: https://${DOMAIN}"
echo "=================================================="

# 1. Caddy (自動SSL・高性能リバースプロキシ) のインストール
if ! command -v caddy &> /dev/null; then
    echo "[1/3] Caddy をインストール中..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg --yes
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq caddy
fi

# 2. Caddyfile の作成 (自動SSL + 8080リバースプロキシ)
echo "[2/3] HTTPS & SSL自動証明書を設定中..."
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
${DOMAIN} {
    reverse_proxy 127.0.0.1:8080 {
        header_up Host {host}
        header_up X-Real-IP {remote}
        header_up X-Forwarded-For {remote}
        header_up X-Forwarded-Proto {scheme}
    }
}
EOF

# 3. Caddy サービスの再起動
echo "[3/3] SSLサービスを起動中..."
sudo systemctl restart caddy
sudo systemctl enable caddy

echo "=================================================="
echo "  🎉 完全HTTPS化が完了いたしました！"
echo "  アクセスURL: https://${DOMAIN}"
echo "=================================================="
