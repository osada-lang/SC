import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="LiveChat AI Studio - 専用デスクトップアプリ")
    parser.add_argument("--server", default="https://9hjxxdwunpfrc-8080.proxy.runpod.net", help="クラウドサーバーのURL")
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    print("==================================================")
    print("  LiveChat AI Studio - 専用デスクトップアプリ起動")
    print(f"  サーバー接続先: {server_url}")
    print("==================================================")

    # macOS App Mode 起動 (アドレスバー・タブなし独立ウィンドウ)
    chrome_path = "/Applications/Google Chrome.app"
    if os.path.exists(chrome_path):
        subprocess.Popen([
            "open", "-na", "Google Chrome", "--args",
            f"--app={server_url}",
            "--window-size=1360,860",
            "--user-data-dir=/tmp/livechat_chrome_profile"
        ])
    else:
        subprocess.Popen(["open", server_url])

if __name__ == "__main__":
    main()
