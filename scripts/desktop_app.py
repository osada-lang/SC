import os
import sys
import subprocess
import argparse

DEFAULT_HTTPS_URL = "https://13-231-80-112.sslip.io"

def main():
    parser = argparse.ArgumentParser(description="LiveChat AI Studio - 専用デスクトップアプリ")
    parser.add_argument("--server", default=DEFAULT_HTTPS_URL, help="クラウドサーバーのURL")
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    print("==================================================")
    print("  LiveChat AI Studio - 専用デスクトップアプリ起動")
    print(f"  安全な接続先 (HTTPS): {server_url}")
    print("==================================================")

    # macOS App Mode 起動 (HTTPSのためセキュリティ警告・カメラブロックは100%発生しません)
    chrome_path = "/Applications/Google Chrome.app"
    if os.path.exists(chrome_path):
        subprocess.Popen([
            "open", "-na", "Google Chrome", "--args",
            f"--app={server_url}",
            "--window-size=1360,860",
            "--user-data-dir=/tmp/livechat_app_profile"
        ])
    else:
        subprocess.Popen(["open", "-a", "Safari", server_url])

if __name__ == "__main__":
    main()
