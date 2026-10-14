import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="LiveChat AI Studio - 専用デスクトップアプリ")
    parser.add_argument("--server", default="https://9hjxxdwunpfrc-8080.proxy.runpod.net", help="クラウドサーバーのURL")
    args = parser.parse_args()

    # pywebview の確認と自動案内
    try:
        import webview
    except ImportError:
        print("[LiveChat] 専用デスクトップGUIの依存ライブラリを準備中 (pywebview)...")
        os.system(f"{sys.executable} -m pip install pywebview")
        import webview

    server_url = args.server.rstrip("/")
    print("==================================================")
    print("  LiveChat AI Studio - 専用デスクトップアプリ起動")
    print(f"  サーバー接続先: {server_url}")
    print("==================================================")

    # ネイティブウィンドウの設定 (1280x800, リサイズ可能, ダークテーマ)
    window = webview.create_window(
        title="LiveChat AI Studio - Realtime Face Swap Control Panel",
        url=server_url,
        width=1340,
        height=820,
        resizable=True,
        min_size=(1000, 650),
        background_color='#0f1117'
    )

    # アプリ起動
    webview.start(debug=False)

if __name__ == "__main__":
    main()
