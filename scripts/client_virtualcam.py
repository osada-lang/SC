import os
import sys
import argparse
import time
import threading
import queue
import cv2
import numpy as np
import requests
import websocket

# 仮想カメラ対応 (pyvirtualcam がインストールされている場合)
try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

is_running = True
display_frame = None
latest_fps = 0.0
latest_latency = 0.0
show_original = False

def capture_and_stream_thread(server_url: str, camera_index: int):
    """
    バックグラウンドスレッドでカメラキャプチャ & クラウドサーバーへ高速ストリーミング
    """
    global is_running, display_frame, latest_fps, latest_latency, show_original
    
    print(f"[クライアント] カメラ (Index: {camera_index}) を起動中...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[クライアント] カメラインデックス {i} で起動しました。")
                break
        else:
            print("エラー: カメラを起動できませんでした。")
            is_running = False
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # WebSocket URL の構築
    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/stream"
    http_fallback_url = server_url.rstrip("/") + "/api/process_frame"

    print(f"[クライアント] ストリーミング接続先: {ws_url}")

    frame_count = 0
    last_fps_time = time.time()

    while is_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if show_original:
            display_frame = frame
            time.sleep(0.03)
            continue

        # 1280x720 ➔ 854x480 送信用にスケーリング
        h, w = frame.shape[:2]
        target_w = 854
        target_h = int(target_w * (h / w))
        small_frame = cv2.resize(frame, (target_w, target_h))

        # JPEG圧縮
        _, enc_jpg = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_bytes = enc_jpg.tobytes()

        start_t = time.time()

        try:
            # HTTP POST による確実な高速フレーム処理
            res = requests.post(
                http_fallback_url,
                files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")},
                timeout=1.0
            )
            if res.status_code == 200:
                resp_arr = np.frombuffer(res.content, np.uint8)
                proc_img = cv2.imdecode(resp_arr, cv2.IMREAD_COLOR)
                if proc_img is not None:
                    display_frame = proc_img
                    latest_latency = (time.time() - start_t) * 1000

                    frame_count += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        latest_fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now
        except Exception:
            pass

    cap.release()
    print("[ストリームスレッド] 終了しました。")


def main():
    global is_running, display_frame, latest_fps, latest_latency, show_original

    parser = argparse.ArgumentParser(description="LiveChat 専用クライアント (Native Window)")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="サーバーURL (例: https://xxxx.proxy.runpod.net)")
    parser.add_argument("--camera", type=int, default=0, help="カメラインデックス (0, 1, 2...)")
    parser.add_argument("--vcam", action="store_true", help="OBS / Stripchat用 仮想カメラ出力")
    args = parser.parse_args()

    print("==================================================")
    print("  LiveChat AI Studio [Native Window Client]")
    print(f"  接続先: {args.server}")
    print("  操作方法:")
    print("    - [q] キー: 終了")
    print("    - [s] キー: 元映像 / AI変換後 切り替え")
    print("==================================================")

    # 通信スレッド起動
    stream_thread = threading.Thread(
        target=capture_and_stream_thread,
        args=(args.server, args.camera),
        daemon=True
    )
    stream_thread.start()

    # 仮想カメラ初期化
    vcam = None
    if args.vcam and HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[仮想カメラ] 有効化完了: {vcam.device}")
        except Exception as e:
            print(f"[仮想カメラ警告] {e}")

    window_name = "LiveChat AI Studio - Dedicated Preview [Press Q to quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while is_running:
            if display_frame is not None:
                view = display_frame.copy()

                # ステータス情報描画
                status_text = f"Latency: {latest_latency:.0f}ms | FPS: {latest_fps:.1f} | Mode: {'Original' if show_original else 'AI Face Swap'}"
                cv2.putText(view, status_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, view)

                if vcam is not None:
                    out_frame = cv2.resize(view, (1280, 720))
                    vcam.send(out_frame)
                    vcam.sleep_until_next_frame()

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                is_running = False
                break
            elif key == ord('s'):
                show_original = not show_original

            time.sleep(0.005)
    except KeyboardInterrupt:
        is_running = False
    finally:
        is_running = False
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        print("[メイン] 正常終了しました。")

if __name__ == "__main__":
    main()
