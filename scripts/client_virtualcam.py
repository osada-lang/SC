import os
import sys
import argparse
import time
import threading
import cv2
import numpy as np
import websocket

# 仮想カメラ対応 (pyvirtualcam または OBS Virtual Camera)
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

def run_streaming_and_vcam(server_url: str, camera_index: int, enable_vcam: bool):
    global is_running, display_frame, latest_fps, latest_latency, show_original

    print(f"[LiveChat] カメラ (Index: {camera_index}) を起動中...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[LiveChat] カメラインデックス {i} で起動しました。")
                break
        else:
            print("エラー: カメラを起動できませんでした。")
            is_running = False
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 仮想カメラデバイスの初期化
    vcam = None
    if enable_vcam and HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print("==================================================")
            print(f"  [仮想カメラ起動成功] デバイス名: {vcam.device}")
            print("  Stripchat / OBS のカメラ設定で上記デバイスを選択してください！")
            print("==================================================")
        except Exception as e:
            print(f"[仮想カメラ警告] 仮想カメラの作成に失敗: {e}")

    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/stream"
    print(f"[LiveChat] クラウド接続先: {ws_url}")

    frame_count = 0
    last_fps_time = time.time()

    def on_message(ws, message):
        global display_frame, latest_fps, frame_count, last_fps_time
        try:
            arr = np.frombuffer(message, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                display_frame = img
                if vcam is not None:
                    out = cv2.resize(img, (1280, 720))
                    vcam.send(out)
                    vcam.sleep_until_next_frame()

                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    latest_fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now
        except Exception:
            pass

    def on_open(ws):
        def send_loop():
            while is_running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if show_original:
                    if vcam is not None:
                        out = cv2.resize(frame, (1280, 720))
                        vcam.send(out)
                        vcam.sleep_until_next_frame()
                    time.sleep(0.03)
                    continue

                # 1280x720 ➔ 854x480 送信用スケーリング
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (854, int(854 * h / w)))
                _, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                try:
                    ws.send(enc.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    break
                time.sleep(0.033) # 30 FPS 送信
        threading.Thread(target=send_loop, daemon=True).start()

    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
    ws.run_forever()

    cap.release()
    if vcam is not None:
        vcam.close()

def main():
    parser = argparse.ArgumentParser(description="LiveChat 仮想カメラスイート")
    parser.add_argument("--server", default="http://54.249.95.222:8080", help="サーバーURL")
    parser.add_argument("--camera", type=int, default=0, help="カメラID")
    parser.add_argument("--no-vcam", action="store_true", help="仮想カメラ出力を無効化")
    args = parser.parse_args()

    run_streaming_and_vcam(args.server, args.camera, not args.no_vcam)

if __name__ == "__main__":
    main()
