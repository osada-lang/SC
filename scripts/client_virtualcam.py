import os
import sys
import argparse
import time
import threading
import queue
import cv2
import numpy as np
import requests

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

# スレッド間通信用キュー
send_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def network_worker(server_url: str):
    """
    バックグラウンドスレッドでクラウドGPUサーバーとフレーム送受信
    """
    global is_running, latest_fps, latest_latency

    http_url = server_url.rstrip("/") + "/api/process_frame"
    frame_count = 0
    last_fps_time = time.time()

    while is_running:
        try:
            frame = send_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # 1280x720 ➔ 854x480 送信用にスケーリング
        h, w = frame.shape[:2]
        target_w = 854
        target_h = int(target_w * (h / w))
        small_frame = cv2.resize(frame, (target_w, target_h))

        _, enc_jpg = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_bytes = enc_jpg.tobytes()

        start_t = time.time()
        try:
            res = requests.post(
                http_url,
                files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")},
                timeout=1.0
            )
            if res.status_code == 200:
                resp_arr = np.frombuffer(res.content, np.uint8)
                proc_img = cv2.imdecode(resp_arr, cv2.IMREAD_COLOR)
                if proc_img is not None:
                    latest_latency = (time.time() - start_t) * 1000

                    if result_queue.full():
                        try: result_queue.get_nowait()
                        except queue.Empty: pass
                    result_queue.put(proc_img)

                    frame_count += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        latest_fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now
        except Exception:
            pass

    print("[通信スレッド] 終了しました。")


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

    # 1. 【超重要】macOSのカメラ初期化は必ずメインスレッドで実行
    print(f"[メインスレッド] カメラ (Index: {args.camera}) を起動中...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[メインスレッド] カメラインデックス {i} で起動しました。")
                break
        else:
            print("エラー: カメラを起動できませんでした。")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. 通信ワーカースレッド起動
    worker_thread = threading.Thread(
        target=network_worker,
        args=(args.server,),
        daemon=True
    )
    worker_thread.start()

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
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # 最新フレームを送信キューへ投入
            if send_queue.full():
                try: send_queue.get_nowait()
                except queue.Empty: pass
            send_queue.put(frame)

            # 処理済みフレームの取得
            try:
                proc_frame = result_queue.get_nowait()
                display_frame = proc_frame
            except queue.Empty:
                pass

            # 画面描画
            view = frame.copy() if show_original else (display_frame if display_frame is not None else frame)
            view = view.copy()

            status_text = f"Latency: {latest_latency:.0f}ms | FPS: {latest_fps:.1f} | Mode: {'Original' if show_original else 'AI Face Swap'}"
            cv2.putText(view, status_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, view)

            if vcam is not None:
                out_frame = cv2.resize(view, (1280, 720))
                vcam.send(out_frame)
                vcam.sleep_until_next_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                is_running = False
                break
            elif key == ord('s'):
                show_original = not show_original

    except KeyboardInterrupt:
        is_running = False
    finally:
        is_running = False
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        print("[メインスレッド] 正常終了しました。")

if __name__ == "__main__":
    main()
