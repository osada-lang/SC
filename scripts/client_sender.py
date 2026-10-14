import os
import sys
import asyncio
import json
import cv2
import aiohttp
import threading
import time
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import av

# グローバル変数
is_running = True
display_frame = None

class CameraVideoTrack(VideoStreamTrack):
    """
    ローカルのWebカメラから映像を取得し、WebRTC送信するカスタムビデオトラック
    """
    def __init__(self):
        super().__init__()
        print("[クライアント] Webカメラを起動しています...")
        self.cap = cv2.VideoCapture(0)
        
        # カメラが見つからない場合、別のインデックスを探索
        if not self.cap.isOpened():
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"[クライアント] カメラインデックス {i} で起動しました。")
                    break
            else:
                print("エラー: カメラを起動できませんでした。")
                sys.exit(1)
        
        # 解像度制限 (720p)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.counter = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # 直接同期的にカメラフレームを読み込む (このトラック自体が別スレッドで駆動するため安全です)
        ret, frame = self.cap.read()
        
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        new_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        
        return new_frame

    def release(self):
        self.cap.release()

async def run_client():
    global is_running
    print("[クライアント] 接続準備中...")
    
    pc = RTCPeerConnection()
    
    camera_track = CameraVideoTrack()
    pc.addTrack(camera_track)

    # サーバーからの「処理済みビデオ」を受け取った時の処理
    @pc.on("track")
    def on_track(track):
        print(f"[クライアント] サーバーから処理済みビデオ受信: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(display_received_track(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[クライアント] 接続状態変化: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            global is_running
            is_running = False

    # SDP Offer の作成と設定
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # シグナリングサーバーへの接続オファー送信
    print("[クライアント] シグナリングサーバーへ接続オファーを送信中...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://127.0.0.1:8080/offer",
                json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type
                }
            ) as response:
                if response.status != 200:
                    print(f"エラー: サーバー応答ステータス {response.status}")
                    camera_track.release()
                    await pc.close()
                    return
                
                answer_data = await response.json()
        except Exception as e:
            print(f"エラー: サーバーへの接続に失敗しました: {e}")
            camera_track.release()
            await pc.close()
            return

    # SDP Answer の設定
    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)
    print("[クライアント] 双方向 WebRTC 接続が確立しました。")

    # 接続維持ループ (別スレッドのメインループが終了するまで待機)
    while is_running:
        await asyncio.sleep(0.5)

    print("[クライアント] 終了処理を行います。")
    camera_track.release()
    await pc.close()

async def display_received_track(track):
    """
    サーバーから処理済み映像データを受け取り、グローバルバッファを更新する
    """
    global is_running, display_frame
    try:
        while is_running:
            frame = await track.recv()
            display_frame = frame.to_ndarray(format="bgr24")
    except Exception as e:
        print(f"[クライアント] トラック受信エラー: {e}")
    finally:
        is_running = False

def start_webrtc_thread():
    """
    WebRTC処理を非同期イベントループで実行するサブスレッド
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_client())
    except Exception as e:
        print(f"[WebRTCスレッド] エラー: {e}")
    finally:
        loop.close()

def main():
    global is_running, display_frame
    print("--------------------------------------------------")
    print("[クライアント] WebRTC 映像送信スクリプトを起動します。")
    print("※ 映像プレビューを選択した状態で 'q' キーを押すと終了します。")
    print("--------------------------------------------------")
    
    # 1. WebRTC処理をバックグラウンドスレッドで開始
    webrtc_thread = threading.Thread(target=start_webrtc_thread, daemon=True)
    webrtc_thread.start()
    
    # 2. 【超重要】真のメインスレッドで OpenCV の GUI 表示ループを実行
    # これにより macOS の GUI スレッドデッドロックが100%解消されます。
    try:
        while is_running:
            if display_frame is not None:
                cv2.imshow("Cast Preview Window (Processed)", display_frame)
            
            # waitKeyで描画イベントを処理する十分な時間（10ms）を与える
            if cv2.waitKey(10) & 0xFF == ord('q'):
                is_running = False
                break
                
            time.sleep(0.01) # CPU負荷を下げるためのウェイト
    except KeyboardInterrupt:
        is_running = False

    print("[メインスレッド] 終了しています...")
    time.sleep(1.0) # バックグラウンドのクローズ処理を待つ
    cv2.destroyAllWindows()
    print("[メインスレッド] 正常終了しました。")

if __name__ == "__main__":
    main()
