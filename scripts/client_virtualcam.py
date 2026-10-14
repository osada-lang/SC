import os
import sys
import argparse
import asyncio
import cv2
import aiohttp
import threading
import time
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import av

# 仮想カメラ対応 (pyvirtualcam がインストールされている場合)
try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

is_running = True
display_frame = None

class CameraVideoTrack(VideoStreamTrack):
    """
    ローカルWebカメラから映像を取得しWebRTCでクラウドへ送信するトラック
    """
    def __init__(self, camera_index=0, width=1280, height=720):
        super().__init__()
        print(f"[クライアント] Webカメラ (Index: {camera_index}) を起動中...")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"[クライアント] カメラインデックス {i} で起動しました。")
                    break
            else:
                print("エラー: 有効なカメラが見つかりませんでした。")
                sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        new_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def release(self):
        self.cap.release()

async def run_webrtc_client(server_url: str):
    global is_running, display_frame
    print(f"[クライアント] サーバーへ接続準備中: {server_url}")

    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    )
    pc = RTCPeerConnection(configuration=config)
    camera_track = CameraVideoTrack()
    pc.addTrack(camera_track)

    @pc.on("track")
    def on_track(track):
        print(f"[クライアント] サーバーから処理済みビデオトラックを受信: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(receive_video_loop(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[クライアント] WebRTC接続状態: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            global is_running
            is_running = False

    # SDP Offer 作成
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # サーバーへの送信
    offer_endpoint = f"{server_url.rstrip('/')}/offer"
    print(f"[クライアント] シグナリング中: {offer_endpoint}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                offer_endpoint,
                json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                timeout=aiohttp.ClientTimeout(total=10)
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

    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)
    print("[クライアント] 双方向 WebRTC 接続が確立しました。ストリーミング開始！")

    while is_running:
        await asyncio.sleep(0.5)

    camera_track.release()
    await pc.close()

async def receive_video_loop(track):
    global is_running, display_frame
    try:
        while is_running:
            frame = await track.recv()
            display_frame = frame.to_ndarray(format="bgr24")
    except Exception as e:
        print(f"[クライアント] 受信ループ終了: {e}")
    finally:
        is_running = False

def start_webrtc_thread(server_url: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_webrtc_client(server_url))
    except Exception as e:
        print(f"[クライアント] エラー: {e}")
    finally:
        loop.close()

def main():
    global is_running, display_frame
    parser = argparse.ArgumentParser(description="LiveChat 仮想カメラストリーミングクライアント")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="AIサーバーのURL (例: https://xxxx.proxy.runpod.net または http://127.0.0.1:8080)")
    parser.add_argument("--vcam", action="store_true", help="OBS / Stripchat配信用仮想Webカメラ出力を有効化")
    args = parser.parse_args()

    print("==================================================")
    print("  LiveChat AI 仮想カメラストリーミングクライアント")
    print(f"  接続先: {args.server}")
    print(f"  仮想カメラ: {'有効' if args.vcam else '無効 (GUIプレビューのみ)'}")
    print("  終了方法: プレビューウィンドウで 'q' キーを押す")
    print("==================================================")

    # WebRTC通信スレッド開始
    webrtc_thread = threading.Thread(target=start_webrtc_thread, args=(args.server,), daemon=True)
    webrtc_thread.start()

    # 仮想カメラ出力コンテキストの準備
    vcam = None
    if args.vcam:
        if HAS_VIRTUAL_CAM:
            try:
                vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                print(f"[仮想カメラ] デバイス起動成功: {vcam.device}")
            except Exception as e:
                print(f"[仮想カメラ警告] 仮想カメラ起動失敗: {e} (プレビューのみで継続します)")
        else:
            print("[仮想カメラ警告] pyvirtualcam が未インストールです。(pip install pyvirtualcam)")

    try:
        while is_running:
            if display_frame is not None:
                # 仮想カメラへフレーム送信
                if vcam is not None:
                    # 720pにリサイズして送信
                    frame_vcam = cv2.resize(display_frame, (1280, 720))
                    vcam.send(frame_vcam)
                    vcam.sleep_until_next_frame()

                # ローカルプレビュー表示
                cv2.imshow("LiveChat AI Stream Preview (Press Q to quit)", display_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                is_running = False
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        is_running = False
    finally:
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        print("[メインスレッド] 正常終了しました。")

if __name__ == "__main__":
    main()
