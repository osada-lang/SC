import os
import sys
import time
import json
import asyncio
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

# AI推論の高速化設定
ort.set_default_logger_severity(3)
available_providers = ort.get_available_providers()
providers = []
if 'CoreMLExecutionProvider' in available_providers:
    providers.append('CoreMLExecutionProvider')
if 'CUDAExecutionProvider' in available_providers:
    providers.append('CUDAExecutionProvider')
providers.append('CPUExecutionProvider')

print(f"[サーバー] ONNXプロバイダー: {providers}")

pcs = set()
relay = MediaRelay()

class FaceSwapServerTrack(VideoStreamTrack):
    """
    受信したビデオフレームをデコードし、非同期でAI顔交換を実行するWebRTCトラック
    """
    def __init__(self, track, model_path, parser_path, target_face):
        super().__init__()
        self.track = track
        self.target_face = target_face
        
        # 1. 顔検出モデルの初期化 (認識漏れを防ぐため標準設定に戻します)
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 2. 顔交換モデル (Inswapper) の初期化
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.swapper = insightface.model_zoo.get_model(
            model_path, download=False, download_zip=False,
            providers=providers, sess_options=sess_opts
        )
        
        # 3. 顔分割マスクモデル (BiSeNet/FaceParser) の初期化
        self.parser_session = ort.InferenceSession(parser_path, providers=providers)
        self.parser_input_name = self.parser_session.get_inputs()[0].name
        
        self.last_time = time.time()
        print("[サーバー] ビデオ処理トラックが初期化されました。")

    async def recv(self):
        # 1. クライアントから送信された映像フレームを受信 (非同期)
        frame = await self.track.recv()
        
        # 2. 【超重要】通信キューに溜まっている「古い未処理フレーム」をすべて読み飛ばして捨てる。
        # クライアントは30fpsで送信しますが、サーバーのAI処理速度がそれより遅い場合、古いフレームが
        # キューに数秒〜数十秒分も雪だるま式に溜まってしまい、これが巨大な遅延の原因になります。
        # 常に最新の1フレームだけをAI処理に回すことで、遅延を「AI計算時間（0.2秒以下）」へ一気に圧縮します。
        while True:
            try:
                # 待ち時間ほぼ0で次のフレームを試しに取得し、取得できたら古いframeを上書きして捨てる
                next_frame = await asyncio.wait_for(self.track.recv(), timeout=0.0001)
                frame = next_frame
            except asyncio.TimeoutError:
                # キューの最新に追いついたらループを終了
                break
                
        img = frame.to_ndarray(format="bgr24")
        
        # 重い同期AI計算（顔交換など）を別スレッド（Executor）で実行。
        # これにより WebRTC の内部イベントループがフリーズするのを防ぎ、通信のデッドロックを完全に回避します。
        loop = asyncio.get_event_loop()
        processed_img = await loop.run_in_executor(
            None,
            self.process_frame_sync,
            img
        )
        
        # 処理済みフレームをWebRTCの出力として返す
        new_frame = frame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def process_frame_sync(self, img):
        """
        別スレッドで実行される同期的な重いAI処理（顔検出・顔交換・BiSeNetマスク・シャープネス）
        """
        try:
            start_time = time.time()
            
            # 1. 配信者の顔を検出
            faces = self.app.get(img)
            swapped_img = img.copy()
            
            if len(faces) > 0:
                for face in faces:
                    # A. 顔交換を実行
                    swapped_img = self.swapper.get(swapped_img, face, self.target_face, paste_back=True)
                    
                    # B. BiSeNet（Face Parser）による高精度な顔・手マスク
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
                    
                    if (x2 - x1) > 0 and (y2 - y1) > 0:
                        face_roi = img[y1:y2, x1:x2]
                        swapped_roi = swapped_img[y1:y2, x1:x2]
                        
                        h_roi, w_roi = face_roi.shape[:2]
                        
                        # 前処理
                        resized_roi = cv2.resize(face_roi, (512, 512))
                        rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB).astype(np.float32)
                        
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        norm_roi = (rgb_roi / 255.0 - mean) / std
                        input_data = np.transpose(norm_roi, (2, 0, 1))
                        input_data = np.expand_dims(input_data, axis=0)
                        
                        # 推論
                        parser_out = self.parser_session.run(None, {self.parser_input_name: input_data})[0]
                        parsing_map = np.squeeze(np.argmax(parser_out, axis=1))
                        
                        # マスク生成 (顔以外のクラスを白 255 と判定)
                        face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                        hand_mask = np.zeros_like(parsing_map, dtype=np.uint8)
                        hand_mask[~np.isin(parsing_map, face_classes)] = 255
                        
                        hand_mask_resized = cv2.resize(hand_mask, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        hand_mask_blurred = cv2.GaussianBlur(hand_mask_resized, (15, 15), 0)
                        
                        alpha_hand = cv2.merge([hand_mask_blurred, hand_mask_blurred, hand_mask_blurred]) / 255.0
                        
                        # C. アンシャープマスクによる高画質化（エッジ強調）
                        gaussian = cv2.GaussianBlur(swapped_roi, (0, 0), 1.5)
                        sharp_roi = cv2.addWeighted(swapped_roi, 1.6, gaussian, -0.6, 0)
                        
                        # D. 合成
                        final_roi = (face_roi * alpha_hand + sharp_roi * (1.0 - alpha_hand)).astype(np.uint8)
                        
                        swapped_img[y1:y2, x1:x2] = final_roi
    
            # 処理速度の計測
            now = time.time()
            latency = (now - start_time) * 1000
            fps = 1.0 / (now - self.last_time) if (now - self.last_time) > 0 else 0
            self.last_time = now
            
            # 画面にパフォーマンス情報を描画
            cv2.putText(swapped_img, f"AI Server Latency: {latency:.1f}ms ({fps:.1f} FPS)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(swapped_img, f"Detected Faces: {len(faces)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return swapped_img
        except Exception as e:
            print(f"\n[AI処理エラー] 例外が発生しました: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return img

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[サーバー] 接続状態変化: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            print("[サーバー] ピア接続をクローズしました。")

    @pc.on("track")
    def on_track(track):
        print(f"[サーバー] ビデオトラックを受信しました: {track.kind}")
        if track.kind == "video":
            video_sender = pc.addTrack(
                FaceSwapServerTrack(
                    relay.subscribe(track),
                    model_path=request.app["inswapper_path"],
                    parser_path=request.app["parser_path"],
                    target_face=request.app["target_face"]
                )
            )

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    cv2.destroyAllWindows()
    print("[サーバー] 正常にシャットダウンしました。")

def main():
    inswapper_path = os.path.join("models", "inswapper_128.onnx")
    parser_path = os.path.join("models", "face_parser.onnx")
    target_image_path = "target.jpg"

    for path in [inswapper_path, parser_path, target_image_path]:
        if not os.path.exists(path):
            print(f"エラー: 必要なファイルが見つかりません: {path}")
            sys.exit(1)

    print("[サーバー] AIモデルの読み込み準備中...")
    # 顔検出用の一時的なFaceAnalysisでターゲット画像を処理
    init_app = FaceAnalysis(name='buffalo_l')
    init_app.prepare(ctx_id=0, det_size=(640, 640))
    target_img = cv2.imread(target_image_path)
    target_faces = init_app.get(target_img)
    if len(target_faces) == 0:
        print("エラー: target.jpg から顔が検出されませんでした。")
        sys.exit(1)
    target_face = target_faces[0]

    app = web.Application()
    app["inswapper_path"] = inswapper_path
    app["parser_path"] = parser_path
    app["target_face"] = target_face
    
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    print("--------------------------------------------------")
    print("[サーバー型PoC] ローカル中継サーバーを起動します。")
    print("  - ポート: 8080 で待機中。")
    print("  - クライアントからの接続確立後、顔交換ウィンドウが自動で開きます。")
    print("--------------------------------------------------")

    web.run_app(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main()
