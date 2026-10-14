import os
import sys

# --- Python 3.12 / skimage 互換性完全パッチ ---
try:
    import skimage.transform
    from skimage.transform._geometric import SimilarityTransform
    skimage.transform.SimilarityTransform = SimilarityTransform
    setattr(sys.modules['skimage.transform'], 'SimilarityTransform', SimilarityTransform)
except Exception as e:
    pass

import json
import asyncio
import uuid
import shutil
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
import av

from ai_pipeline import AIFaceSwapPipeline

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TARGETS_DIR = os.path.join(BASE_DIR, "targets")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(TARGETS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# デフォルトターゲットの初期設定
default_target = os.path.join(PROJECT_ROOT, "target.jpg")
if os.path.exists(default_target) and not os.path.exists(os.path.join(TARGETS_DIR, "default.jpg")):
    shutil.copy(default_target, os.path.join(TARGETS_DIR, "default.jpg"))

# AIパイプライン初期化
pipeline = AIFaceSwapPipeline(
    models_dir=MODELS_DIR,
    default_target_path=os.path.join(TARGETS_DIR, "default.jpg") if os.path.exists(os.path.join(TARGETS_DIR, "default.jpg")) else None
)

# 保存されている既存のターゲット画像をロード
for filename in os.listdir(TARGETS_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        tid = os.path.splitext(filename)[0]
        if tid not in pipeline.targets:
            pipeline.set_target_from_file(tid, os.path.join(TARGETS_DIR, filename))

app = FastAPI(title="LiveChat AI Face Swap Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/targets", StaticFiles(directory=TARGETS_DIR), name="targets")

pcs = set()
relay = MediaRelay()
connected_websockets = set()
latest_meta = {"fps": 0, "latency_ms": 0, "faces_detected": 0}

# WebRTC STUN 設定 (NAT越え・クラウド接続用)
RTC_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
)

class FaceSwapVideoTrack(VideoStreamTrack):
    """
    WebRTC受信映像からリアルタイムでAI顔交換処理を行い、クライアントに返送するカスタムトラック
    """
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        global latest_meta
        # 1. クライアントから映像フレームを受信
        frame = await self.track.recv()

        # 2. 超低遅延化: キューに溜まった古い未処理フレームをすべてスキップして破棄
        while True:
            try:
                next_frame = await asyncio.wait_for(self.track.recv(), timeout=0.0001)
                frame = next_frame
            except asyncio.TimeoutError:
                break

        img = frame.to_ndarray(format="bgr24")

        # 3. 別スレッドでAI処理を実行（WebRTCイベントループのブロッキングを完全防止）
        loop = asyncio.get_event_loop()
        processed_img, meta = await loop.run_in_executor(
            None,
            pipeline.process_frame,
            img
        )
        latest_meta = meta

        # 4. 処理済みフレームをWebRTCの出力として返す (Safari互換タイムスタンプ)
        pts, time_base = await self.next_timestamp()
        new_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>LiveChat Face Swap Server is running.</h2><p>Static UI not found.</p>")

@app.post("/offer")
async def rtc_offer(params: dict):
    """WebRTC 接続確立用エンドポイント"""
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WebRTC] 接続状態: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        print(f"[WebRTC] トラック受信: {track.kind}")
        if track.kind == "video":
            pc.addTrack(FaceSwapVideoTrack(relay.subscribe(track)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

@app.get("/api/status")
async def api_status():
    """現在のサーバー状態・設定・統計情報を取得"""
    status = pipeline.get_status()
    status["stats"] = latest_meta
    status["active_peer_connections"] = len(pcs)
    return JSONResponse(status)

@app.post("/api/settings")
async def api_settings(settings: dict):
    """リアルタイムパラメータの更新"""
    pipeline.set_parameters(
        mix_ratio=settings.get("mix_ratio"),
        enable_gfpgan=settings.get("enable_gfpgan"),
        enable_face_parser=settings.get("enable_face_parser"),
        sharpness=settings.get("sharpness"),
        skin_blend=settings.get("skin_blend")
    )
    return JSONResponse({"status": "ok", "current": pipeline.get_status()})

@app.get("/api/targets")
async def api_get_targets():
    """登録されているターゲット画像一覧を返却"""
    targets = []
    for tid, info in pipeline.targets.items():
        targets.append({
            "id": tid,
            "url": f"/targets/{os.path.basename(info['image_path'])}",
            "is_active": (tid == pipeline.current_target_id)
        })
    return JSONResponse({"targets": targets, "current_target_id": pipeline.current_target_id})

@app.post("/api/targets/select")
async def api_select_target(data: dict):
    """アクティブなターゲット画像の切り替え"""
    target_id = data.get("target_id")
    if target_id not in pipeline.targets:
        raise HTTPException(status_code=404, detail="Target ID not found")
    
    pipeline.current_target_id = target_id
    pipeline.current_target_face = pipeline.targets[target_id]["face"]
    return JSONResponse({"status": "ok", "current_target_id": target_id})

@app.post("/api/targets/upload")
async def api_upload_target(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """新しいターゲット画像のアップロードと顔特徴量抽出"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")
    
    target_id = name.strip() if name and name.strip() else f"target_{uuid.uuid4().hex[:8]}"
    target_filename = f"{target_id}{file_ext}"
    target_path = os.path.join(TARGETS_DIR, target_filename)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    success = pipeline.set_target_from_file(target_id, target_path)
    if not success:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=400, detail="Face not detected in uploaded image")

    return JSONResponse({
        "status": "ok",
        "target_id": target_id,
        "url": f"/targets/{target_filename}"
    })

@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """リアルタイムで統計情報(FPS/Latency)をWeb UIにブロードキャスト"""
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            data = {
                "stats": latest_meta,
                "status": pipeline.get_status()
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception:
        connected_websockets.discard(websocket)

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"==================================================")
    print(f"  LiveChat AI Server 起動中: http://{host}:{port}")
    print(f"==================================================")
    uvicorn.run("app:app", host=host, port=port, reload=False)
