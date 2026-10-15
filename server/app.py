import os
import sys
import json
import asyncio
import uuid
import shutil
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
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

# デフォルトターゲット初期化
default_target = os.path.join(PROJECT_ROOT, "target.jpg")
if os.path.exists(default_target) and not os.path.exists(os.path.join(TARGETS_DIR, "default.jpg")):
    shutil.copy(default_target, os.path.join(TARGETS_DIR, "default.jpg"))

# AIパイプライン初期化
pipeline = AIFaceSwapPipeline(
    models_dir=MODELS_DIR,
    default_target_path=os.path.join(TARGETS_DIR, "default.jpg") if os.path.exists(os.path.join(TARGETS_DIR, "default.jpg")) else None
)

# 登録ターゲット画像の自動読み込み
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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/targets", StaticFiles(directory=TARGETS_DIR), name="targets")

pcs = set()
relay = MediaRelay()
latest_meta = {"fps": 0, "latency_ms": 0, "faces_detected": 0}

# WebRTC STUN 設定
RTC_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
)

class FaceSwapVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        global latest_meta
        frame = await self.track.recv()

        # 最新フレーム以外を破棄して遅延を最小化
        while True:
            try:
                next_frame = await asyncio.wait_for(self.track.recv(), timeout=0.0001)
                frame = next_frame
            except asyncio.TimeoutError:
                break

        img = frame.to_ndarray(format="bgr24")
        loop = asyncio.get_event_loop()
        processed_img, meta = await loop.run_in_executor(None, pipeline.process_frame, img)
        latest_meta = meta

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
    return HTMLResponse("<h2>LiveChat AI Server Running</h2>")

# 1. リアルタイム高速フレーム処理エンドポイント (HTTP直接ストリーム)
@app.post("/api/process_frame")
async def api_process_frame(file: UploadFile = File(...)):
    global latest_meta
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid frame")

        loop = asyncio.get_event_loop()
        processed_img, meta = await loop.run_in_executor(None, pipeline.process_frame, img)
        latest_meta = meta

        _, enc_img = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=enc_img.tobytes(), media_type="image/jpeg")
    except Exception as e:
        print(f"[API Process Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 2. 超低遅延常時接続 WebSocket ストリーミング
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    global latest_meta
    await websocket.accept()
    loop = asyncio.get_event_loop()
    print("[WebSocket Stream] 常時接続ストリーム確立 (30 FPS)")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            processed_img, meta = await loop.run_in_executor(None, pipeline.process_frame, img)
            latest_meta = meta

            _, enc_img = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await websocket.send_bytes(enc_img.tobytes())
    except WebSocketDisconnect:
        print("[WebSocket Stream] 正常切断")
    except Exception as e:
        print(f"[WebSocket Stream エラー] {e}")

# 3. WebRTC シグナリング
@app.post("/offer")
async def rtc_offer(params: dict):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(FaceSwapVideoTrack(relay.subscribe(track)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

# 3. 設定 & ターゲット管理 API
@app.get("/api/status")
async def api_status():
    status = pipeline.get_status()
    status["stats"] = latest_meta
    return JSONResponse(status)

@app.post("/api/settings")
async def api_settings(settings: dict):
    pipeline.set_parameters(
        mix_ratio=settings.get("mix_ratio"),
        enable_gfpgan=settings.get("enable_gfpgan"),
        enable_face_parser=settings.get("enable_face_parser"),
        sharpness=settings.get("sharpness"),
        skin_blend=settings.get("skin_blend"),
        beauty_smooth=settings.get("beauty_smooth"),
        beauty_whitening=settings.get("beauty_whitening")
    )
    return JSONResponse({"status": "ok", "current": pipeline.get_status()})

@app.get("/api/targets")
async def api_get_targets():
    targets = [
        {"id": tid, "url": f"/targets/{os.path.basename(info['image_path'])}", "is_active": (tid == pipeline.current_target_id)}
        for tid, info in pipeline.targets.items()
    ]
    return JSONResponse({"targets": targets, "current_target_id": pipeline.current_target_id})

@app.post("/api/targets/select")
async def api_select_target(data: dict):
    target_id = data.get("target_id")
    if target_id not in pipeline.targets:
        raise HTTPException(status_code=404, detail="Target ID not found")
    pipeline.current_target_id = target_id
    pipeline.current_target_face = pipeline.targets[target_id]["face"]
    return JSONResponse({"status": "ok", "current_target_id": target_id})

@app.post("/api/targets/upload")
async def api_upload_target(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")
    
    tid = name.strip() if name and name.strip() else f"target_{uuid.uuid4().hex[:8]}"
    target_path = os.path.join(TARGETS_DIR, f"{tid}{ext}")

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not pipeline.set_target_from_file(tid, target_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=400, detail="Face not detected")

    return JSONResponse({"status": "ok", "target_id": tid, "url": f"/targets/{tid}{ext}"})

@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"stats": latest_meta, "status": pipeline.get_status()})
            await asyncio.sleep(0.2)
    except Exception:
        pass

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
    print(f"  LiveChat AI Server Running: http://{host}:{port}")
    print(f"==================================================")
    uvicorn.run("app:app", host=host, port=port, reload=False)
