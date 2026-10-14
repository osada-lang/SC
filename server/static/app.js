let pc = null;
let localStream = null;
let statsWs = null;
let streamWs = null;
let isStreaming = false;
let isShowingLocal = false;

// DOM要素
const remoteVideo = document.getElementById('remoteVideo');
const remoteCanvas = document.getElementById('remoteCanvas');
const localVideo = document.getElementById('localVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const viewModeBadge = document.getElementById('viewModeBadge');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnToggleSource = document.getElementById('btnToggleSource');
const cameraSelect = document.getElementById('cameraSelect');

const canvasCtx = remoteCanvas ? remoteCanvas.getContext('2d') : null;
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

const connectionStatus = document.getElementById('connectionStatus');
const statLatency = document.getElementById('statLatency');
const statFPS = document.getElementById('statFPS');
const statFaces = document.getElementById('statFaces');

const targetGallery = document.getElementById('targetGallery');
const fileInput = document.getElementById('fileInput');
const btnUploadTrigger = document.getElementById('btnUploadTrigger');

const sliderMixRatio = document.getElementById('sliderMixRatio');
const mixRatioVal = document.getElementById('mixRatioVal');
const toggleGFPGAN = document.getElementById('toggleGFPGAN');
const toggleFaceParser = document.getElementById('toggleFaceParser');
const sliderSharpness = document.getElementById('sliderSharpness');
const sharpnessVal = document.getElementById('sharpnessVal');
const sliderSkinBlend = document.getElementById('sliderSkinBlend');
const skinBlendVal = document.getElementById('skinBlendVal');

// --- ターゲット画像管理 ---
async function fetchTargets() {
  try {
    const res = await fetch('/api/targets');
    const data = await res.json();
    renderGallery(data.targets, data.current_target_id);
  } catch (err) {
    console.error('ターゲット取得エラー:', err);
  }
}

function renderGallery(targets, currentId) {
  targetGallery.innerHTML = '';
  targets.forEach(t => {
    const div = document.createElement('div');
    div.className = `target-thumb ${t.id === currentId ? 'active' : ''}`;
    div.innerHTML = `
      <img src="${t.url}" alt="${t.id}">
      ${t.id === currentId ? '<span class="target-badge">選択中</span>' : ''}
    `;
    div.addEventListener('click', () => selectTarget(t.id));
    targetGallery.appendChild(div);
  });
}

async function selectTarget(targetId) {
  try {
    await fetch('/api/targets/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: targetId })
    });
    fetchTargets();
  } catch (err) {
    console.error('ターゲット選択エラー:', err);
  }
}

btnUploadTrigger.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  btnUploadTrigger.disabled = true;
  btnUploadTrigger.textContent = '解析・アップロード中...';

  try {
    const res = await fetch('/api/targets/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (res.ok) {
      await fetchTargets();
    } else {
      alert(`アップロード失敗: ${result.detail || '顔が検出されませんでした'}`);
    }
  } catch (err) {
    alert('アップロード中に通信エラーが発生しました');
  } finally {
    btnUploadTrigger.disabled = false;
    btnUploadTrigger.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
      </svg>
      新しい見本画像をアップロード
    `;
    fileInput.value = '';
  }
});

// --- パラメータ制御 ---
function updateMixRatioLabel(val) {
  const targetPct = val;
  const sourcePct = 100 - val;
  mixRatioVal.textContent = `見本 ${targetPct}% : 本人 ${sourcePct}%`;
}

async function pushSettings() {
  const mixRatio = parseInt(sliderMixRatio.value, 10) / 100.0;
  const enableGFPGAN = toggleGFPGAN.checked;
  const enableFaceParser = toggleFaceParser.checked;
  const sharpness = parseInt(sliderSharpness.value, 10) / 10.0;
  const skinBlend = parseInt(sliderSkinBlend.value, 10) / 100.0;

  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mix_ratio: mixRatio,
        enable_gfpgan: enableGFPGAN,
        enable_face_parser: enableFaceParser,
        sharpness: sharpness,
        skin_blend: skinBlend
      })
    });
  } catch (err) {
    console.error('設定更新エラー:', err);
  }
}

sliderMixRatio.addEventListener('input', (e) => {
  updateMixRatioLabel(e.target.value);
  pushSettings();
});

toggleGFPGAN.addEventListener('change', pushSettings);
toggleFaceParser.addEventListener('change', pushSettings);

sliderSharpness.addEventListener('input', (e) => {
  sharpnessVal.textContent = `${(e.target.value / 10).toFixed(1)}x`;
  pushSettings();
});

sliderSkinBlend.addEventListener('input', (e) => {
  skinBlendVal.textContent = `${e.target.value}%`;
  pushSettings();
});

// --- カメラデバイスの検出・列挙 ---
async function populateCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    cameraSelect.innerHTML = '<option value="">カメラ検出非対応ブラウザ</option>';
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');

    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = '<option value="">カメラが見つかりません</option>';
      return;
    }

    const currentVal = cameraSelect.value;
    cameraSelect.innerHTML = '';
    
    videoDevices.forEach((dev, index) => {
      const option = document.createElement('option');
      option.value = dev.deviceId;
      option.textContent = dev.label || `カメラ ${index + 1} (${dev.deviceId.slice(0, 5)}...)`;
      cameraSelect.appendChild(option);
    });

    if (currentVal && Array.from(cameraSelect.options).some(o => o.value === currentVal)) {
      cameraSelect.value = currentVal;
    }
  } catch (err) {
    console.warn('カメラ一覧取得エラー:', err);
    cameraSelect.innerHTML = '<option value="">デフォルトカメラ</option>';
  }
}

// 段階的解像度フォールバックによるカメラ取得
async function getCameraStream(deviceId) {
  const constraintsList = [
    // 1. 指定デバイス 720p
    {
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    },
    // 2. 指定デバイス 480p
    {
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    },
    // 3. 指定デバイス 任意解像度
    {
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined
      },
      audio: false
    },
    // 4. デフォルトカメラ (完全フォールバック)
    {
      video: true,
      audio: false
    }
  ];

  let lastError = null;
  for (const constraints of constraintsList) {
    try {
      console.log('[カメラ起動試行]', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      return stream;
    } catch (err) {
      console.warn('[カメラ試行失敗]', constraints, err);
      lastError = err;
    }
  }

  throw lastError || new Error('利用可能なカメラが見つかりませんでした');
}

// --- リアルタイム・ストリーミング (WebSocket & WebRTC ハイブリッド) ---
async function startStream() {
  btnStart.disabled = true;
  btnStart.textContent = '接続中...';
  setConnectionStatus('connecting', '接続中...');

  try {
    const selectedDeviceId = cameraSelect.value || undefined;
    
    // 1. 段階的フォールバックでカメラ起動
    localStream = await getCameraStream(selectedDeviceId);
    localVideo.srcObject = localStream;
    await localVideo.play().catch(e => console.warn('localVideo play error:', e));
    await populateCameraDevices();

    isStreaming = true;
    btnStart.textContent = '配信中';
    btnStop.disabled = false;
    setConnectionStatus('connected', 'オンライン (接続完了)');

    // 2. WebSocket 高速ストリームの開始
    startWebSocketStream();

    // 3. 統計WebSocket開始
    connectStatsWebSocket();

    // 4. WebRTC接続も並行試行
    tryStartWebRTC();
  } catch (err) {
    console.error('接続失敗:', err);
    alert(`接続に失敗しました: ${err.message}\nブラウザのカメラ使用許可設定を確認してください。`);
    stopStream();
  }
}

function startWebSocketStream() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  streamWs = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
  streamWs.binaryType = 'arraybuffer';

  let waitingForResponse = false;
  let lastSendTime = Date.now();
  let wsFailed = false;

  streamWs.onopen = () => {
    console.log('[WebSocket Stream] 接続確立 - 映像送信ループ開始');
    videoPlaceholder.classList.add('hidden');
    const checkReady = () => {
      if (!isStreaming) return;
      if (localVideo.videoWidth > 0 && localVideo.videoHeight > 0) {
        sendNextFrame();
      } else {
        setTimeout(checkReady, 50);
      }
    };
    checkReady();
  };

  streamWs.onmessage = async (event) => {
    waitingForResponse = false;
    if (!isStreaming) return;

    try {
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const imgBitmap = await createImageBitmap(blob);
      
      if (remoteCanvas.width !== imgBitmap.width || remoteCanvas.height !== imgBitmap.height) {
        remoteCanvas.width = imgBitmap.width;
        remoteCanvas.height = imgBitmap.height;
      }
      canvasCtx.drawImage(imgBitmap, 0, 0);
      videoPlaceholder.classList.add('hidden');
    } catch (e) {
      console.warn('フレーム描画エラー:', e);
    }

    if (isStreaming) {
      requestAnimationFrame(sendNextFrame);
    }
  };

  streamWs.onerror = (e) => {
    console.warn('[WebSocket Stream] エラー発生 ➔ HTTP直接ストリームへフォールバックします');
    if (!wsFailed) {
      wsFailed = true;
      startHttpStreamLoop();
    }
  };

  streamWs.onclose = () => {
    console.log('[WebSocket Stream] 切断 ➔ HTTP直接ストリームへフォールバックします');
    if (!wsFailed) {
      wsFailed = true;
      startHttpStreamLoop();
    }
  };

  function sendNextFrame() {
    if (!isStreaming || !localVideo.videoWidth || wsFailed) return;

    if (waitingForResponse && (Date.now() - lastSendTime < 500)) {
      return;
    }

    captureCanvas.width = localVideo.videoWidth;
    captureCanvas.height = localVideo.videoHeight;
    captureCtx.drawImage(localVideo, 0, 0);

    waitingForResponse = true;
    lastSendTime = Date.now();

    captureCanvas.toBlob((blob) => {
      if (blob && streamWs && streamWs.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(buf => streamWs.send(buf));
      } else {
        waitingForResponse = false;
      }
    }, 'image/jpeg', 0.75);
  }
}

// プロキシ・ファイアウォール完全回避のHTTP POSTストリーミング
let isHttpLoopRunning = false;
async function startHttpStreamLoop() {
  if (isHttpLoopRunning || !isStreaming) return;
  isHttpLoopRunning = true;
  console.log('[HTTP Stream] 安定直接ストリーミングループ開始');
  videoPlaceholder.classList.add('hidden');

  let inFlight = false;
  let frameCount = 0;
  let lastFpsCalc = Date.now();

  const loop = async () => {
    if (!isStreaming) {
      isHttpLoopRunning = false;
      return;
    }

    if (localVideo.videoWidth > 0 && !inFlight) {
      captureCanvas.width = localVideo.videoWidth;
      captureCanvas.height = localVideo.videoHeight;
      captureCtx.drawImage(localVideo, 0, 0);

      inFlight = true;
      captureCanvas.toBlob(async (blob) => {
        if (!blob || !isStreaming) {
          inFlight = false;
          return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const startTime = performance.now();
        try {
          const res = await fetch('/api/process_frame', {
            method: 'POST',
            body: formData
          });

          if (res.ok) {
            const respBlob = await res.blob();
            const imgBitmap = await createImageBitmap(respBlob);
            if (remoteCanvas.width !== imgBitmap.width || remoteCanvas.height !== imgBitmap.height) {
              remoteCanvas.width = imgBitmap.width;
              remoteCanvas.height = imgBitmap.height;
            }
            canvasCtx.drawImage(imgBitmap, 0, 0);

            // FPS・遅延の計算とUI更新
            const latency = Math.round(performance.now() - startTime);
            statLatency.textContent = `${latency} ms`;
            frameCount++;
            const now = Date.now();
            if (now - lastFpsCalc >= 1000) {
              statFPS.textContent = `${Math.round(frameCount * 1000 / (now - lastFpsCalc))}`;
              frameCount = 0;
              lastFpsCalc = now;
            }
          }
        } catch (err) {
          console.warn('[HTTP Stream] 通信エラー:', err);
        } finally {
          inFlight = false;
          if (isStreaming) {
            requestAnimationFrame(loop);
          }
        }
      }, 'image/jpeg', 0.75);
    } else {
      if (isStreaming) {
        requestAnimationFrame(loop);
      }
    }
  };

  requestAnimationFrame(loop);
}

async function tryStartWebRTC() {
  try {
    const config = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
    pc = new RTCPeerConnection(config);

    pc.ontrack = (event) => {
      if (event.track.kind === 'video') {
        const stream = (event.streams && event.streams[0]) ? event.streams[0] : new MediaStream([event.track]);
        remoteVideo.srcObject = stream;
        remoteVideo.play().catch(() => {});
      }
    };

    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
    });

    if (response.ok) {
      const answer = await response.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
    }
  } catch (e) {
    console.log('[WebRTC] オプション接続スキップ (WebSocketで継続):', e.message);
  }
}

function stopStream() {
  isStreaming = false;
  if (streamWs) {
    streamWs.close();
    streamWs = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  if (statsWs) {
    statsWs.close();
    statsWs = null;
  }

  remoteVideo.srcObject = null;
  localVideo.srcObject = null;
  if (canvasCtx) {
    canvasCtx.clearRect(0, 0, remoteCanvas.width, remoteCanvas.height);
  }
  videoPlaceholder.classList.remove('hidden');

  btnStart.disabled = false;
  btnStart.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    配信・プレビュー開始
  `;
  btnStop.disabled = true;
  setConnectionStatus('disconnected', '未接続');

  statLatency.textContent = '-- ms';
  statFPS.textContent = '--';
  statFaces.textContent = '0';
}

function setConnectionStatus(state, text) {
  connectionStatus.className = `connection-status ${state}`;
  connectionStatus.querySelector('.status-text').textContent = text;
}

// 統計WebSocket
function connectStatsWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  statsWs = new WebSocket(`${protocol}//${window.location.host}/ws/stats`);

  statsWs.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.stats) {
        statLatency.textContent = `${data.stats.latency_ms} ms`;
        statFPS.textContent = `${data.stats.fps}`;
        statFaces.textContent = `${data.stats.faces_detected}`;
      }
    } catch (e) {}
  };
}

// 元映像トグル
btnToggleSource.addEventListener('click', () => {
  isShowingLocal = !isShowingLocal;
  if (isShowingLocal) {
    localVideo.classList.remove('hidden');
    remoteVideo.classList.add('hidden');
    if (remoteCanvas) remoteCanvas.classList.add('hidden');
    viewModeBadge.textContent = 'カメラ元映像 (オリジナル)';
    btnToggleSource.textContent = 'AI変換後に戻す';
  } else {
    localVideo.classList.add('hidden');
    remoteVideo.classList.remove('hidden');
    if (remoteCanvas) remoteCanvas.classList.remove('hidden');
    viewModeBadge.textContent = 'AI 変換後プレビュー';
    btnToggleSource.textContent = '元映像と比較';
  }
});

btnStart.addEventListener('click', startStream);
btnStop.addEventListener('click', stopStream);

// 初期化
fetchTargets();
updateMixRatioLabel(sliderMixRatio.value);
populateCameraDevices();

// デバイスの接続/切断イベントを検知して更新
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', populateCameraDevices);
}
