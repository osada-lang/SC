let localStream = null;
let statsWs = null;
let isStreaming = false;
let isShowingLocal = false;

// DOM要素
const remoteCanvas = document.getElementById('remoteCanvas');
const localVideo = document.getElementById('localVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const viewModeBadge = document.getElementById('viewModeBadge');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnToggleSource = document.getElementById('btnToggleSource');
const btnShutdown = document.getElementById('btnShutdown');
const cameraSelect = document.getElementById('cameraSelect');
const audioSelect = document.getElementById('audioSelect');

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
const sliderGFPGANWeight = document.getElementById('sliderGFPGANWeight');
const gfpganWeightVal = document.getElementById('gfpganWeightVal');
const toggleFaceParser = document.getElementById('toggleFaceParser');
const toggleExcludeMouth = document.getElementById('toggleExcludeMouth');
const sliderBeautySmooth = document.getElementById('sliderBeautySmooth');
const beautySmoothVal = document.getElementById('beautySmoothVal');
const sliderBeautyWhitening = document.getElementById('sliderBeautyWhitening');
const beautyWhiteningVal = document.getElementById('beautyWhiteningVal');
const sliderSharpness = document.getElementById('sliderSharpness');
const sharpnessVal = document.getElementById('sharpnessVal');
const sliderSkinBlend = document.getElementById('sliderSkinBlend');
const skinBlendVal = document.getElementById('skinBlendVal');

const canvasCtx = remoteCanvas.getContext('2d');
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

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
    const res = await fetch('/api/targets/upload', { method: 'POST', body: formData });
    const result = await res.json();
    if (res.ok) {
      await fetchTargets();
    } else {
      alert(`アップロード失敗: ${result.detail || '顔が検出されませんでした'}`);
    }
  } catch (err) {
    alert('通信エラーが発生しました');
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
  mixRatioVal.textContent = `見本 ${val}% : 本人 ${100 - val}%`;
}

async function pushSettings() {
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mix_ratio: parseInt(sliderMixRatio.value, 10) / 100.0,
        enable_gfpgan: toggleGFPGAN.checked,
        gfpgan_weight: parseInt(sliderGFPGANWeight.value, 10) / 100.0,
        enable_face_parser: toggleFaceParser.checked,
        exclude_mouth: toggleExcludeMouth ? toggleExcludeMouth.checked : true,
        sharpness: parseInt(sliderSharpness.value, 10) / 10.0,
        skin_blend: parseInt(sliderSkinBlend.value, 10) / 100.0,
        beauty_smooth: parseInt(sliderBeautySmooth.value, 10) / 100.0,
        beauty_whitening: parseInt(sliderBeautyWhitening.value, 10) / 100.0
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
sliderGFPGANWeight.addEventListener('input', (e) => {
  gfpganWeightVal.textContent = `${e.target.value}%`;
  pushSettings();
});
toggleFaceParser.addEventListener('change', pushSettings);
if (toggleExcludeMouth) toggleExcludeMouth.addEventListener('change', pushSettings);
sliderBeautySmooth.addEventListener('input', (e) => {
  beautySmoothVal.textContent = `${e.target.value}%`;
  pushSettings();
});
sliderBeautyWhitening.addEventListener('input', (e) => {
  beautyWhiteningVal.textContent = `${e.target.value}%`;
  pushSettings();
});
sliderSharpness.addEventListener('input', (e) => {
  sharpnessVal.textContent = `${(e.target.value / 10).toFixed(1)}x`;
  pushSettings();
});
sliderSkinBlend.addEventListener('input', (e) => {
  skinBlendVal.textContent = `${e.target.value}%`;
  pushSettings();
});

// --- カメラ＆マイクデバイスの検出 ---
async function populateCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    
    // カメラ
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    if (videoDevices.length > 0) {
      const currentCam = cameraSelect.value;
      cameraSelect.innerHTML = '';
      videoDevices.forEach((dev, index) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `カメラ ${index + 1}`;
        cameraSelect.appendChild(opt);
      });
      if (currentCam && Array.from(cameraSelect.options).some(o => o.value === currentCam)) {
        cameraSelect.value = currentCam;
      }
    }

    // マイク
    if (audioSelect) {
      const audioDevices = devices.filter(d => d.kind === 'audioinput');
      if (audioDevices.length > 0) {
        const currentMic = audioSelect.value;
        audioSelect.innerHTML = '';
        audioDevices.forEach((dev, index) => {
          const opt = document.createElement('option');
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `マイク ${index + 1}`;
          audioSelect.appendChild(opt);
        });
        if (currentMic && Array.from(audioSelect.options).some(o => o.value === currentMic)) {
          audioSelect.value = currentMic;
        }
      }
    }
  } catch (e) {}
}

async function getCameraStream(deviceId) {
  let lastError = null;

  // 1. 指定デバイスがあれば ideal で試す
  if (deviceId && deviceId.trim() !== '') {
    try {
      console.log('[カメラ] 指定デバイスで試行:', deviceId);
      return await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { ideal: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
    } catch (e) {
      console.warn('[カメラ] 指定デバイス試行失敗:', e);
      lastError = e;
    }
  }

  // 2. 最もシンプルな標準カメラ要求 (ブラウザの許可プロンプトを最優先トリガー)
  const attempts = [
    { video: true, audio: false },
    { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }
  ];

  for (const constraints of attempts) {
    try {
      console.log('[カメラ] 標準制約で試行:', constraints);
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      console.warn('[カメラ] 制約試行失敗:', constraints, e);
      lastError = e;
    }
  }

  const errDetail = lastError ? `${lastError.name}: ${lastError.message}` : 'カメラが見つかりません';
  throw new Error(`利用可能なカメラが見つかりませんでした (${errDetail})。\nブラウザのカメラ許可設定を確認してください。`);
}

// --- リアルタイム ストリーミング (高速直接パイプライン) ---
let streamWs = null;

async function startStream() {
  btnStart.disabled = true;
  btnStart.textContent = '接続中...';
  setConnectionStatus('connecting', '接続中...');

  try {
    localStream = await getCameraStream(cameraSelect.value || undefined);
    localVideo.srcObject = localStream;
    await localVideo.play();
    await populateCameraDevices();

    isStreaming = true;
    btnStart.textContent = '配信中';
    btnStop.disabled = false;
    setConnectionStatus('connected', 'オンライン (RTX 4090)');
    videoPlaceholder.classList.add('hidden');

    startFastWebSocketStream();
    connectStatsWebSocket();
  } catch (err) {
    console.error('起動エラー:', err);
    alert(`接続失敗: ${err.message}`);
    stopStream();
  }
}

function startFastWebSocketStream() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  streamWs = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
  streamWs.binaryType = 'arraybuffer';

  let inFlightCount = 0;
  const maxInFlight = 2; // パイプライン非同期送信
  let frameCount = 0;
  let lastFpsCalc = Date.now();
  let sendTimer = null;
  const sendTimestamps = [];

  streamWs.onopen = () => {
    console.log('[Fast WS Stream] 常時接続確立 - 1280x720 HD送信開始');
    
    // 30 FPS タイマーでフレームを連続プッシュ
    sendTimer = setInterval(() => {
      if (!isStreaming || !localVideo.videoWidth) return;
      if (inFlightCount >= maxInFlight) return; // ネットワーク渋滞時は間引き

      // 高精細 1280x720 HD画質で送信 (背景・服もクリアに維持)
      const targetW = 1280;
      const targetH = Math.round(targetW * (localVideo.videoHeight / localVideo.videoWidth));
      captureCanvas.width = targetW;
      captureCanvas.height = targetH;
      captureCtx.drawImage(localVideo, 0, 0, targetW, targetH);

      const sendStart = performance.now();
      inFlightCount++;
      sendTimestamps.push(sendStart);

      captureCanvas.toBlob((blob) => {
        if (blob && streamWs && streamWs.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then(buf => streamWs.send(buf));
        } else {
          inFlightCount = Math.max(0, inFlightCount - 1);
          sendTimestamps.shift();
        }
      }, 'image/jpeg', 0.82);
    }, 33);
  };

  streamWs.onmessage = async (event) => {
    inFlightCount = Math.max(0, inFlightCount - 1);
    const sentTime = sendTimestamps.shift();
    if (sentTime) {
      const lat = Math.round(performance.now() - sentTime);
      statLatency.textContent = `${lat} ms`;
    }

    if (!isStreaming) return;

    try {
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const imgBitmap = await createImageBitmap(blob);
      if (remoteCanvas.width !== imgBitmap.width || remoteCanvas.height !== imgBitmap.height) {
        remoteCanvas.width = imgBitmap.width;
        remoteCanvas.height = imgBitmap.height;
      }
      canvasCtx.drawImage(imgBitmap, 0, 0);

      frameCount++;
      const now = Date.now();
      if (now - lastFpsCalc >= 1000) {
        statFPS.textContent = `${Math.round(frameCount * 1000 / (now - lastFpsCalc))}`;
        frameCount = 0;
        lastFpsCalc = now;
      }
    } catch (e) {
      console.warn('フレーム描画エラー:', e);
    }
  };

  streamWs.onerror = () => {
    console.warn('[Fast WS Stream] WebSocketエラー ➔ HTTPパイプラインへ自動切替');
    if (sendTimer) clearInterval(sendTimer);
    startHttpFallbackLoop();
  };

  streamWs.onclose = () => {
    if (sendTimer) clearInterval(sendTimer);
    if (isStreaming) startHttpFallbackLoop();
  };
}

function startHttpFallbackLoop() {
  let inFlight = false;
  let frameCount = 0;
  let lastFpsCalc = Date.now();

  const loop = async () => {
    if (!isStreaming) return;

    if (localVideo.videoWidth > 0 && !inFlight) {
      const targetW = 640;
      const targetH = Math.round(targetW * (localVideo.videoHeight / localVideo.videoWidth));
      captureCanvas.width = targetW;
      captureCanvas.height = targetH;
      captureCtx.drawImage(localVideo, 0, 0, targetW, targetH);

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
          const res = await fetch('/api/process_frame', { method: 'POST', body: formData });
          if (res.ok) {
            const respBlob = await res.blob();
            const imgBitmap = await createImageBitmap(respBlob);
            if (remoteCanvas.width !== imgBitmap.width || remoteCanvas.height !== imgBitmap.height) {
              remoteCanvas.width = imgBitmap.width;
              remoteCanvas.height = imgBitmap.height;
            }
            canvasCtx.drawImage(imgBitmap, 0, 0);

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
        } catch (e) {
          console.warn('フレーム通信エラー:', e);
        } finally {
          inFlight = false;
          if (isStreaming) requestAnimationFrame(loop);
        }
      }, 'image/jpeg', 0.70);
    } else {
      if (isStreaming) requestAnimationFrame(loop);
    }
  };

  requestAnimationFrame(loop);
}

function stopStream() {
  isStreaming = false;
  if (streamWs) {
    streamWs.close();
    streamWs = null;
  }
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  if (statsWs) {
    statsWs.close();
    statsWs = null;
  }

  localVideo.srcObject = null;
  canvasCtx.clearRect(0, 0, remoteCanvas.width, remoteCanvas.height);
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

function connectStatsWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  statsWs = new WebSocket(`${protocol}//${window.location.host}/ws/stats`);
  statsWs.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.stats) statFaces.textContent = `${data.stats.faces_detected}`;
    } catch (e) {}
  };
}

// 元映像トグル
btnToggleSource.addEventListener('click', () => {
  isShowingLocal = !isShowingLocal;
  if (isShowingLocal) {
    localVideo.classList.remove('hidden');
    remoteCanvas.classList.add('hidden');
    viewModeBadge.textContent = 'カメラ元映像 (オリジナル)';
    btnToggleSource.textContent = 'AI変換後に戻す';
  } else {
    localVideo.classList.add('hidden');
    remoteCanvas.classList.remove('hidden');
    viewModeBadge.textContent = 'AI 変換後プレビュー';
    btnToggleSource.textContent = '元映像と比較';
  }
});

btnStart.addEventListener('click', startStream);
btnStop.addEventListener('click', stopStream);

if (btnShutdown) {
  btnShutdown.addEventListener('click', async () => {
    if (!confirm('配信を終了し、サーバーを停止して課金を止めますか？\n（再度使う時はAWS管理画面から起動できます）')) return;
    
    btnShutdown.disabled = true;
    btnShutdown.textContent = '停止中...';
    try {
      await fetch('/api/shutdown_server', { method: 'POST' });
    } catch (e) {}
    alert('サーバーの停止処理を開始しました。\n課金は数秒以内に停止します。アプリを終了します。');
    window.close();
  });
}

// 初期化
fetchTargets();
updateMixRatioLabel(sliderMixRatio.value);
populateCameraDevices();
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', populateCameraDevices);
}
