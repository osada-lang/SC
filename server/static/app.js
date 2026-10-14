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
const cameraSelect = document.getElementById('cameraSelect');

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
        enable_face_parser: toggleFaceParser.checked,
        sharpness: parseInt(sliderSharpness.value, 10) / 10.0,
        skin_blend: parseInt(sliderSkinBlend.value, 10) / 100.0
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

// --- カメラデバイスの検出 ---
async function populateCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    if (videoDevices.length === 0) return;

    const currentVal = cameraSelect.value;
    cameraSelect.innerHTML = '';
    videoDevices.forEach((dev, index) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `カメラ ${index + 1}`;
      cameraSelect.appendChild(opt);
    });
    if (currentVal && Array.from(cameraSelect.options).some(o => o.value === currentVal)) {
      cameraSelect.value = currentVal;
    }
  } catch (e) {}
}

async function getCameraStream(deviceId) {
  const constraints = [
    { video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    { video: true, audio: false }
  ];
  for (const c of constraints) {
    try {
      return await navigator.mediaDevices.getUserMedia(c);
    } catch (e) {}
  }
  throw new Error('利用可能なカメラが見つかりませんでした');
}

// --- リアルタイム ストリーミング (高速直接パイプライン) ---
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

    startStreamingLoop();
    connectStatsWebSocket();
  } catch (err) {
    console.error('起動エラー:', err);
    alert(`接続失敗: ${err.message}`);
    stopStream();
  }
}

function startStreamingLoop() {
  let inFlight = false;
  let frameCount = 0;
  let lastFpsCalc = Date.now();

  const loop = async () => {
    if (!isStreaming) return;

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
          const res = await fetch('/api/process_frame', { method: 'POST', body: formData });
          if (res.ok) {
            const respBlob = await res.blob();
            const imgBitmap = await createImageBitmap(respBlob);
            if (remoteCanvas.width !== imgBitmap.width || remoteCanvas.height !== imgBitmap.height) {
              remoteCanvas.width = imgBitmap.width;
              remoteCanvas.height = imgBitmap.height;
            }
            canvasCtx.drawImage(imgBitmap, 0, 0);

            // FPS・遅延の計算
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
      }, 'image/jpeg', 0.80);
    } else {
      if (isStreaming) requestAnimationFrame(loop);
    }
  };

  requestAnimationFrame(loop);
}

function stopStream() {
  isStreaming = false;
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

// 初期化
fetchTargets();
updateMixRatioLabel(sliderMixRatio.value);
populateCameraDevices();
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', populateCameraDevices);
}
