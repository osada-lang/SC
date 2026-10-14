# RunPod クラウドGPU サーバー構築 ＆ 運用ガイド

本手順書は、格安クラウドGPUサービス「RunPod」を利用して、NVIDIA RTX 4090 サーバー上で本リアルタイム顔交換システムを最安・最高パフォーマンス（30〜60fps・超低遅延）で起動・検証するためのマニュアルです。

---

## 1. 費用の目安
* **推奨GPU**: NVIDIA RTX 4090 (24GB VRAM) または RTX 3090
* **利用料金**: **約 $0.34 〜 $0.69 / 時間 (約50円〜105円 / 時)**
* ※使った時間だけの従量課金です。テスト終了後に「Stop」すれば課金は完全に止まります。

---

## 2. RunPod インスタンス（Pod）の立ち上げ手順

1. [RunPod 公式サイト](https://www.runpod.io/) にログイン（未登録の場合はアカウント作成 ＆ $10程度チャージ）。
2. **「Pods」➔「+ Deploy」** をクリック。
3. **GPUの選択**:
   * **「RTX 4090」** (または RTX 3090) を選択。
4. **テンプレートの選択**:
   * **「RunPod PyTorch 2.1」**（または PyTorch 2.x / CUDA 12.x 搭載の公式テンプレート）を選択。
5. **ポート開放設定 (重要)**:
   * **「Customize Deployment」** をクリック。
   * **「Expose HTTP Ports」** に `8080` を追加。
6. **「Deploy On-Demand」** をクリックして起動（1〜2分で起動します）。

---

## 3. サーバーへの接続とプログラムの起動

### ステップ 1: Webターミナル（JupyterLab または Web Terminal）を開く
* 起動したPodの「Connect」➔ **「Connect to Web Terminal」**（または JupyterLab の Terminal）をクリック。

### ステップ 2: プログラムの配置とセットアップ
ターミナル上で以下のコマンドを順に実行します。

```bash
# 1. ワークスペースへ移動
cd /workspace

# 2. リポジトリをクローン（またはファイルをアップロード）
git clone <あなたのリポジトリURL> LiveChat
cd LiveChat

# 3. 依存ライブラリのインストール (CUDA GPU版 ONNX Runtime)
pip install onnxruntime-gpu insightface fastapi uvicorn[standard] aiortc opencv-python-headless av python-multipart websockets aiofiles pillow

# 4. サーバーの起動
cd server
python app.py
```

起動に成功すると、以下のように表示されます：
```text
[AI Pipeline] 使用プロバイダー: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[AI Pipeline] FaceAnalysis を初期化中...
[AI Pipeline] Inswapper を初期化中...
[AI Pipeline] パイプライン準備完了
==================================================
  LiveChat AI Server 起動中: http://0.0.0.0:8080
==================================================
```

---

## 4. Mac / クライアントからの接続と操作

1. RunPodの管理画面で、起動したPodの **「Connect」➔「HTTP Service (Port 8080)」** のURL（例: `https://xxxx-8080.proxy.runpod.net`）をクリック。
2. ブラウザ上で **LiveChat AI Studio（Web管理画面）** が開きます。
3. **「配信・プレビュー開始」** をクリックすると、Macのカメラが起動し、クラウドGPUで超リアルタイム（CUDA高速処理）に顔交換された映像がヌルヌルとプレビュー表示されます。
4. **コントロール**:
   * ターゲット画像をクリックして瞬時に切り替え
   * 新しい見本顔写真をドラッグ＆ドロップでアップロード
   * ミックス比率スライダーで「本人30% : 見本70%」など好みのリアル度合いに微調整
   * GFPGAN超解像スイッチで目元・肌を高精細化

---

## 5. テスト終了後のサーバー停止（課金ストップ）

検証が終わったら、RunPodのダッシュボードで該当Podの **「Stop (一時停止)」** ボタンを押してください。
* Stop中はGPU料金（1時間約100円）の課金が即座に停止します。
* 再度テストしたい時は **「Start」** を押せば、設定を保持したまま数秒で再開できます。
