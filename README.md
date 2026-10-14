# LiveChat AI Studio - リアルタイム顔交換＆ブレンド配信システム

ライブ配信（Stripchat等）に特化した、**クラウドGPU対応の超低遅延リアルタイム顔交換・顔ブレンド（ミックス）システム**です。
Webブラウザから操作可能なコントロールパネル（管理画面）を備えており、リアルタイムでの顔写真切り替え、ブレンド比率（本人×見本）の調整、GFPGAN高画質化を直感的に行うことができます。

---

## 🌟 主な機能と特徴

1. **リアルタイム顔ミックス（ブレンド）機能**:
   * 見本の顔 100% の置換だけでなく、「見本 70% : 本人 30%」のようにスライダーで比率を微調整可能。
   * 本人のリアルな表情・目線・筋肉の動きを活かした、極めて自然な変装を実現。
2. **Web管理画面 (LiveChat AI Studio)**:
   * ブラウザからカメラ映像の送信＆AI変換後映像のリアルタイムプレビュー。
   * 見本顔写真のドラッグ＆ドロップ・アップロードとワンクリック切り替え。
   * GFPGAN超解像・目元高画質化スイッチ（GPU環境向け）。
   * 手や異物の透過を防ぐ AI マスク（FaceParser / BiSeNet）。
3. **クラウドGPUサーバー対応 (NVIDIA RTX 4090 / RunPod)**:
   * 1時間あたり約50円〜100円の格安GPUサーバー上で動作し、60FPS・超低遅延（< 100ms）を実現。
4. **仮想カメラ（OBS / Stripchat）連携**:
   * クラウドで変換された映像をクライアントPC上で仮想Webカメラとして出力し、配信プラットフォームへ直接流し込み可能。

---

## 📁 ディレクトリ構成

```text
LiveChat/
├── server/                    # 統合AIバックエンドサーバー
│   ├── app.py                 # FastAPI + WebRTC (aiortc) サーバー
│   ├── ai_pipeline.py         # 顔検出・ミックス・Inswapper・FaceParser・GFPGAN統合モジュール
│   ├── static/                # Web管理画面フロントエンド (HTML / JS / CSS)
│   └── targets/               # 登録されたターゲット見本画像
├── models/                    # AIモデル群 (Inswapper, FaceParser, GFPGAN)
├── scripts/                   # クライアントスクリプト群
│   ├── client_virtualcam.py   # 仮想カメラ出力対応ストリーミングクライアント
│   ├── client_sender.py       # 旧WebRTCテストクライアント
│   └── face_swap_poc.py       # ローカル単体PoCスクリプト
├── docs/
│   └── RUNPOD_GUIDE.md        # RunPodクラウドGPU構築・運用マニュアル
├── Dockerfile                 # クラウドGPU用Dockerイメージ定義
└── requirements-server.txt    # サーバー用Python依存パッケージ一覧
```

---

## 🚀 クイックスタート

### 1. サーバーの起動 (ローカルまたはクラウドGPU)

#### 必要な依存関係のインストール
```bash
pip install -r requirements-server.txt
```

#### サーバー起動コマンド
```bash
cd server
python app.py
```
起動後、ブラウザで **`http://localhost:8080`** にアクセスするとWeb管理画面が表示されます。

---

### 2. クラウドGPU（RunPod）での運用手順
詳細な手順は [docs/RUNPOD_GUIDE.md](docs/RUNPOD_GUIDE.md) をご覧ください。
* RTX 4090 インスタンスを約 $0.69/時 で起動し、ポート8080を開放して即座にWeb管理画面から利用可能です。

---

### 3. OBS / Stripchatへの仮想カメラ出力（オプション）

ブラウザ単体でのプレビューに加え、OBSや配信用ブラウザに仮想Webカメラとして映像を認識させたい場合は、クライアントPC側で以下を実行します：

```bash
# 依存パッケージ（仮想カメラ）のインストール
pip install pyvirtualcam aiohttp aiortc opencv-python av

# 仮想カメラクライアントの起動
python scripts/client_virtualcam.py --server http://<サーバーIPまたはRunPodのURL>:8080 --vcam
```
Stripchatの配信設定画面でカメラデバイスとして「LiveChat / OBS Virtual Camera」を選択するだけで配信が開始されます。
