import os
import sys
import time
import copy
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

# =====================================================================
# ArcFace 128x128 基準ランドマーク (Inswapper標準)
# =====================================================================
ARCFACE_DST_128 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# マスク用楕円テンプレート (128x128)
MASK_128 = np.zeros((128, 128), dtype=np.float32)
cv2.ellipse(MASK_128, (64, 68), (48, 60), 0, 0, 360, 1.0, -1)
MASK_128 = cv2.GaussianBlur(MASK_128, (19, 19), 0)
MASK_128 = np.expand_dims(MASK_128, axis=-1)


class AIFaceSwapPipeline:
    """
    リアルタイム顔交換・顔ブレンド・手マスク・高画質化を司る統合AIパイプライン (Direct ONNX 完全独立版)
    """
    def __init__(self, models_dir="models", default_target_path=None):
        self.models_dir = models_dir
        self.providers = self._get_optimal_providers()
        print(f"[AI Pipeline] 使用プロバイダー: {self.providers}")

        # リアルタイム調整パラメータ
        self.mix_ratio = 0.70          # 0.0(本人100%) 〜 1.0(見本100%)
        self.enable_face_parser = False # 手・異物の透過防止マスク
        self.enable_gfpgan = False     # GFPGAN超解像
        self.sharpness = 1.4           # アンシャープマスク強度
        self.skin_blend = 0.15         # 元の肌の馴染ませ率

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 1. 顔検出モデル (FaceAnalysis)
        print("[AI Pipeline] FaceAnalysis を初期化中...")
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # 2. Inswapper 128 (Direct ONNX セッション)
        inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
        if not os.path.exists(inswapper_path):
            raise FileNotFoundError(f"モデルが見つかりません: {inswapper_path}")
        
        print("[AI Pipeline] Inswapper ONNX セッションを初期化中...")
        self.swapper_session = ort.InferenceSession(inswapper_path, providers=self.providers, sess_opts=sess_opts)
        
        # モデルの入力・出力名を取得
        inputs = self.swapper_session.get_inputs()
        self.input_target_name = inputs[0].name # 'target' (1, 3, 128, 128)
        self.input_source_name = inputs[1].name # 'source' (1, 512)
        
        # Inswapper内部の E-map (特徴量マッピング行列) を取得 (512x512)
        # onnxモデルのinitializerからemapを取得
        self.emap = None
        try:
            import onnx
            model_proto = onnx.load(inswapper_path)
            for init in model_proto.graph.initializer:
                if init.name == 'emap':
                    self.emap = onnx.numpy_helper.to_array(init)
                    print(f"[AI Pipeline] Inswapper emap 行列をロード成功: {self.emap.shape}")
                    break
        except Exception as e:
            print(f"[AI Pipeline] emap 自動抽出警告: {e}")

        # 3. 顔・手セグメンテーションモデル (FaceParser)
        parser_path = os.path.join(models_dir, "face_parser.onnx")
        self.parser_session = None
        if os.path.exists(parser_path):
            try:
                print("[AI Pipeline] FaceParser を初期化中...")
                self.parser_session = ort.InferenceSession(parser_path, providers=self.providers, sess_options=sess_opts)
                self.parser_input_name = self.parser_session.get_inputs()[0].name
            except Exception as e:
                print(f"[AI Pipeline] FaceParser ロード警告: {e}")

        # ターゲット顔のキャッシュ管理
        self.targets = {}
        self.current_target_id = None
        self.current_target_face = None

        if default_target_path and os.path.exists(default_target_path):
            self.set_target_from_file("default", default_target_path)

        self.last_frame_time = time.time()
        print("[AI Pipeline] パイプライン準備完了 (Direct ONNX 稼働)")

    def _get_optimal_providers(self):
        available = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    def set_target_from_file(self, target_id: str, image_path: str) -> bool:
        """ターゲット画像を読み込み、顔特徴量を抽出して登録"""
        img = cv2.imread(image_path)
        if img is None:
            return False

        faces = self.app.get(img)
        if len(faces) == 0:
            return False

        target_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        self.targets[target_id] = {
            "image_path": image_path,
            "face": target_face,
            "embedding": target_face.normed_embedding.copy()
        }
        self.current_target_id = target_id
        self.current_target_face = target_face
        return True

    def set_parameters(self, mix_ratio=None, enable_gfpgan=None, enable_face_parser=None, sharpness=None, skin_blend=None):
        if mix_ratio is not None:
            self.mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        if enable_gfpgan is not None:
            self.enable_gfpgan = bool(enable_gfpgan)
        if enable_face_parser is not None:
            self.enable_face_parser = bool(enable_face_parser)
        if sharpness is not None:
            self.sharpness = float(np.clip(sharpness, 0.0, 3.0))
        if skin_blend is not None:
            self.skin_blend = float(np.clip(skin_blend, 0.0, 1.0))

    def get_status(self):
        return {
            "current_target_id": self.current_target_id,
            "available_targets": list(self.targets.keys()),
            "mix_ratio": self.mix_ratio,
            "enable_gfpgan": self.enable_gfpgan,
            "enable_face_parser": self.enable_face_parser and (self.parser_session is not None),
            "sharpness": self.sharpness,
            "skin_blend": self.skin_blend,
            "providers": self.providers
        }

    def _swap_face_direct(self, img: np.ndarray, face, latent_embedding: np.ndarray) -> np.ndarray:
        """
        Direct ONNX による直接顔交換処理 (外部ライブラリ依存ゼロ)
        """
        kps = face.kps
        # 1. 相似変換行列 (2x3) を計算
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_128)
        if M is None:
            return img

        # 2. 128x128 にクロップ & アライメント
        crop_img = cv2.warpAffine(img, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        
        # 3. Inswapper 用に入力テンソルを作成 (1, 3, 128, 128), range [0, 1]
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(crop_rgb, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # 4. Latent Embedding の E-map マッピング
        source_emb = latent_embedding.reshape(1, -1)
        if self.emap is not None:
            source_emb = np.dot(source_emb, self.emap)
            source_emb /= np.linalg.norm(source_emb)

        # 5. ONNX 推論実行
        pred = self.swapper_session.run(None, {
            self.input_target_name: blob,
            self.input_source_name: source_emb.astype(np.float32)
        })[0]

        # 6. 出力テンソルを BGR 画像に変換
        pred_img = np.squeeze(pred, axis=0)
        pred_img = np.transpose(pred_img, (1, 2, 0))
        pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

        # 7. スキンブレンド & シャープネス
        if self.skin_blend > 0:
            pred_bgr = cv2.addWeighted(crop_img, self.skin_blend, pred_bgr, 1.0 - self.skin_blend, 0)

        if self.sharpness > 1.0:
            gaussian = cv2.GaussianBlur(pred_bgr, (0, 0), 1.5)
            pred_bgr = cv2.addWeighted(pred_bgr, self.sharpness, gaussian, 1.0 - self.sharpness, 0)

        # 8. 逆アフィン変換で元画像にスムーズに貼り戻し (高品質CUBIC補間)
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]

        warped_swapped = cv2.warpAffine(pred_bgr, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        warped_mask = cv2.warpAffine(MASK_128, IM, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = np.expand_dims(warped_mask, axis=-1)

        # 背景や服・体は元画像 100%、顔の領域のみを滑らかに合成
        result = (img.astype(np.float32) * (1.0 - warped_mask) + warped_swapped.astype(np.float32) * warped_mask).astype(np.uint8)
        return result

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        1フレームの高速AI顔交換処理
        """
        start_time = time.time()
        
        if self.current_target_face is None or frame is None:
            return frame, {"fps": 0, "latency_ms": 0, "faces_detected": 0}

        try:
            faces = self.app.get(frame)
            swapped_frame = frame.copy()
            
            if len(faces) > 0:
                for face in faces:
                    # 1. 特徴量（512次元ベクトル）のリアルタイム線形ブレンド
                    target_emb = self.current_target_face.normed_embedding
                    source_emb = face.normed_embedding
                    
                    alpha = self.mix_ratio
                    blended_emb = (target_emb * alpha) + (source_emb * (1.0 - alpha))
                    norm = np.linalg.norm(blended_emb)
                    if norm > 0:
                        blended_emb = blended_emb / norm

                    # 2. 直接 ONNX 推論による顔交換
                    swapped_frame = self._swap_face_direct(swapped_frame, face, blended_emb)

        except Exception as e:
            print(f"[AI Pipeline エラー] {e}")
            import traceback
            traceback.print_exc()
            swapped_frame = frame
            faces = []

        now = time.time()
        latency_ms = (now - start_time) * 1000
        fps = 1.0 / (now - self.last_frame_time) if (now - self.last_frame_time) > 0 else 0
        self.last_frame_time = now

        meta = {
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
            "faces_detected": len(faces)
        }
        return swapped_frame, meta
