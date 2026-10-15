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
# ArcFace 基準ランドマーク (Inswapper 128px & GFPGAN 512px)
# =====================================================================
ARCFACE_DST_128 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

ARCFACE_DST_512 = ARCFACE_DST_128 * 4.0

# 楕円マスクテンプレート (128x128 & 512x512)
MASK_128 = np.zeros((128, 128), dtype=np.float32)
cv2.ellipse(MASK_128, (64, 68), (48, 60), 0, 0, 360, 1.0, -1)
MASK_128 = cv2.GaussianBlur(MASK_128, (19, 19), 0)
MASK_128 = np.expand_dims(MASK_128, axis=-1)

MASK_512 = np.zeros((512, 512), dtype=np.float32)
cv2.ellipse(MASK_512, (256, 272), (192, 240), 0, 0, 360, 1.0, -1)
MASK_512 = cv2.GaussianBlur(MASK_512, (35, 35), 0)
MASK_512 = np.expand_dims(MASK_512, axis=-1)


# =====================================================================
# 1. SLERP (球面線形補間: 歪みのない真の顔モーフィング)
# =====================================================================
def slerp_vectors(v0: np.ndarray, v1: np.ndarray, t: float, dot_threshold=0.9995) -> np.ndarray:
    """
    512次元潜在球面上を滑らかに回転補間するSLERP数学モデル
    """
    v0_norm = v0 / np.linalg.norm(v0)
    v1_norm = v1 / np.linalg.norm(v1)
    
    dot = np.dot(v0_norm, v1_norm)
    if dot > dot_threshold:
        res = v0_norm + t * (v1_norm - v0_norm)
        return res / np.linalg.norm(res)
    
    dot = np.clip(dot, -1.0, 1.0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    res = (s0 * v0_norm) + (s1 * v1_norm)
    return res / np.linalg.norm(res)


# =====================================================================
# 2. LAB色空間カラー＆ライティング自動適合 (Reinhard Color Transfer)
# =====================================================================
def apply_lab_color_transfer(source_img: np.ndarray, target_ref: np.ndarray) -> np.ndarray:
    """
    ターゲット顔の色味・明度を、部屋の照明（元のカメラ映像）に100%自動同調させて貼り付け感を消滅させる
    """
    try:
        src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(target_ref, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1)) + 1e-6
        ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1)) + 1e-6

        # 色調のスケーリング
        matched_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
        matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return source_img


# =====================================================================
# 3. アゴ周り・口周りのシャドウ＆髭ノイズ除去 (Stubble Cleanser)
# =====================================================================
def cleanse_shadow_and_stubble(crop_img: np.ndarray) -> np.ndarray:
    """
    Inswapper入力前にアゴ周りの暗い影や髭の黒ずみテクスチャを滑らかにクレンジングする
    """
    try:
        h, w = crop_img.shape[:2]
        # アゴ・口周り領域 (下部 45%)
        jaw_y = int(h * 0.55)
        jaw_roi = crop_img[jaw_y:h, :]

        # 影・テクスチャの平滑化 (エッジ保持バイラテラル)
        cleansed_jaw = cv2.bilateralFilter(jaw_roi, d=9, sigmaColor=75, sigmaSpace=75)
        
        # なだらかなグラデーションで合成
        alpha = np.linspace(0.0, 0.7, h - jaw_y)[:, np.newaxis, np.newaxis]
        crop_img[jaw_y:h, :] = (jaw_roi * (1.0 - alpha) + cleansed_jaw * alpha).astype(np.uint8)
        return crop_img
    except Exception:
        return crop_img


# =====================================================================
# 統合AI顔交換パイプライン (カメレオンAI 完全準拠版)
# =====================================================================
class AIFaceSwapPipeline:
    def __init__(self, models_dir="models", default_target_path=None):
        self.models_dir = models_dir
        self.providers = self._get_optimal_providers()
        print(f"[AI Pipeline] 使用プロバイダー: {self.providers}")

        # リアルタイム調整パラメータ
        self.mix_ratio = 0.70          # 0.0(本人100%) 〜 1.0(見本100%) (SLERP制御)
        self.enable_face_parser = False # 手・異物の透過防止マスク
        self.enable_gfpgan = True      # GFPGAN超解像 (512px高精細)
        self.gfpgan_weight = 0.85      # GFPGAN復元強度 (0.0〜1.0)
        self.sharpness = 1.3           # アンシャープマスク強度
        self.skin_blend = 0.15         # 元の肌の馴染ませ率
        self.beauty_smooth = 0.60      # 美肌スムージング (陶器肌)
        self.beauty_whitening = 0.20   # トーンアップ美白
        self.enable_color_match = True # 照明・色調マッチング

        # 時間的ブレ防止 (Temporal Smoothing EMA キャッシュ)
        self.prev_kps = None
        self.prev_embedding = None
        self.ema_alpha = 0.75          # 追従係数

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
        self.swapper_session = ort.InferenceSession(inswapper_path, providers=self.providers, sess_options=sess_opts)
        
        inputs = self.swapper_session.get_inputs()
        self.input_target_name = inputs[0].name
        self.input_source_name = inputs[1].name
        
        # Inswapper E-map 行列取得
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

        # 3. GFPGAN v1.4 超解像モデル (512px)
        gfpgan_path = os.path.join(models_dir, "GFPGANv1.4.onnx")
        self.gfpgan_session = None
        if os.path.exists(gfpgan_path):
            try:
                print(f"[AI Pipeline] GFPGAN v1.4 超解像モデルをロード中 ({gfpgan_path})...")
                self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=self.providers, sess_options=sess_opts)
                self.gfpgan_input_name = self.gfpgan_session.get_inputs()[0].name
                print("[AI Pipeline] GFPGAN v1.4 超解像エンジン準備完了 (512px)")
            except Exception as e:
                print(f"[AI Pipeline] GFPGAN 初期化警告: {e}")

        # 4. 顔・手セグメンテーションモデル (FaceParser)
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
        print("[AI Pipeline] パイプライン準備完了 (カメレオンAI 全機能統合モード)")

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

    def set_parameters(self, mix_ratio=None, enable_gfpgan=None, gfpgan_weight=None, enable_face_parser=None, sharpness=None, skin_blend=None, beauty_smooth=None, beauty_whitening=None):
        if mix_ratio is not None:
            self.mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        if enable_gfpgan is not None:
            self.enable_gfpgan = bool(enable_gfpgan)
        if gfpgan_weight is not None:
            self.gfpgan_weight = float(np.clip(gfpgan_weight, 0.0, 1.0))
        if enable_face_parser is not None:
            self.enable_face_parser = bool(enable_face_parser)
        if sharpness is not None:
            self.sharpness = float(np.clip(sharpness, 0.0, 3.0))
        if skin_blend is not None:
            self.skin_blend = float(np.clip(skin_blend, 0.0, 1.0))
        if beauty_smooth is not None:
            self.beauty_smooth = float(np.clip(beauty_smooth, 0.0, 1.0))
        if beauty_whitening is not None:
            self.beauty_whitening = float(np.clip(beauty_whitening, 0.0, 1.0))

    def get_status(self):
        return {
            "current_target_id": self.current_target_id,
            "available_targets": list(self.targets.keys()),
            "mix_ratio": self.mix_ratio,
            "enable_gfpgan": self.enable_gfpgan and (self.gfpgan_session is not None),
            "gfpgan_ready": (self.gfpgan_session is not None),
            "gfpgan_weight": self.gfpgan_weight,
            "enable_face_parser": self.enable_face_parser and (self.parser_session is not None),
            "sharpness": self.sharpness,
            "skin_blend": self.skin_blend,
            "beauty_smooth": self.beauty_smooth,
            "beauty_whitening": self.beauty_whitening,
            "providers": self.providers
        }

    def _enhance_face_gfpgan(self, cropped_face_512: np.ndarray) -> np.ndarray:
        if self.gfpgan_session is None:
            return cropped_face_512
        try:
            rgb = cv2.cvtColor(cropped_face_512, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb / 127.5) - 1.0
            norm = np.transpose(norm, (2, 0, 1))
            norm = np.expand_dims(norm, axis=0)

            out = self.gfpgan_session.run(None, {self.gfpgan_input_name: norm})[0]
            out_img = np.squeeze(out, axis=0)
            out_img = np.transpose(out_img, (1, 2, 0))
            out_img = np.clip((out_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            if self.gfpgan_weight < 1.0:
                enhanced = cv2.addWeighted(cropped_face_512, 1.0 - self.gfpgan_weight, enhanced, self.gfpgan_weight, 0)
            return enhanced
        except Exception as e:
            return cropped_face_512

    def _swap_face_direct(self, img: np.ndarray, face, latent_embedding: np.ndarray) -> np.ndarray:
        """
        全処理統合エンジン:
        1. ランドマーク平滑化 (EMA)
        2. アゴ・口周りシャドウクレンジング
        3. Inswapper 128px
        4. LAB照明・カラーマッチング
        5. GFPGAN 512px 超解像
        6. 美肌・陶器肌フィルター
        7. 手・指オクルージョン保護
        """
        # A. ランドマークの時間的平滑化 (EMA: ジッター・微細な震えの除去)
        kps = face.kps
        if self.prev_kps is not None and np.linalg.norm(self.prev_kps - kps) < 30.0:
            kps = self.prev_kps * (1.0 - self.ema_alpha) + kps * self.ema_alpha
        self.prev_kps = kps

        # 1. 相似変換行列 (128px)
        M128, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_128)
        if M128 is None:
            return img

        # 2. 128x128 クロップ & アゴ周りシャドウクレンジング (無精髭誤認防止)
        crop_img_128 = cv2.warpAffine(img, M128, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        cleansed_crop_128 = cleanse_shadow_and_stubble(crop_img_128.copy())
        
        # 3. Inswapper 推論
        crop_rgb = cv2.cvtColor(cleansed_crop_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(crop_rgb, (2, 0, 1))[np.newaxis, ...]

        source_emb = latent_embedding.reshape(1, -1)
        if self.emap is not None:
            source_emb = np.dot(source_emb, self.emap)
            source_emb /= np.linalg.norm(source_emb)

        pred = self.swapper_session.run(None, {
            self.input_target_name: blob,
            self.input_source_name: source_emb.astype(np.float32)
        })[0]

        pred_img = np.squeeze(pred, axis=0)
        pred_img = np.transpose(pred_img, (1, 2, 0))
        pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

        # 4. LAB照明・カラーマッチング (部屋の照明に完璧に同調)
        if self.enable_color_match:
            pred_bgr = apply_lab_color_transfer(pred_bgr, crop_img_128)

        # 5. GFPGAN 512px 超解像 (目元・瞳・肌の復元)
        use_512 = (self.enable_gfpgan and self.gfpgan_session is not None)
        if use_512:
            pred_512 = cv2.resize(pred_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
            pred_bgr = self._enhance_face_gfpgan(pred_512)
            M_final, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_512)
            mask_template = MASK_512
        else:
            M_final = M128
            mask_template = MASK_128

        if M_final is None:
            return img

        # 6. 美肌スムージング (陶器肌) & トーンアップ美白
        if self.beauty_smooth > 0:
            smoothed = cv2.bilateralFilter(pred_bgr, d=7, sigmaColor=40, sigmaSpace=40)
            pred_bgr = cv2.addWeighted(pred_bgr, 1.0 - self.beauty_smooth * 0.7, smoothed, self.beauty_smooth * 0.7, 0)

        if self.beauty_whitening > 0:
            hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + self.beauty_whitening * 0.18), 0, 255)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - self.beauty_whitening * 0.12), 0, 255)
            pred_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 7. シャープネス強調
        if self.sharpness > 1.0:
            gaussian = cv2.GaussianBlur(pred_bgr, (0, 0), 1.5)
            pred_bgr = cv2.addWeighted(pred_bgr, self.sharpness, gaussian, 1.0 - self.sharpness, 0)

        # 8. 逆アフィン変換で元画像にスムーズに貼り戻し
        IM = cv2.invertAffineTransform(M_final)
        h, w = img.shape[:2]

        warped_swapped = cv2.warpAffine(pred_bgr, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        warped_mask = cv2.warpAffine(mask_template, IM, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = np.expand_dims(warped_mask, axis=-1)

        # 9. 手・異物のオクルージョン保護 (ピース・手の透け防止)
        bbox = face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            face_roi = img[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, np.array([0, 25, 50], dtype=np.uint8), np.array([25, 170, 255], dtype=np.uint8))
            
            hand_edges = cv2.bitwise_and(skin_mask, edges)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            hand_mask = cv2.morphologyEx(hand_edges, cv2.MORPH_CLOSE, kernel)
            hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
            hand_mask_blur = cv2.GaussianBlur(hand_mask, (11, 11), 0)
            
            full_hand_mask = np.zeros((h, w, 1), dtype=np.float32)
            full_hand_mask[y1:y2, x1:x2, 0] = (hand_mask_blur / 255.0) * 0.85
            warped_mask = np.clip(warped_mask * (1.0 - full_hand_mask), 0.0, 1.0)

        result = (img.astype(np.float32) * (1.0 - warped_mask) + warped_swapped.astype(np.float32) * warped_mask).astype(np.uint8)
        return result

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        start_time = time.time()
        
        if self.current_target_face is None or frame is None:
            return frame, {"fps": 0, "latency_ms": 0, "faces_detected": 0}

        try:
            faces = self.app.get(frame)
            swapped_frame = frame.copy()
            
            if len(faces) > 0:
                for face in faces:
                    target_emb = self.current_target_face.normed_embedding
                    source_emb = face.normed_embedding
                    
                    # 1. SLERP による真の顔モーフィング (本人 ➔ 見本)
                    blended_emb = slerp_vectors(source_emb, target_emb, self.mix_ratio)

                    # 2. 全処理統合エンジンによる顔交換
                    swapped_frame = self._swap_face_direct(swapped_frame, face, blended_emb)

        except Exception as e:
            print(f"[AI Pipeline エラー] {e}")
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
