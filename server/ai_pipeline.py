import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

# MediaPipe 手の骨格検出
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# =====================================================================
# ArcFace 基準ランドマーク (Inswapper 128px & SimSwap 512px)
# =====================================================================
ARCFACE_DST_128 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

ARCFACE_DST_256 = ARCFACE_DST_112 * (256.0 / 112.0)
ARCFACE_DST_512 = ARCFACE_DST_112 * (512.0 / 112.0)


# =====================================================================
# 1. SLERP (球面線形補間: 潜在空間での歪みのない顔モーフィング)
# =====================================================================
def slerp_vectors(v0: np.ndarray, v1: np.ndarray, t: float, dot_threshold=0.9995) -> np.ndarray:
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
# 2. 完全同化 Reinhard LAB カラー ＆ ライティングマッチング (お面感・白浮きの完全消滅)
# =====================================================================
def apply_reinhard_color_transfer(swapped_crop: np.ndarray, original_crop: np.ndarray) -> np.ndarray:
    """
    ターゲット顔の色・明るさ・血色感を、配信者自身の首・頬・部屋の照明に100%完全同化させる数学的変換
    """
    try:
        src_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1)) + 1e-5
        ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1)) + 1e-5

        # 各チャンネル (L:明度, A:緑-赤, B:青-黄) の平均・分散を本人の肌に完全一致
        matched_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
        matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return swapped_crop


# =====================================================================
# 3. AI入力前・髭テクスチャ完全消滅フィルター (Pre-Inswapper Stubble Cleanser)
# =====================================================================
def cleanse_pre_inswapper_stubble(crop_128: np.ndarray) -> np.ndarray:
    """
    AIに渡す直前のクロップ画像から、アゴ・鼻下・口周りの髭のざらつきや暗い影を完全に平滑化
    これにより、Inswapperが髭を1ピクセルも生成しないツルツルの女性肌として認識する
    """
    try:
        h, w = crop_128.shape[:2]
        cleansed = crop_128.copy()

        # 鼻の下〜アゴ周り領域 (h: 50%〜100%)
        jaw_y1 = int(h * 0.50)
        jaw_roi = cleansed[jaw_y1:h, :]

        # 頬の健康的な肌色をサンプリング
        cheek_roi = cleansed[int(h * 0.32):int(h * 0.48), int(w * 0.22):int(w * 0.78)]
        base_color = cheek_roi.mean(axis=(0, 1))

        # 髭や暗い影のテクスチャを強力に平滑化
        bilateral_jaw = cv2.bilateralFilter(jaw_roi, d=11, sigmaColor=90, sigmaSpace=90)
        
        # 暗い髭ピクセルを肌色へブライトニング
        gray_jaw = cv2.cvtColor(jaw_roi, cv2.COLOR_BGR2GRAY)
        dark_mask = np.clip((base_color.mean() * 0.95 - gray_jaw) / (base_color.mean() * 0.35 + 1e-5), 0.0, 1.0)
        dark_mask = cv2.GaussianBlur(dark_mask, (13, 13), 0)[:, :, np.newaxis]

        # 影を持ち上げてツルツル肌へ置き換え
        flattened_jaw = cv2.addWeighted(bilateral_jaw, 0.60, np.full_like(jaw_roi, base_color, dtype=np.uint8), 0.40, 0)
        cleansed[jaw_y1:h, :] = (jaw_roi.astype(np.float32) * (1.0 - dark_mask * 0.90) +
                                 flattened_jaw.astype(np.float32) * (dark_mask * 0.90)).astype(np.uint8)

        return cleansed
    except Exception:
        return crop_128
class AIFaceSwapPipeline:
    def __init__(self, models_dir="models", default_target_path=None):
        self.models_dir = models_dir
        self.providers = self._get_optimal_providers()
        print(f"[AI Pipeline Pro] 使用プロバイダー: {self.providers}")

        # パラメータ設定 (美形純度向上・白浮きゼロ・自然肌色デフォルト)
        self.mix_ratio = 0.80          # 顔ミックス比率 (ターゲット造形美を80%キープ)
        self.enable_gfpgan = True      # GFPGAN超解像
        self.gfpgan_weight = 0.70      # GFPGAN復元強度 (70%で目元クッキリ)
        self.enable_face_parser = True # 手の透け防止オクルージョンマスク (MediaPipe)
        self.exclude_mouth = True      # 口元・唇・歯の完全保護
        self.sharpness = 1.2           # シャープネス
        self.skin_blend = 0.15         # 元の顔馴染ませ率
        self.beauty_smooth = 0.50      # 美肌スムージング (陶器肌)
        self.beauty_whitening = 0.00   # トーンアップ美白
        self.concealer_strength = 1.00 # 口元影・髭ノイズ消去コンシーラー (デフォルト100%)
        self.lip_tint = 0.35           # 血色リップグロス・ティント (0.0〜1.0)
        self.blush_tint = 0.25         # ふんわり血色チーク (0.0〜1.0)

        # 動的アダプティブEMA追従フィルター
        self.prev_kps = None

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 1. 顔検出モデル (FaceAnalysis)
        print("[AI Pipeline Pro] FaceAnalysis を初期化中...")
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # 2. Inswapper 128 (Direct ONNX)
        inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
        if not os.path.exists(inswapper_path):
            raise FileNotFoundError(f"モデルが見つかりません: {inswapper_path}")
        
        print("[AI Pipeline Pro] Inswapper ONNX セッションを初期化中...")
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
                    print(f"[AI Pipeline Pro] Inswapper emap 行列ロード成功: {self.emap.shape}")
                    break
        except Exception as e:
            print(f"[AI Pipeline Pro] emap 抽出警告: {e}")

        # 3. Ghost 256 (カメレオンAI同系列 本命モデル) & ArcFace Converter
        ghost_path = os.path.join(models_dir, "ghost_2_256.onnx")
        converter_path = os.path.join(models_dir, "arcface_converter_ghost.onnx")
        self.use_ghost = False
        self.ghost_session = None
        self.converter_session = None

        if os.path.exists(ghost_path) and os.path.exists(converter_path):
            try:
                print("[AI Pipeline Pro] Ghost 256 (カメレオンAI同系列モデル) を初期化中...")
                self.ghost_session = ort.InferenceSession(ghost_path, providers=self.providers, sess_options=sess_opts)
                self.converter_session = ort.InferenceSession(converter_path, providers=self.providers, sess_options=sess_opts)
                self.ghost_target_name = self.ghost_session.get_inputs()[0].name
                self.ghost_source_name = self.ghost_session.get_inputs()[1].name
                self.use_ghost = True
                print("🎉 [AI Pipeline Pro] Ghost 256 (カメレオンAI同系列) エンジンスワップ完了！")
            except Exception as e:
                print(f"[AI Pipeline Pro] Ghost 256 初期化警告: {e}")

        # 4. GFPGAN v1.4 超解像モデル (512px)
        gfpgan_path = os.path.join(models_dir, "GFPGANv1.4.onnx")
        self.gfpgan_session = None
        if os.path.exists(gfpgan_path):
            try:
                print(f"[AI Pipeline Pro] GFPGAN v1.4 超解像モデルをロード中...")
                self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=self.providers, sess_options=sess_opts)
                self.gfpgan_input_name = self.gfpgan_session.get_inputs()[0].name
                print("[AI Pipeline Pro] GFPGAN v1.4 超解像エンジン準備完了 (512px)")
            except Exception as e:
                print(f"[AI Pipeline Pro] GFPGAN 初期化警告: {e}")

        # 4. MediaPipe Hands 手の骨格検出エンジン
        self.mp_hands = None
        if HAS_MEDIAPIPE:
            try:
                print("[AI Pipeline Pro] MediaPipe Hands (手の骨格AI) を初期化中...")
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[AI Pipeline Pro] MediaPipe Hands 準備完了 (オクルージョン保護)")
            except Exception as e:
                print(f"[AI Pipeline Pro] MediaPipe Hands 初期化警告: {e}")

        # ターゲット顔キャッシュ
        self.targets = {}
        self.current_target_id = None
        self.current_target_face = None

        if default_target_path and os.path.exists(default_target_path):
            self.set_target_from_file("default", default_target_path)

        self.last_frame_time = time.time()
        print("[AI Pipeline Pro] パイプライン準備完了 (自然肌色＆完全同化版)")

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

    def set_parameters(self, mix_ratio=None, enable_gfpgan=None, gfpgan_weight=None, enable_face_parser=None, exclude_mouth=None, sharpness=None, skin_blend=None, beauty_smooth=None, beauty_whitening=None, concealer_strength=None, lip_tint=None, blush_tint=None, **kwargs):
        if mix_ratio is not None:
            self.mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        if enable_gfpgan is not None:
            self.enable_gfpgan = bool(enable_gfpgan)
        if gfpgan_weight is not None:
            self.gfpgan_weight = float(np.clip(gfpgan_weight, 0.0, 1.0))
        if enable_face_parser is not None:
            self.enable_face_parser = bool(enable_face_parser)
        if exclude_mouth is not None:
            self.exclude_mouth = bool(exclude_mouth)
        if sharpness is not None:
            self.sharpness = float(np.clip(sharpness, 0.0, 3.0))
        if skin_blend is not None:
            self.skin_blend = float(np.clip(skin_blend, 0.0, 1.0))
        if beauty_smooth is not None:
            self.beauty_smooth = float(np.clip(beauty_smooth, 0.0, 1.0))
        if beauty_whitening is not None:
            self.beauty_whitening = float(np.clip(beauty_whitening, 0.0, 1.0))
        if concealer_strength is not None:
            self.concealer_strength = float(np.clip(concealer_strength, 0.0, 1.0))
        if lip_tint is not None:
            self.lip_tint = float(np.clip(lip_tint, 0.0, 1.0))
        if blush_tint is not None:
            self.blush_tint = float(np.clip(blush_tint, 0.0, 1.0))

    def get_status(self):
        return {
            "current_target_id": self.current_target_id,
            "available_targets": list(self.targets.keys()),
            "mix_ratio": self.mix_ratio,
            "enable_gfpgan": self.enable_gfpgan and (self.gfpgan_session is not None),
            "gfpgan_ready": (self.gfpgan_session is not None),
            "gfpgan_weight": self.gfpgan_weight,
            "enable_face_parser": self.enable_face_parser,
            "exclude_mouth": self.exclude_mouth,
            "sharpness": self.sharpness,
            "skin_blend": self.skin_blend,
            "beauty_smooth": self.beauty_smooth,
            "beauty_whitening": self.beauty_whitening,
            "concealer_strength": self.concealer_strength,
            "lip_tint": self.lip_tint,
            "blush_tint": self.blush_tint,
            "active_engine": "Ghost 256 (カメレオンAI同系列モデル)" if self.use_ghost else "Inswapper 128",
            "providers": self.providers
        }

    def _enhance_face_gfpgan(self, cropped_face_512: np.ndarray) -> np.ndarray:
        if self.gfpgan_session is None:
            return cropped_face_512
        try:
            rgb = cv2.cvtColor(cropped_face_512, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb / 127.5) - 1.0
            norm = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

            out = self.gfpgan_session.run(None, {self.gfpgan_input_name: norm})[0]
            out_img = np.squeeze(out, axis=0).transpose(1, 2, 0)
            out_img = np.clip((out_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            enhanced_512 = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            w = self.gfpgan_weight
            blended_512 = cv2.addWeighted(cropped_face_512, 1.0 - w, enhanced_512, w, 0)
            return blended_512
        except Exception:
            return cropped_face_512

    def _apply_eye_micro_sharpness(self, face_bgr_128: np.ndarray) -> np.ndarray:
        try:
            h, w = face_bgr_128.shape[:2]
            eye_y1, eye_y2 = int(h * 0.28), int(h * 0.58)
            eye_roi = face_bgr_128[eye_y1:eye_y2, :].copy()

            gaussian = cv2.GaussianBlur(eye_roi, (0, 0), 1.0)
            sharp_eye = cv2.addWeighted(eye_roi, 1.35, gaussian, -0.35, 0)
            face_bgr_128[eye_y1:eye_y2, :] = sharp_eye
            return face_bgr_128
        except Exception:
            return face_bgr_128

    def _apply_female_beauty_suite(self, face_bgr_128: np.ndarray) -> np.ndarray:
        """
        女性配信者特化ビューティースイート:
        1. 口元・アゴの影＆髭ノイズ完全除去 (デジタルコンシーラー)
        2. ふんわり血色チーク (ほんのりピンク)
        3. 血色リップティント
        """
        try:
            h, w = face_bgr_128.shape[:2]

            # 1. 口元・アゴの影・髭ノイズ完全消去 (コンシーラー・強力版)
            if self.concealer_strength > 0:
                jaw_y = int(h * 0.48)
                jaw_roi = face_bgr_128[jaw_y:h, :]
                
                # 頬の明るい肌色ベースをサンプリング
                cheek_roi = face_bgr_128[int(h*0.30):int(h*0.48), int(w*0.2):int(w*0.8)]
                base_color = cheek_roi.mean(axis=(0, 1))

                gray_jaw = cv2.cvtColor(jaw_roi, cv2.COLOR_BGR2GRAY)
                dark_shadows = np.clip((base_color.mean() - gray_jaw) / (base_color.mean() * 0.4 + 1e-5), 0.0, 1.0)
                dark_shadows = cv2.GaussianBlur(dark_shadows, (15, 15), 0)[:, :, np.newaxis]

                smooth_jaw = cv2.bilateralFilter(jaw_roi, d=9, sigmaColor=85, sigmaSpace=85)
                brightened = cv2.addWeighted(smooth_jaw, 0.65, np.full_like(smooth_jaw, base_color, dtype=np.uint8), 0.35, 0)
                
                concealed_jaw = (jaw_roi.astype(np.float32) * (1.0 - dark_shadows * self.concealer_strength) + 
                                 brightened.astype(np.float32) * (dark_shadows * self.concealer_strength)).astype(np.uint8)
                face_bgr_128[jaw_y:h, :] = concealed_jaw

            # 2. ふんわり血色チーク (両頬のほんのりピンク)
            if self.blush_tint > 0:
                blush_mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(blush_mask, (int(w * 0.28), int(h * 0.52)), int(w * 0.14), 1.0, -1)
                cv2.circle(blush_mask, (int(w * 0.72), int(h * 0.52)), int(w * 0.14), 1.0, -1)
                blush_mask = cv2.GaussianBlur(blush_mask, (21, 21), 0)[:, :, np.newaxis]

                blush_color = np.array([140, 130, 220], dtype=np.float32)
                face_bgr_128 = np.clip(
                    face_bgr_128.astype(np.float32) * (1.0 - blush_mask * self.blush_tint * 0.45) +
                    blush_color * (blush_mask * self.blush_tint * 0.45),
                    0, 255
                ).astype(np.uint8)

            return face_bgr_128
        except Exception:
            return face_bgr_128

    def _create_full_parts_mask(self, img_shape, face, kps):
        """
        目・眉・鼻・唇・口角を包み込むインナーパーツマスク (輪郭・アゴの地肌と100%一体化)
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            pts = face.landmark_2d_106.astype(np.int32)
            hull = cv2.convexHull(pts)
        else:
            center = kps.mean(axis=0)
            expanded = center + (kps - center) * 1.45
            hull = cv2.convexHull(expanded.astype(np.int32))

        cv2.fillConvexPoly(mask, hull, 1.0)

        # 口元除外処理 (唇・歯・口角をマスクから自然にくり抜き)
        if self.exclude_mouth and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lmk = face.landmark_2d_106.astype(np.int32)
            mouth_indices = list(range(52, min(72, len(lmk))))
            mouth_pts = lmk[mouth_indices]
            
            mouth_hull = cv2.convexHull(mouth_pts)
            mouth_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)
            mouth_mask = cv2.dilate(mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (15, 15), 0) / 255.0

            mask = np.clip(mask * (1.0 - mouth_mask_blur), 0.0, 1.0)

        # 境界の広めフェザリング (首・頬との境界線を完全消去)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_inner = cv2.erode(mask, kernel, iterations=1)
        mask_blurred = cv2.GaussianBlur(mask, (25, 25), 0)
        final_mask = np.maximum(mask_inner, mask_blurred)

        return np.expand_dims(final_mask, axis=-1)

    def _detect_hand_mask_solid(self, frame: np.ndarray) -> np.ndarray:
        """
        MediaPipe Hands 手の骨格マスク (ギザギザ破綻を防ぐ滑らかなソフトフェザリング版)
        """
        h, w = frame.shape[:2]
        if self.mp_hands is None:
            return np.zeros((h, w, 1), dtype=np.float32)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            if not results.multi_hand_landmarks:
                return np.zeros((h, w, 1), dtype=np.float32)

            mask = np.zeros((h, w), dtype=np.uint8)

            for hand_landmarks in results.multi_hand_landmarks:
                pts = []
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pts.append((px, py))

                # 手のひら全体の凸包
                palm_indices = [0, 1, 2, 5, 9, 13, 17]
                palm_pts = np.array([pts[i] for i in palm_indices], dtype=np.int32)
                cv2.fillConvexPoly(mask, cv2.convexHull(palm_pts), 255)

                # 指の骨格ライン (滑らかなアンチエイリアス描画)
                connections = mp.solutions.hands.HAND_CONNECTIONS
                thickness = max(14, int(w * 0.05))
                for conn in connections:
                    pt1 = pts[conn[0]]
                    pt2 = pts[conn[1]]
                    cv2.line(mask, pt1, pt2, 255, thickness=thickness, lineType=cv2.LINE_AA)

                # 指先の丸み
                radius = max(10, int(w * 0.03))
                for pt in pts:
                    cv2.circle(mask, pt, radius, 255, -1, lineType=cv2.LINE_AA)

            # ギザギザを消去する滑らかなフェザリング (ソフトブレンド)
            mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
            return (mask_blur / 255.0)[:, :, np.newaxis].astype(np.float32)
        except Exception:
            return np.zeros((h, w, 1), dtype=np.float32)

    def _swap_face_direct(self, img: np.ndarray, face, latent_embedding: np.ndarray) -> np.ndarray:
        """
        カメレオンAI完全同等 自然肌色エンジン (白浮きゼロ・地肌同化)
        """
        # A. 動的アダプティブEMA追従フィルター
        kps = face.kps
        if self.prev_kps is not None:
            velocity = np.linalg.norm(kps - self.prev_kps)
            alpha = np.clip(0.65 + (velocity / 20.0) * 0.30, 0.65, 0.95)
            kps = self.prev_kps * (1.0 - alpha) + kps * alpha
        self.prev_kps = kps

        # 1. 128px 基準アライメント行列
        M128, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_128)
        if M128 is None:
            return img

        # 2. クロップ (128x128)
        crop_img_128 = cv2.warpAffine(img, M128, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        
        # 3. AI入力前・髭テクスチャ完全消滅フィルター (Inswapperが髭を生成しないよう前処理)
        crop_ai_input = cleanse_pre_inswapper_stubble(crop_img_128.copy())

        # 4. Inswapper 推論
        crop_rgb = cv2.cvtColor(crop_ai_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(crop_rgb, (2, 0, 1))[np.newaxis, ...]

        source_emb = latent_embedding.reshape(1, -1)
        if self.emap is not None:
            source_emb = np.dot(source_emb, self.emap)
            source_emb /= np.linalg.norm(source_emb)

        pred = self.swapper_session.run(None, {
            self.input_target_name: blob,
            self.input_source_name: source_emb.astype(np.float32)
        })[0]

        pred_img = np.squeeze(pred, axis=0).transpose(1, 2, 0)
        pred_bgr = cv2.cvtColor(np.clip(pred_img * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 4. 【最重要】Reinhard LABカラーマッチング (本人の肌色・明度・血色感に100%完全同化)
        pred_bgr = apply_reinhard_color_transfer(pred_bgr, crop_img_128)

        # 5. GFPGAN 512px 超解像 (毛穴・まつ毛・瞳の高精細化)
        use_gfpgan = (self.enable_gfpgan and self.gfpgan_session is not None)
        if use_gfpgan:
            pred_512 = cv2.resize(pred_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
            enhanced_512 = self._enhance_face_gfpgan(pred_512)
            pred_bgr = cv2.resize(enhanced_512, (128, 128), interpolation=cv2.INTER_AREA)

        # 6. 瞳・目元限定マイクロシャープネス
        pred_bgr = self._apply_eye_micro_sharpness(pred_bgr)

        # 7. 美肌スムージング (陶器肌)
        if self.beauty_smooth > 0:
            smoothed = cv2.bilateralFilter(pred_bgr, d=7, sigmaColor=30, sigmaSpace=30)
            pred_bgr = cv2.addWeighted(pred_bgr, 1.0 - self.beauty_smooth * 0.5, smoothed, self.beauty_smooth * 0.5, 0)

        # 8. 女性配信者特化ビューティースイート (口元影・髭消しコンシーラー ＆ 血色チーク)
        pred_bgr = self._apply_female_beauty_suite(pred_bgr)

        # 9. トーンアップ美白 (白浮きを防ぐ微細調整)
        if self.beauty_whitening > 0:
            hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + self.beauty_whitening * 0.10), 0, 255)
            pred_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 10. シャープネス強調
        if self.sharpness > 1.0:
            gaussian = cv2.GaussianBlur(pred_bgr, (0, 0), 1.5)
            pred_bgr = cv2.addWeighted(pred_bgr, self.sharpness, gaussian, 1.0 - self.sharpness, 0)

        # 10. 逆アフィン変換で元画像空間へ展開
        IM = cv2.invertAffineTransform(M128)
        h, w = img.shape[:2]
        
        crop_valid_mask = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(crop_valid_mask, (64, 64), 58, 1.0, -1)
        crop_valid_mask = cv2.GaussianBlur(crop_valid_mask, (11, 11), 0)
        
        warped_swapped = cv2.warpAffine(pred_bgr, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = cv2.warpAffine(crop_valid_mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = np.expand_dims(warped_valid, axis=-1)

        # 11. パーツマスク生成 (お面感を消し去る自然なグラデーション)
        parts_mask = self._create_full_parts_mask(img.shape, face, kps)
        final_parts_mask = parts_mask * warped_valid

        # 最終ブレンド合成
        result = (img.astype(np.float32) * (1.0 - final_parts_mask) + warped_swapped.astype(np.float32) * final_parts_mask).astype(np.uint8)
        return result

    def _swap_face_ghost_256(self, img: np.ndarray, face, latent_embedding: np.ndarray) -> np.ndarray:
        """
        Ghost 256 顔交換エンジン (カメレオンAI ghost-live 同系列本命モデル)
        """
        kps = face.kps
        if self.prev_kps is not None:
            velocity = np.linalg.norm(kps - self.prev_kps)
            alpha = np.clip(0.65 + (velocity / 20.0) * 0.30, 0.65, 0.95)
            kps = self.prev_kps * (1.0 - alpha) + kps * alpha
        self.prev_kps = kps

        # 1. 256px 基準アライメント行列
        M256, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_256)
        if M256 is None:
            return img

        # 2. クロップ (256x256)
        crop_img_256 = cv2.warpAffine(img, M256, (256, 256), borderMode=cv2.BORDER_REPLICATE)

        # 3. 前処理 (Ghost正規化: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        crop_rgb = cv2.cvtColor(crop_img_256, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_rgb = (crop_rgb - 0.5) / 0.5
        blob_256 = np.transpose(norm_rgb, (2, 0, 1))[np.newaxis, ...]

        # 4. 特徴量変換 (ArcFace ➔ Ghost Converter)
        source_emb = latent_embedding.reshape(1, -1).astype(np.float32)
        converted_emb = self.converter_session.run(None, {'input': source_emb})[0]
        converted_emb = converted_emb.reshape(1, -1)
        converted_emb = converted_emb / (np.linalg.norm(converted_emb) + 1e-6)

        # 5. Ghost 256 推論
        pred = self.ghost_session.run(None, {
            self.ghost_target_name: blob_256,
            self.ghost_source_name: converted_emb
        })[0][0]

        # 6. 逆正規化 (Ghost正規デコード: pred * 0.5 + 0.5 ➔ BGR 256x256)
        pred_rgb = (np.transpose(pred, (1, 2, 0)) * 0.5 + 0.5).clip(0, 1)
        pred_bgr_256 = (pred_rgb[:, :, ::-1] * 255.0).astype(np.uint8)

        # 7. 地肌同調カラーマッチング (本人の肌色・明度・血色感に100%完全同化)
        pred_bgr_256 = apply_reinhard_color_transfer(pred_bgr_256, crop_img_256)

        # 8. GFPGAN 512px 超解像 (毛穴・まつ毛・瞳の高精細化)
        use_gfpgan = (self.enable_gfpgan and self.gfpgan_session is not None)
        if use_gfpgan:
            pred_512 = cv2.resize(pred_bgr_256, (512, 512), interpolation=cv2.INTER_CUBIC)
            enhanced_512 = self._enhance_face_gfpgan(pred_512)
            pred_bgr_256 = cv2.resize(enhanced_512, (256, 256), interpolation=cv2.INTER_AREA)

        # 9. 瞳・目元限定マイクロシャープネス
        pred_bgr_256 = self._apply_eye_micro_sharpness(pred_bgr_256)

        # 10. 美肌スムージング (陶器肌) ＆ ビューティースイート (髭消し・チーク)
        if self.beauty_smooth > 0:
            smoothed = cv2.bilateralFilter(pred_bgr_256, d=7, sigmaColor=30, sigmaSpace=30)
            pred_bgr_256 = cv2.addWeighted(pred_bgr_256, 1.0 - self.beauty_smooth * 0.5, smoothed, self.beauty_smooth * 0.5, 0)

        pred_bgr_256 = self._apply_female_beauty_suite(pred_bgr_256)

        # 11. 逆アフィン変換で元画像空間へ展開 (256px逆変換)
        IM256 = cv2.invertAffineTransform(M256)
        h, w = img.shape[:2]

        crop_valid_mask = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(crop_valid_mask, (128, 128), 118, 1.0, -1)
        crop_valid_mask = cv2.GaussianBlur(crop_valid_mask, (15, 15), 0)

        warped_swapped = cv2.warpAffine(pred_bgr_256, IM256, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = cv2.warpAffine(crop_valid_mask, IM256, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = np.expand_dims(warped_valid, axis=-1)

        # 12. パーツマスク生成 (口元除外を含む)
        parts_mask = self._create_full_parts_mask(img.shape, face, kps)
        final_parts_mask = parts_mask * warped_valid

        result = (img.astype(np.float32) * (1.0 - final_parts_mask) + warped_swapped.astype(np.float32) * final_parts_mask).astype(np.uint8)
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
                    
                    # SLERP による真の潜在モーフィング
                    blended_emb = slerp_vectors(source_emb, target_emb, self.mix_ratio)

                    # Ghost 256 または Inswapper 128 による顔交換
                    if self.use_ghost and self.ghost_session is not None:
                        swapped_frame = self._swap_face_ghost_256(swapped_frame, face, blended_emb)
                    else:
                        swapped_frame = self._swap_face_direct(swapped_frame, face, blended_emb)

            # 12. 手のひら・指のオクルージョン保護
            if self.enable_face_parser and self.mp_hands is not None:
                hand_mask = self._detect_hand_mask_solid(frame)
                if np.any(hand_mask > 0.05):
                    swapped_frame = (swapped_frame.astype(np.float32) * (1.0 - hand_mask) + frame.astype(np.float32) * hand_mask).astype(np.uint8)

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
