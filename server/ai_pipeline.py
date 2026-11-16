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
# 2. LAB色空間 カラー ＆ CLAHE局所コントラスト適応 (照明・陰影完全同期)
# =====================================================================
def apply_lab_and_clahe_matching(swapped_crop: np.ndarray, original_crop: np.ndarray) -> np.ndarray:
    try:
        src_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean = src_lab.mean(axis=(0, 1))
        ref_mean = ref_lab.mean(axis=(0, 1))

        # 色調の平均トーンを部屋の照明に同調
        matched_lab = src_lab.copy()
        matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0] - src_mean[0] + ref_mean[0], 0, 255)
        matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1] - src_mean[1] + ref_mean[1], 0, 255)
        matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2] - src_mean[2] + ref_mean[2], 0, 255)

        # CLAHE (局所適応ヒストグラム平滑化) で部屋の陰影の深さを顔に同期
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        matched_lab[:, :, 0] = clahe.apply(matched_lab[:, :, 0].astype(np.uint8)).astype(np.float32)

        return cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    except Exception:
        return swapped_crop


# =====================================================================
# 統合AI顔交換パイプライン (カメレオンAI 完全網羅プロ仕様版)
# =====================================================================
class AIFaceSwapPipeline:
    def __init__(self, models_dir="models", default_target_path=None):
        self.models_dir = models_dir
        self.providers = self._get_optimal_providers()
        print(f"[AI Pipeline Pro] 使用プロバイダー: {self.providers}")

        # パラメータ設定
        self.mix_ratio = 0.70          # 顔ミックス比率 (0.0〜1.0)
        self.enable_gfpgan = True      # GFPGAN超解像
        self.gfpgan_weight = 0.70      # GFPGAN復元強度
        self.enable_face_parser = True # 手の透け防止オクルージョンマスク (MediaPipe面結合)
        self.exclude_mouth = True      # 口元・唇・歯の完全保護
        self.sharpness = 1.3           # シャープネス
        self.skin_blend = 0.15         # 元の顔馴染ませ率
        self.beauty_smooth = 0.60      # 美肌スムージング (陶器肌)
        self.beauty_whitening = 0.20   # トーンアップ美白

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

        # 3. GFPGAN v1.4 超解像モデル (512px)
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
        print("[AI Pipeline Pro] パイプライン準備完了 (カメレオンAI 完全網羅プロ仕様)")

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

    def set_parameters(self, mix_ratio=None, enable_gfpgan=None, gfpgan_weight=None, enable_face_parser=None, exclude_mouth=None, sharpness=None, skin_blend=None, beauty_smooth=None, beauty_whitening=None, **kwargs):
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
        """
        瞳・目元限定マイクロ高周波シャープネス (目に生きた人間のツヤ・キャッチライトを与える)
        """
        try:
            h, w = face_bgr_128.shape[:2]
            eye_y1, eye_y2 = int(h * 0.28), int(h * 0.58)
            eye_roi = face_bgr_128[eye_y1:eye_y2, :].copy()

            # 高周波エッジ強調 (瞳のハイライトとまつ毛をクッキリ)
            gaussian = cv2.GaussianBlur(eye_roi, (0, 0), 1.0)
            sharp_eye = cv2.addWeighted(eye_roi, 1.45, gaussian, -0.45, 0)
            face_bgr_128[eye_y1:eye_y2, :] = sharp_eye
            return face_bgr_128
        except Exception:
            return face_bgr_128

    def _create_full_parts_mask(self, img_shape, face, kps):
        """
        目・眉・鼻・唇・口角を包み込むインナーパーツマスク
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            pts = face.landmark_2d_106.astype(np.int32)
            hull = cv2.convexHull(pts)
        else:
            center = kps.mean(axis=0)
            expanded = center + (kps - center) * 1.55
            hull = cv2.convexHull(expanded.astype(np.int32))

        cv2.fillConvexPoly(mask, hull, 1.0)

        # 口元除外処理 (口・歯・唇は本人100%キープ)
        if self.exclude_mouth and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lmk = face.landmark_2d_106.astype(np.int32)
            mouth_indices = list(range(52, min(72, len(lmk))))
            mouth_pts = lmk[mouth_indices]
            
            mouth_hull = cv2.convexHull(mouth_pts)
            mouth_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)
            mouth_mask = cv2.dilate(mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
            mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (11, 11), 0) / 255.0

            mask = np.clip(mask * (1.0 - mouth_mask_blur), 0.0, 1.0)

        # 境界フェザリング
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask_inner = cv2.erode(mask, kernel, iterations=1)
        mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
        final_mask = np.maximum(mask_inner, mask_blurred)

        return np.expand_dims(final_mask, axis=-1)

    def _detect_hand_mask_solid(self, frame: np.ndarray) -> np.ndarray:
        """
        MediaPipe骨格 ＋ 手のひら・指の肉厚完全面結合 (手の透けを物理的ゼロ化)
        """
        h, w = frame.shape[:2]
        if self.mp_hands is None:
            return np.zeros((h, w, 1), dtype=np.float32)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            if not results.multi_hand_landmarks:
                return np.zeros((h, w, 1), dtype=np.float32)

            solid_mask = np.zeros((h, w), dtype=np.uint8)

            for hand_landmarks in results.multi_hand_landmarks:
                pts = []
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pts.append([px, py])
                    # 関節・指先の球体肉厚カバー (十分な太さで保護)
                    cv2.circle(solid_mask, (px, py), int(w * 0.045), 255, -1)
                
                pts = np.array(pts, dtype=np.int32)
                # 手のひら全体の凸包ポリゴン
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(solid_mask, hull, 255)

                # 指の骨格太線 (太さ w * 0.08 で肉厚を完全カバー)
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for conn in connections:
                    pt1 = pts[conn[0]]
                    pt2 = pts[conn[1]]
                    cv2.line(solid_mask, tuple(pt1), tuple(pt2), 255, thickness=int(w * 0.08))

            # 境界の膨張と滑らかなフェザリング (透け漏れ完全カット)
            solid_mask = cv2.dilate(solid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
            solid_mask_blur = cv2.GaussianBlur(solid_mask, (11, 11), 0)
            return (solid_mask_blur / 255.0)[:, :, np.newaxis].astype(np.float32)
        except Exception:
            return np.zeros((h, w, 1), dtype=np.float32)

    def _swap_face_direct(self, img: np.ndarray, face, latent_embedding: np.ndarray) -> np.ndarray:
        """
        カメレオンAI完全網羅プロ仕様エンジン
        """
        # A. 動的アダプティブEMA追従フィルター (静止時は完全固定・動いた瞬間は遅延ゼロ追従)
        kps = face.kps
        if self.prev_kps is not None:
            velocity = np.linalg.norm(kps - self.prev_kps)
            # 移動速度に応じた適応係数 (静止時 0.65 ➔ 素早い動き時 0.95)
            alpha = np.clip(0.65 + (velocity / 20.0) * 0.30, 0.65, 0.95)
            kps = self.prev_kps * (1.0 - alpha) + kps * alpha
        self.prev_kps = kps

        # 1. 128px 基準アライメント行列
        M128, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST_128)
        if M128 is None:
            return img

        # 2. クロップ (128x128)
        crop_img_128 = cv2.warpAffine(img, M128, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        
        # 3. Inswapper 推論
        crop_rgb = cv2.cvtColor(crop_img_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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

        # 4. LAB照明・CLAHE局所コントラスト適応 (部屋の陰影の深さに完全同期)
        pred_bgr = apply_lab_and_clahe_matching(pred_bgr, crop_img_128)

        # 5. GFPGAN 512px 超解像 (毛穴・まつ毛・瞳の高精細化)
        use_gfpgan = (self.enable_gfpgan and self.gfpgan_session is not None)
        if use_gfpgan:
            pred_512 = cv2.resize(pred_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
            enhanced_512 = self._enhance_face_gfpgan(pred_512)
            pred_bgr = cv2.resize(enhanced_512, (128, 128), interpolation=cv2.INTER_AREA)

        # 6. 瞳・目元限定マイクロシャープネス (生気・ツヤ・瞳の光)
        pred_bgr = self._apply_eye_micro_sharpness(pred_bgr)

        # 7. 美肌スムージング (陶器肌) & トーンアップ美白
        if self.beauty_smooth > 0:
            smoothed = cv2.bilateralFilter(pred_bgr, d=7, sigmaColor=35, sigmaSpace=35)
            pred_bgr = cv2.addWeighted(pred_bgr, 1.0 - self.beauty_smooth * 0.6, smoothed, self.beauty_smooth * 0.6, 0)

        if self.beauty_whitening > 0:
            hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + self.beauty_whitening * 0.15), 0, 255)
            pred_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 8. シャープネス強調
        if self.sharpness > 1.0:
            gaussian = cv2.GaussianBlur(pred_bgr, (0, 0), 1.5)
            pred_bgr = cv2.addWeighted(pred_bgr, self.sharpness, gaussian, 1.0 - self.sharpness, 0)

        # 9. 逆アフィン変換で元画像空間へ展開 (境界クランプで紫色ノイズ完全ゼロ化)
        IM = cv2.invertAffineTransform(M128)
        h, w = img.shape[:2]
        
        crop_valid_mask = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(crop_valid_mask, (64, 64), 58, 1.0, -1)
        crop_valid_mask = cv2.GaussianBlur(crop_valid_mask, (11, 11), 0)
        
        warped_swapped = cv2.warpAffine(pred_bgr, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = cv2.warpAffine(crop_valid_mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = np.expand_dims(warped_valid, axis=-1)

        # 10. パーツマスク生成 (口元除外を含む)
        parts_mask = self._create_full_parts_mask(img.shape, face, kps)
        final_parts_mask = parts_mask * warped_valid

        # 最終ブレンド合成
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

                    # カメレオンAI完全網羅プロ仕様エンジンによる顔交換
                    swapped_frame = self._swap_face_direct(swapped_frame, face, blended_emb)

            # 11. 手のひら・指の肉厚完全面結合オクルージョン保護 (手の透けを物理的ゼロ化)
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
