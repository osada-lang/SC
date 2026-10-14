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
# 純粋なNumPyによるアフィン変換・Umeyamaアルゴリズム (skimage依存の完全根絶)
# =====================================================================
def umeyama_transform(src, dst, estimate_scale=True):
    """
    5点顔ランドマークから相似変換行列 (2x3) を計算する純粋NumPy実装
    skimageのバージョン互換性問題を完全に回避します。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag([1, -1]), V))
        else:
            T[:dim, :dim] = np.dot(U, V)
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        src_var = np.var(src, axis=0).sum()
        scale = 1.0 / src_var * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T[:2, :]  # 2x3 行列を返却

# insightfaceの変換処理を純粋NumPy実装でオーバーライド
class CustomSimilarityTransform:
    def __init__(self):
        self.params = np.eye(3)

    def estimate(self, src, dst):
        M = umeyama_transform(src, dst, estimate_scale=True)
        self.params = np.vstack([M, [0, 0, 1]])
        return True

    @property
    def matrix(self):
        return self.params

# skimage.transform を自前クラスで完全置換
import sys
import types
custom_st_module = types.ModuleType("skimage.transform")
custom_st_module.SimilarityTransform = CustomSimilarityTransform
sys.modules["skimage.transform"] = custom_st_module


# =====================================================================
# 統合AI顔交換パイプライン
# =====================================================================
class AIFaceSwapPipeline:
    """
    リアルタイム顔交換・顔ブレンド・手マスク・高画質化を司る統合AIパイプライン
    """
    def __init__(self, models_dir="models", default_target_path=None):
        self.models_dir = models_dir
        self.providers = self._get_optimal_providers()
        print(f"[AI Pipeline] 使用プロバイダー: {self.providers}")

        # リアルタイム調整パラメータ
        self.mix_ratio = 0.70          # 0.0(本人100%) 〜 1.0(見本100%)
        self.enable_face_parser = True # 手・異物の透過防止マスク
        self.enable_gfpgan = False     # GFPGAN超解像 (GPU推奨)
        self.sharpness = 1.6           # アンシャープマスク強度
        self.skin_blend = 0.2          # 元の肌の馴染ませ率

        # 1. 顔検出モデル (FaceAnalysis)
        print("[AI Pipeline] FaceAnalysis を初期化中...")
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # 2. 顔交換モデル (Inswapper)
        inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
        if not os.path.exists(inswapper_path):
            raise FileNotFoundError(f"モデルが見つかりません: {inswapper_path}")
        
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print("[AI Pipeline] Inswapper を初期化中...")
        self.swapper = insightface.model_zoo.get_model(
            inswapper_path,
            download=False,
            download_zip=False,
            providers=self.providers,
            sess_options=sess_opts
        )

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

        # 4. 超解像モデル (GFPGAN)
        gfpgan_path = os.path.join(models_dir, "GFPGANv1.4.onnx")
        if not os.path.exists(gfpgan_path):
            gfpgan_path = os.path.join(models_dir, "gfpgan_1.4.onnx")
        
        self.gfpgan_session = None
        if os.path.exists(gfpgan_path):
            try:
                print(f"[AI Pipeline] GFPGAN をロード中 ({gfpgan_path})...")
                self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=self.providers, sess_options=sess_opts)
                self.gfpgan_input_name = self.gfpgan_session.get_inputs()[0].name
            except Exception as e:
                print(f"[AI Pipeline] GFPGAN 初期化警告: {e}")

        # ターゲット顔のキャッシュ管理
        self.targets = {}
        self.current_target_id = None
        self.current_target_face = None

        if default_target_path and os.path.exists(default_target_path):
            self.set_target_from_file("default", default_target_path)

        self.last_frame_time = time.time()
        print("[AI Pipeline] パイプライン準備完了")

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
        """リアルタイムパラメータの更新"""
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
        """現在のステータスを返却"""
        return {
            "current_target_id": self.current_target_id,
            "available_targets": list(self.targets.keys()),
            "mix_ratio": self.mix_ratio,
            "enable_gfpgan": self.enable_gfpgan and (self.gfpgan_session is not None),
            "enable_face_parser": self.enable_face_parser and (self.parser_session is not None),
            "sharpness": self.sharpness,
            "skin_blend": self.skin_blend,
            "providers": self.providers
        }

    def _enhance_face_gfpgan(self, cropped_face):
        """GFPGAN超解像"""
        if self.gfpgan_session is None:
            return cropped_face
        try:
            h, w = cropped_face.shape[:2]
            resized = cv2.resize(cropped_face, (512, 512))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb / 127.5) - 1.0
            norm = np.transpose(norm, (2, 0, 1))
            norm = np.expand_dims(norm, axis=0)

            out = self.gfpgan_session.run(None, {self.gfpgan_input_name: norm})[0]
            out_img = np.squeeze(out, axis=0)
            out_img = np.transpose(out_img, (1, 2, 0))
            out_img = np.clip((out_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            return cv2.resize(out_bgr, (w, h))
        except Exception:
            return cropped_face

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """1フレームの顔検出・ブレンド・顔交換・マスク・高画質化"""
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

                    mixed_target_face = copy.copy(self.current_target_face)
                    mixed_target_face.normed_embedding = blended_emb
                    mixed_target_face.embedding = blended_emb

                    # 2. Inswapper による顔交換
                    swapped_frame = self.swapper.get(swapped_frame, face, mixed_target_face, paste_back=True)

                    # 3. マスク・高画質化後処理
                    bbox = face.bbox.astype(int)
                    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                    x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                    
                    if (x2 - x1) > 0 and (y2 - y1) > 0:
                        face_roi = frame[y1:y2, x1:x2]
                        swapped_roi = swapped_frame[y1:y2, x1:x2]
                        h_roi, w_roi = face_roi.shape[:2]

                        # A. GFPGAN 超解像
                        if self.enable_gfpgan and self.gfpgan_session is not None:
                            swapped_roi = self._enhance_face_gfpgan(swapped_roi)

                        # B. 手・異物マスク (FaceParser)
                        if self.enable_face_parser and self.parser_session is not None:
                            resized_roi = cv2.resize(face_roi, (512, 512))
                            rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB).astype(np.float32)
                            norm_roi = (rgb_roi / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                            input_data = np.expand_dims(np.transpose(norm_roi, (2, 0, 1)), axis=0)

                            parser_out = self.parser_session.run(None, {self.parser_input_name: input_data})[0]
                            parsing_map = np.squeeze(np.argmax(parser_out, axis=1))

                            face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                            hand_mask = np.zeros_like(parsing_map, dtype=np.uint8)
                            hand_mask[~np.isin(parsing_map, face_classes)] = 255

                            hand_mask_resized = cv2.resize(hand_mask, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                            hand_mask_blurred = cv2.GaussianBlur(hand_mask_resized, (15, 15), 0)
                            alpha_hand = cv2.merge([hand_mask_blurred]*3) / 255.0
                        else:
                            alpha_hand = np.zeros((h_roi, w_roi, 3), dtype=np.float32)

                        # C. スキンブレンド & アンシャープマスク
                        if self.skin_blend > 0:
                            swapped_roi = cv2.addWeighted(face_roi, self.skin_blend, swapped_roi, 1.0 - self.skin_blend, 0)

                        if self.sharpness > 1.0:
                            gaussian = cv2.GaussianBlur(swapped_roi, (0, 0), 1.5)
                            swapped_roi = cv2.addWeighted(swapped_roi, self.sharpness, gaussian, 1.0 - self.sharpness, 0)

                        # D. 最終合成
                        final_roi = (face_roi * alpha_hand + swapped_roi * (1.0 - alpha_hand)).astype(np.uint8)
                        swapped_frame[y1:y2, x1:x2] = final_roi
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
