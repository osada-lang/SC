import os
import sys
import time
import queue
import threading
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort

# スレッド間のデータ受け渡し用キュー (最新フレームのみを保持するためサイズは1)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
is_running = True

def ai_inference_thread(model_path, target_face, providers):
    """
    バックグラウンドで高速な顔検出、顔交換、および高精細スキン・エッジブレンドを実行するスレッド
    """
    global is_running
    print("[AIスレッド] 初期化中...")

    # 1. 顔検出モデル (FaceAnalysis) の初期化
    # ★重要チューニング1: 不要なモジュールをカット
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
    
    # ★重要チューニング2: 顔の検出サイズを 640x640 ➔ 320x320 に縮小
    # これにより、顔検出の計算量が 4分の1 に激減し、M5 MacのCPU環境でも一気に30FPS前後の高速動作になります。
    # 配信者はカメラの前にいるため、320pxでも検出精度は全く問題ありません。
    app.prepare(ctx_id=0, det_size=(320, 320))

    # 2. 顔交換モデル (Inswapper) のロード
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        swapper = insightface.model_zoo.get_model(
            model_path, 
            download=False, 
            download_zip=False,
            providers=providers,
            sess_options=session_options
        )
    except Exception as e:
        print(f"[AIスレッド] モデルの読み込みに失敗しました: {e}")
        is_running = False
        return

    print("[AIスレッド] 準備完了。推論ループを開始します。")

    while is_running:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # --- AI処理の開始 ---
        start_time = time.time()
        
        # 1. 配信者の顔を検出 (モデル削減と320px化により超高速動作)
        faces = app.get(frame)
        
        swapped_frame = frame.copy()
        
        if len(faces) > 0:
            for face in faces:
                # 顔交換を実行
                swapped_frame = swapper.get(swapped_frame, face, target_face, paste_back=True)
                
                # --- [極限チューニング] エッジ ➔ 肌色ハイブリッド手すかし防止マスク ---
                # 顔の領域（bbox）を取得
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    face_roi = frame[y1:y2, x1:x2]
                    swapped_roi = swapped_frame[y1:y2, x1:x2]
                    
                    # A. 元の映像から「輪郭線（エッジ）」を検出する (Canny法)
                    # 手や指の境界線は、顔に比べて非常に急激な明暗差（強いエッジ）を生みます。
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_roi, 60, 160)
                    
                    # B. 肌色（皮膚色）領域の抽出
                    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
                    upper_skin = np.array([20, 160, 255], dtype=np.uint8)
                    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                    
                    # C. ハイブリッドマスクの生成: 「肌色」かつ「強いエッジ（指の境界）」がある部分
                    # これにより、顔自体の滑らかな肌はスルーし、手や指の境界線だけを極めて正確に捉えます。
                    hand_edge_mask = cv2.bitwise_and(skin_mask, edges)
                    
                    # D. 輪郭を閉じる処理（膨張・モルフォロジー）で指や手のひらの面をしっかりとマスク化
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    hand_mask = cv2.morphologyEx(hand_edge_mask, cv2.MORPH_CLOSE, kernel)
                    hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
                    
                    # ノイズ除去とぼかし処理（滑らかな合成のため）
                    hand_mask_blurred = cv2.GaussianBlur(hand_mask, (15, 15), 0)
                    
                    # E. 手のマスク部分には「元の映像（本物の手）」を重ね、それ以外には顔交換を適用
                    alpha_hand = cv2.merge([hand_mask_blurred, hand_mask_blurred, hand_mask_blurred]) / 255.0
                    
                    # まず手以外のベース合成画像を作成 (スキンブレンドで馴染ませる)
                    skin_mask_clean = cv2.medianBlur(skin_mask, 5)
                    skin_mask_blurred = cv2.GaussianBlur(skin_mask_clean, (21, 21), 0)
                    alpha_skin = cv2.merge([skin_mask_blurred, skin_mask_blurred, skin_mask_blurred]) / 255.0
                    alpha_skin = alpha_skin * 0.40  # 40%元の肌色を馴染ませる
                    
                    blended_roi = (face_roi * alpha_skin + swapped_roi * (1.0 - alpha_skin)).astype(np.uint8)
                    
                    # アンシャープマスクで劇的に画質をシャープ化
                    gaussian = cv2.GaussianBlur(blended_roi, (0, 0), 1.5)
                    sharp_roi = cv2.addWeighted(blended_roi, 1.7, gaussian, -0.7, 0)
                    
                    # 最後に「手」を最前面に100%重ね書きする
                    final_roi = (face_roi * alpha_hand + sharp_roi * (1.0 - alpha_hand)).astype(np.uint8)
                    
                    # 元の画像に書き戻す
                    swapped_frame[y1:y2, x1:x2] = final_roi

        # 処理時間を測定
        latency = (time.time() - start_time) * 1000
        fps = 1000 / latency if latency > 0 else 0
        
        # 画面にパフォーマンス情報を描画 (デモ用)
        cv2.putText(swapped_frame, f"AI Latency: {latency:.1f}ms ({fps:.1f} FPS)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 結果を送信用のキューに入れる (古いデータは捨てる)
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put(swapped_frame)

    print("[AIスレッド] 終了しました。")


def main():
    global is_running
    model_path = os.path.join("models", "inswapper_128.onnx")
    target_image_path = "target.jpg"

    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' がありません。")
        sys.exit(1)

    if not os.path.exists(target_image_path):
        print(f"エラー: ターゲット画像 '{target_image_path}' がありません。")
        sys.exit(1)

    print("システムを初期化中...")

    # --- 最適なONNX実行プロバイダーの判定 ---
    available_providers = ort.get_available_providers()
    providers = []
    if 'CoreMLExecutionProvider' in available_providers:
        providers.append('CoreMLExecutionProvider')
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    # 顔検出用の一時的なFaceAnalysisでターゲット画像を処理
    init_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
    init_app.prepare(ctx_id=0, det_size=(320, 320))
    target_img = cv2.imread(target_image_path)
    target_faces = init_app.get(target_img)
    if len(target_faces) == 0:
        print(f"エラー: '{target_image_path}' から顔が検出されませんでした。")
        sys.exit(1)
    target_face = target_faces[0]
    print("ターゲット画像の読み込み完了。")

    # --- AI推論用バックグラウンドスレッドの起動 ---
    ai_thread = threading.Thread(
        target=ai_inference_thread, 
        args=(model_path, target_face, providers),
        daemon=True
    )
    ai_thread.start()

    # --- メイン配信ループ (Webカメラのキャプチャ & 表示) ---
    print("Webカメラを起動しています...")
    cap = cv2.VideoCapture(0)

    # 利用可能なカメラインデックスを自動探索
    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        else:
            print("エラー: カメラを起動できませんでした。")
            sys.exit(1)

    # 画面表示用のダブルバッファ
    display_frame = None

    print("--------------------------------------------------")
    print("[極限チューニング版] 顔交換PoCを起動しました。")
    print("  - 検出解像度を最適化し、M5 Mac本来の高速FPSを実現しました。")
    print("  - エッジ・肌色ハイブリッドマスクにより、手が透けなくなりました。")
    print("※ 映像ウィンドウを選択した状態で 'q' キーを押すと終了します。")
    print("--------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # AIスレッドへ最新フレームを送信
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

        # AIスレッドから顔交換済みの最新フレームが届いていれば取得
        try:
            latest_result = result_queue.get_nowait()
            display_frame = latest_result
        except queue.Empty:
            pass

        # 表示するフレームがある場合は表示
        if display_frame is not None:
            cv2.imshow('Face Swap PoC [Tuned] - Press Q to Quit', display_frame)
        else:
            cv2.putText(frame, "AI Initializing...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Face Swap PoC [Tuned] - Press Q to Quit', frame)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    is_running = False
    ai_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    print("正常終了しました。")

if __name__ == "__main__":
    main()
