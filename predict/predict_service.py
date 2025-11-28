import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from deepctr.models import FiBiNET
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from extract_title_features import extract_raw_signals
from title_embedding_for_predict import get_title_embedding
import requests
from PIL import Image
import cv2 
from io import BytesIO
import numpy as np
import random
from tensorflow.python.keras.layers import LSTM
# --- KHAI BÁO CẤU HÌNH ---
EMBEDDING_DIM_TITLE = 768
EMBEDDING_DIM_IMAGE = 512
DENSE_BASE_FEATURES = ['brightness', 'contrast', 'attractiveness_score', 'face_count', 
                       'caps_ratio', 'question_count', 'length_chars', 'sentiment', 
                       'view_velocity_proxy','separator_count']
SPARSE_FEATURES = ['has_face', 'has_emoji', 'has_question', 'sentiment_category'] 

TITLE_EMBED_COLS = [f'embed_{i+1}' for i in range(EMBEDDING_DIM_TITLE)]
IMAGE_EMBED_COLS = [f'dim_{i+1}' for i in range(EMBEDDING_DIM_IMAGE)]

# Tạo danh sách tên feature đầy đủ theo thứ tự mô hình yêu cầu
ALL_FEATURE_NAMES = SPARSE_FEATURES + DENSE_BASE_FEATURES + TITLE_EMBED_COLS + IMAGE_EMBED_COLS 

# --- CÁC HÀM TRÍCH XUẤT FEATURE  ---
def download_thumbnail(thumbnail_url):
    """Tải thumbnail từ URL và trả về PIL Image (RGB) hoặc None."""
    try:
        resp = requests.get(thumbnail_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

def resize_and_normalize(image, size=(224, 224)):
    """Resize ảnh và trả về (PIL_resized, numpy_normalized[0..1])."""
    img_resized = image.resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img_resized).astype(np.float32)
    return img_resized, arr / 255.0

def _to_uint8(arr):
    """Chuyển numpy array về uint8 an toàn (nhận biết nếu đang ở [0,1])."""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)

def _to_gray_uint8(arr):
    """Trả về ảnh grayscale uint8 từ array RGB hoặc grayscale."""
    u8 = _to_uint8(arr)
    if u8.ndim == 3 and u8.shape[2] == 3:
        return cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return u8

def calculate_brightness(img_array):
    """Độ sáng trung bình trên scale [0,1]."""
    gray = _to_gray_uint8(img_array)
    return float(np.mean(gray) / 255.0)

def calculate_contrast(img_array):
    """Độ tương phản (std) trên scale [0,1]."""
    gray = _to_gray_uint8(img_array)
    return float(np.std(gray) / 255.0)

def detect_faces_on_original(img_pil):
    """Đếm mặt bằng Haar Cascade; trả về 0 nếu không load được cascade."""
    try:
        arr = np.asarray(img_pil).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if cascade.empty():
            return 0
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50,50))
        return int(len(faces))
    except Exception:
        return 0

def extract_thumbnail_features(thumbnail_url):
    """
    Trích xuất các feature cơ bản từ thumbnail:
    brightness, contrast, attractiveness_score, face_count và 512-dim image embedding (fallback zeros).
    """
    EMBEDDING_DIM_IMAGE = 512

    img = download_thumbnail(thumbnail_url)
    if img is None:
        base = {'brightness': 0.5, 'contrast': 0.5, 'attractiveness_score': 0.5, 'face_count': 0}
        return {**base, **{f'dim_{i+1}': 0.0 for i in range(EMBEDDING_DIM_IMAGE)}}

    face_count = detect_faces_on_original(img)
    _, img_norm = resize_and_normalize(img)
    brightness = calculate_brightness(img_norm)
    contrast = calculate_contrast(img_norm)

    # heuristic attractiveness (cân nhắc: thay bằng model thực nếu có)
    attractiveness = 0.45 + 0.4 * brightness - 0.05 * contrast + 0.08 * min(face_count, 1)
    attractiveness = float(np.clip(attractiveness + random.uniform(-0.03, 0.03), 0.0, 1.0))

    image_embedding = np.zeros(EMBEDDING_DIM_IMAGE, dtype=float)

    return {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'attractiveness_score': attractiveness,
        'face_count': int(face_count),
        **{f'dim_{i+1}': float(v) for i, v in enumerate(image_embedding)}
    }

def extract_embeddings(title, thumbnail_url):
    title_vector = get_title_embedding(title)
    title_embed_dict = {
        f'embed_{i+1}': float(title_vector[i])
        for i in range(EMBEDDING_DIM_TITLE)
    }

    # Image embedding
    try:
        import json, os
        from urllib.parse import urlparse
        
        embeddings_np = np.load('embeddings/thumbnail_embeddings.npy')
        with open('embeddings/image_names.json', 'r') as f:
            image_names = json.load(f)

        parsed = urlparse(thumbnail_url)
        img_stem = os.path.splitext(os.path.basename(parsed.path))[0]

        if img_stem in image_names:
            idx = image_names.index(img_stem)
            vec = embeddings_np[idx]
            image_embed_dict = {
                f'dim_{i+1}': float(vec[i]) for i in range(len(vec))
            }
        else:
            image_embed_dict = {name: 0.0 for name in IMAGE_EMBED_COLS}

    except Exception:
        image_embed_dict = {name: 0.0 for name in IMAGE_EMBED_COLS}

    return {**title_embed_dict, **image_embed_dict}

# --- TẢI CÁC ĐỐI TƯỢNG ĐÃ LƯU ---
try:
    model = tf.keras.models.load_model('fibinet_model_final', custom_objects={'FiBiNET': FiBiNET})
    mms = joblib.load('scaler_mms.pkl')
    mean_view_velocity = joblib.load('mean_view_velocity.pkl')
    
    # Tải tất cả LabelEncoder và lưu vào dictionary
    le_dict = {}
    for feat in SPARSE_FEATURES:
        le_dict[feat] = joblib.load(f'le_{feat}.pkl')
    
    print("✅ Đã tải thành công model và các đối tượng tiền xử lý.")
except Exception as e:
    print(f"LỖI FATAL: Không thể tải các đối tượng cần thiết. Vui lòng kiểm tra file đã lưu. Lỗi: {e}")
    model = None
    raise SystemExit(1)

# --- TIỀN XỬ LÝ VÀ DỰ ĐOÁN ---

def predict_new_content(title_raw: str, thumbnail_url: str) -> dict:
    """
    Chạy toàn bộ luồng xử lý và dự đoán cho dữ liệu người dùng mới.
    """
    # 1. TRÍCH XUẤT TẤT CẢ FEATURE THÔ
    title_feats = extract_raw_signals(title_raw)
    thumb_feats = extract_thumbnail_features(thumbnail_url)
    embeddings = extract_embeddings(title_raw, thumbnail_url)
    
    raw_data = {**title_feats, **thumb_feats, **embeddings}
    
    # Gán giá trị mặc định cho FEATURE THIẾU (Hiệu suất chưa có)
    raw_data['view_velocity_proxy'] = mean_view_velocity
    
    df_new = pd.DataFrame([raw_data])

    # --- BƯỚC MỚI: TẠO SPARSE FEATURES TẠI THỜI ĐIỂM DỰ ĐOÁN ---
    
    # 1. Tạo các Binary Flags (Từ dữ liệu thô vừa trích xuất)
    df_new['has_face'] = (df_new['face_count'] > 0).astype(int)
    df_new['has_emoji'] = (df_new['emoji_count'] > 0).astype(int)
    df_new['has_question'] = (df_new['question_count'] > 0).astype(int)
    

    # 2. Xử lý Sentiment thành Feature Phân loại
    # Định nghĩa các ngưỡng (Threshold) cho điểm số sentiment (-1.0 đến +1.0)
    NEGATIVE_THRESHOLD = -0.2
    POSITIVE_THRESHOLD = 0.2

    df_new['sentiment_category'] = np.select(
        [
            df_new['sentiment'] > POSITIVE_THRESHOLD, 
            df_new['sentiment'] < NEGATIVE_THRESHOLD
        ], 
        [
            'positive', # Tạm thời dùng string để LabelEncoder xử lý
            'negative' 
        ],
        default='neutral'
    )
    # đảm bảo tất cả DENSE_BASE_FEATURES tồn tại trước khi scale
    for col in DENSE_BASE_FEATURES:
        if col not in df_new.columns:
            df_new[col] = 0.0

    # A. Scaling (chỉ áp dụng cho các base feature)
    df_new[DENSE_BASE_FEATURES] = mms.transform(df_new[DENSE_BASE_FEATURES])
    
    # sentiment_category hiện đang là string: 'positive', 'neutral', 'negative'
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    sent_text = df_new.at[0, 'sentiment_category']
    df_new.at[0, 'sentiment_category'] = mapping.get(sent_text, 1)  # fallback = neutral


    # đảm bảo tất cả feature trong ALL_FEATURE_NAMES tồn tại (fill 0 nếu thiếu)
    for name in ALL_FEATURE_NAMES:
        if name not in df_new.columns:
            df_new[name] = 0.0

    # tạo X_new dictionary với numpy arrays, ép dtype   
    X_new = {}
    for name in ALL_FEATURE_NAMES:
        if name not in df_new.columns:
            df_new[name] = 0.0  # fallback
        arr = df_new[name].values
        # Nếu feature là sparse -> int32, còn lại float32
        if name in SPARSE_FEATURES:
            X_new[name] = arr.astype(np.int32)
        else:
            X_new[name] = arr.astype(np.float32)


    # Dự đoán (robust lấy scalar)
    pred = model.predict(X_new)
    y_pred_proba = float(np.squeeze(pred))

    return {
        "title": title_raw,
        "optimization_score": y_pred_proba,
        "prediction_class": "CAO TIỀM NĂNG" if y_pred_proba > 0.5 else "THẤP/TRUNG BÌNH"
    }

# --- Demo chạy nếu file chạy trực tiếp ---
if __name__ == "__main__":
    new_title = "Người Nhanh Nhất Thế Giới Đối Đầu Robot"
    new_thumbnail_url = "image.png"
    print("\n--- BẮT ĐẦU PHÂN TÍCH DỮ LIỆU MỚI ---")
    result = predict_new_content(new_title, new_thumbnail_url)
    print(f"\nTiêu đề: {result['title']}")
    print(f"Điểm xác suất CTR click vào dựa trên Thumbnail và Tiêu đề  : {result['optimization_score']:.4f}")
    print(f"Phân loại: {result['prediction_class']}")
    if result['prediction_class'] == 'CAO TIỀM NĂNG':
        print("🎉 Kết quả: Tiêu đề và Thumbnail này có tiềm năng CTR cao.")
