import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from deepctr.models import FiBiNET
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from extract_title_feature import extract_raw_signals
from title_embedding_for_predict import get_title_embedding
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

def extract_thumbnail_features(thumbnail_url):
    """Giả lập trích xuất các feature từ thumbnail."""
    # (Đây là nơi bạn chạy ResNet50, tính toán face_count, brightness, v.v.)
    return {
        'brightness': random.uniform(0.5, 0.9),
        'contrast': random.uniform(0.7, 0.95),
        'attractiveness_score': random.uniform(0.8, 0.98),
        'face_count': random.choice([0, 1, 2]),
    }

def extract_embeddings(title, thumbnail_url):

    # Trích xuất vector cho title 
    title_vector = get_title_embedding(title)
    # Chuyển vector 768 chiều thành dictionary (embed_1: value, embed_2: value, ...)
    title_embed_dict = {
        name: title_vector[i] 
        for i, name in enumerate(TITLE_EMBED_COLS)
    }
    # --- 2. IMAGE EMBEDDING (TỪ FILE ĐÃ TIỀN XỬ LÝ) ---
    try:
        import json, os
        from urllib.parse import urlparse

        # load precomputed embeddings and names
        embeddings_np = np.load('embeddings/thumbnail_embeddings.npy')
        with open('embeddings/image_names.json', 'r') as f:
            image_names = json.load(f)

        # extract image stem from URL (last path segment, no ext)
        parsed = urlparse(thumbnail_url)
        img_stem = os.path.splitext(os.path.basename(parsed.path))[0]

        if img_stem in image_names:
            idx = image_names.index(img_stem)
            vec = embeddings_np[idx]
            image_embeddict = {f'dim{i+1}': float(vec[i]) for i in range(len(vec))}
        else:
            image_embed_dict = {name: 0.0 for name in IMAGE_EMBED_COLS}
    except Exception:
        image_embed_dict = {name: np.random.randn() for name in IMAGE_EMBED_COLS}

    return {title_embed_dict, image_embed_dict}

# --- TẢI CÁC ĐỐI TƯỢNG ĐÃ LƯU ---
try:
    model = tf.keras.models.load_model('fibinet_ctr_model', custom_objects={'FiBiNET': FiBiNET})
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
    
    if le_dict.get('sentiment_category') is not None:
        # lấy giá trị string rồi transform -> scalar
        cat = df_new.at[0, 'sentiment_category']
        df_new.at[0, 'sentiment_category'] = int(le_dict['sentiment_category'].transform([cat])[0])
    else:
        # fallback: map manual
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_new.at[0, 'sentiment_category'] = mapping.get(df_new.at[0, 'sentiment_category'], 1)

    # đảm bảo tất cả feature trong ALL_FEATURE_NAMES tồn tại (fill 0 nếu thiếu)
    for name in ALL_FEATURE_NAMES:
        if name not in df_new.columns:
            df_new[name] = 0.0

    # tạo X_new dictionary với numpy arrays
    X_new = {name: df_new[name].values for name in ALL_FEATURE_NAMES}

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
    new_title = "CÁCH TÔI kiếm $1000/tháng (KHÔNG CẦN VỐN) | Bài học kinh doanh mới"
    new_thumbnail_url = "https://example.com/new_thumb_01.jpg"
    print("\n--- BẮT ĐẦU PHÂN TÍCH DỮ LIỆU MỚI ---")
    result = predict_new_content(new_title, new_thumbnail_url)
    print(f"\nTiêu đề: {result['title']}")
    print(f"Điểm tối ưu hóa (Xác suất Top Tier): {result['optimization_score']:.4f}")
    print(f"Phân loại: {result['prediction_class']}")
    if result['prediction_class'] == 'CAO TIỀM NĂNG':
        print("🎉 Kết quả: Tiêu đề và Thumbnail này có tiềm năng CTR cao.")