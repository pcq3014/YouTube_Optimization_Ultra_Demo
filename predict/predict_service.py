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
        'has_face': 1 if random.choice([0, 1]) else 0
    }

def extract_embeddings(title, thumbnail_url):

    # Trích xuất vector cho title 
    title_vector = get_title_embedding(title)
    # Chuyển vector 768 chiều thành dictionary (embed_1: value, embed_2: value, ...)
    title_embed_dict = {
        name: title_vector[i] 
        for i, name in enumerate(TITLE_EMBED_COLS)
    }
    # --- 2. IMAGE EMBEDDING (GIẢ LẬP ĐỂ HOÀN THIỆN LUỒNG) ---
    # Thay thế bằng hàm ResNet50 thực tế khi bạn tích hợp nó
    image_embed_dict = {name: np.random.randn() for name in IMAGE_EMBED_COLS}
    
    return {**title_embed_dict, **image_embed_dict}

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
    exit() # Dừng chương trình nếu không tải được

# --- 2. HÀM CHÍNH: TIỀN XỬ LÝ VÀ DỰ ĐOÁN ---

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
    
    # 2. CHUẨN HÓA VÀ MÃ HÓA (Dùng .transform)
    
    # A. Scaling (chỉ áp dụng cho các base feature)
    df_new[DENSE_BASE_FEATURES] = mms.transform(df_new[DENSE_BASE_FEATURES])
    
    # B. Encoding (Áp dụng LabelEncoder)
    for feat in SPARSE_FEATURES:
        # Cột sentiment_category cần được chuyển sang string nếu chưa
        if feat == 'sentiment_category' and isinstance(df_new[feat].iloc[0], str):
             # Lấy category (string) để transform
             category = df_new[feat].iloc[0] 
             # Chuyển string thành ID số nguyên đã học
             df_new[feat] = le_dict[feat].transform([category]) 
        else:
             # Các cột binary (0/1) đã là số, chỉ cần đảm bảo đúng type
             df_new[feat] = df_new[feat].astype(int)

    # 3. ĐỊNH DẠNG ĐẦU VÀO CHO FI BI NET
    # Tạo Dictionary X_new với các giá trị mảng NumPy theo thứ tự feature_names
    X_new = {name: df_new[name].values for name in ALL_FEATURE_NAMES}
    
    # 4. DỰ ĐOÁN
    # Chú ý: model.predict nhận dictionary X_new
    y_pred_proba = model.predict(X_new)[0][0]
    
    # 5. GIẢI THÍCH (Bỏ qua SHAP để đơn giản hóa, chỉ trả về điểm)
    
    return {
        "title": title_raw,
        "optimization_score": float(y_pred_proba),
        "prediction_class": "CAO TIỀM NĂNG" if y_pred_proba > 0.5 else "THẤP/TRUNG BÌNH"
    }


# --- 3. PHẦN CHẠY DEMO ---

if __name__ == "__main__":
    
    new_title = "CÁCH TÔI kiếm $1000/tháng (KHÔNG CẦN VỐN) | Bài học kinh doanh mới"
    new_thumbnail_url = "https://example.com/new_thumb_01.jpg"
    
    print("\n--- BẮT ĐẦU PHÂN TÍCH DỮ LIỆU MỚI ---")
    
    # Gọi hàm chính
    result = predict_new_content(new_title, new_thumbnail_url)
    
    print(f"\nTiêu đề: {result['title']}")
    print(f"Điểm tối ưu hóa (Xác suất Top Tier): {result['optimization_score']:.4f}")
    print(f"Phân loại: {result['prediction_class']}")
    
    if result['prediction_class'] == 'CAO TIỀM NĂNG':
        print("🎉 Kết quả: Tiêu đề và Thumbnail này có tiềm năng CTR cao.")