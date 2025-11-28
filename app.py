from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
import base64

# Add predict folder to path
sys.path.append(str(Path(__file__).parent / "predict"))
sys.path.append(str(Path(__file__).parent / "backend"))

from predict_service import predict_new_content
from thumbnail_score_service import score_thumbnail

app = FastAPI(
    title="YouTube Video Optimizer",
    description="Tối ưu hóa tiêu đề và thumbnail video YouTube trước khi đăng tải",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class OptimizationRequest(BaseModel):
    title: str
    video_url: str

class OptimizationResponse(BaseModel):
    title: str
    optimization_score: float
    prediction_class: str
    thumbnail_analysis: dict
    recommendations: list[str]

# Helper function to extract video ID and get thumbnail URL
def get_thumbnail_url(video_input: str) -> str:
    """
    Convert YouTube URL or Video ID to thumbnail URL
    """
    import re
    
    # If already a thumbnail URL, return as is
    if 'ytimg.com' in video_input or 'ggpht.com' in video_input:
        return video_input
    
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
    ]
    
    video_id = None
    for pattern in patterns:
        match = re.search(pattern, video_input)
        if match:
            video_id = match.group(1)
            break
    
    if not video_id:
        raise ValueError("Không thể trích xuất Video ID. Vui lòng nhập URL YouTube hợp lệ hoặc Video ID.")
    
    # Return high quality thumbnail URL
    return f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

# Helper function to generate recommendations
def generate_recommendations(score: float, thumb_analysis: dict, title: str) -> list[str]:
    recommendations = []
    
    # Score-based recommendations
    if score < 0.4:
        recommendations.append("Điểm CTR dự đoán thấp. Nên tối ưu hóa cả tiêu đề và thumbnail để tăng khả năng thu hút người xem.")
    elif score < 0.6:
        recommendations.append("Điểm CTR ở mức trung bình. Video có tiềm năng nhưng vẫn có thể cải thiện đáng kể.")
    else:
        recommendations.append("Điểm CTR dự đoán cao! Tiêu đề và thumbnail này có tiềm năng thu hút người xem tốt.")
    
    # Thumbnail analysis recommendations
    brightness = thumb_analysis.get('brightness', 0.5)
    if brightness < 0.3:
        recommendations.append("Thumbnail có độ sáng thấp. Tăng độ sáng sẽ giúp thumbnail nổi bật hơn trên trang kết quả tìm kiếm.")
    elif brightness > 0.8:
        recommendations.append("Thumbnail hơi sáng quá mức. Giảm độ sáng một chút để cân bằng và dễ nhìn hơn.")
    
    contrast = thumb_analysis.get('contrast', 0.5)
    if contrast < 0.2:
        recommendations.append("Độ tương phản thấp làm thumbnail kém nổi bật. Tăng độ tương phản giữa các yếu tố trong ảnh.")
    elif contrast > 0.7:
        recommendations.append("Độ tương phản cao giúp thumbnail nổi bật tốt.")
    
    face_count = thumb_analysis.get('face_count', 0)
    if face_count == 0:
        recommendations.append("Không phát hiện khuôn mặt trong thumbnail. Thêm khuôn mặt người có thể tăng sự kết nối và CTR cho một số loại nội dung.")
    elif face_count == 1:
        recommendations.append("Có một khuôn mặt trong thumbnail - tốt cho việc tạo kết nối với người xem.")
    elif face_count > 2:
        recommendations.append("Có nhiều khuôn mặt trong thumbnail. Đảm bảo không làm người xem bị phân tâm hoặc rối mắt.")
    
    # Title recommendations
    title_len = len(title)
    if title_len < 30:
        recommendations.append("Tiêu đề khá ngắn. Cân nhắc thêm chi tiết hoặc từ khóa để mô tả rõ hơn về nội dung.")
    elif title_len > 80:
        recommendations.append("Tiêu đề dài có thể bị cắt trên thiết bị di động. Cân nhắc rút ngắn để đảm bảo hiển thị đầy đủ.")
    else:
        recommendations.append("Độ dài tiêu đề phù hợp, dễ đọc trên mọi thiết bị.")
    
    if '?' in title:
        recommendations.append("Tiêu đề có câu hỏi - đây là cách tốt để kích thích tò mò và tương tác của người xem.")
    
    # Check for numbers in title
    import re
    if re.search(r'\d+', title):
        recommendations.append("Tiêu đề có số - điều này thường thu hút sự chú ý và tăng CTR.")
    
    # Quality score for thumbnail
    thumb_score = thumb_analysis.get('predicted_score', 50)
    if thumb_score < 40:
        recommendations.append("Điểm chất lượng thumbnail thấp. Nên thiết kế lại với màu sắc tươi sáng, text rõ ràng và hình ảnh chất lượng cao.")
    elif thumb_score >= 70:
        recommendations.append("Thumbnail có chất lượng tốt với các yếu tố hình ảnh cân đối.")
    
    return recommendations

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_video(request: OptimizationRequest):
    """
    Phân tích và tối ưu hóa tiêu đề và thumbnail video YouTube
    """
    try:
        # Convert video URL/ID to thumbnail URL
        thumbnail_url = get_thumbnail_url(request.video_url)
        
        # Run prediction
        result = predict_new_content(
            title_raw=request.title,
            thumbnail_url=thumbnail_url
        )
        
        # Get thumbnail analysis
        try:
            thumb_analysis = score_thumbnail(image_url=thumbnail_url)
        except Exception as e:
            print(f"Thumbnail analysis warning: {e}")
            # Fallback values if thumbnail analysis fails
            thumb_analysis = {
                "brightness": 0.5,
                "contrast": 0.5,
                "face_count": 0,
                "predicted_score": 50.0,
                "model_used": "fallback"
            }
        
        # Add URL to thumbnail analysis
        thumb_analysis['url'] = thumbnail_url
        
        # Generate recommendations
        recommendations = generate_recommendations(
            score=result['optimization_score'],
            thumb_analysis=thumb_analysis,
            title=request.title
        )
        
        return OptimizationResponse(
            title=result['title'],
            optimization_score=result['optimization_score'],
            prediction_class=result['prediction_class'],
            thumbnail_analysis=thumb_analysis,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@app.post("/api/optimize-with-upload")
async def optimize_with_upload(
    title: str = Form(...),
    thumbnail_file: UploadFile = File(...)
):
    """
    Phân tích video với thumbnail được upload
    """
    try:
        # Read uploaded image
        image_bytes = await thumbnail_file.read()
        
        # Convert to base64 for preview
        img_base64 = base64.b64encode(image_bytes).decode()
        thumbnail_preview_url = f"data:image/jpeg;base64,{img_base64}"
        
        # Analyze thumbnail
        try:
            thumb_analysis = score_thumbnail(image_bytes=image_bytes)
        except Exception as e:
            print(f"Thumbnail analysis warning: {e}")
            thumb_analysis = {
                "brightness": 0.5,
                "contrast": 0.5,
                "face_count": 0,
                "predicted_score": 50.0,
                "model_used": "fallback"
            }
        
        thumb_analysis['url'] = thumbnail_preview_url
        
        # For uploaded files, we use a simplified prediction
        # You can implement full prediction with image embedding here
        from extract_title_features import extract_raw_signals
        from title_embedding_for_predict import get_title_embedding
        import numpy as np
        import pandas as pd
        
        # Extract title features
        title_feats = extract_raw_signals(title)
        
        # Get title embedding
        title_vector = get_title_embedding(title)
        title_embed_dict = {
            f'embed_{i+1}': float(title_vector[i])
            for i in range(768)
        }
        
        # Use thumbnail analysis results for image features
        image_feats = {
            'brightness': thumb_analysis['brightness'],
            'contrast': thumb_analysis['contrast'],
            'attractiveness_score': (thumb_analysis['predicted_score'] / 100.0),
            'face_count': thumb_analysis['face_count']
        }
        
        # Create dummy image embedding (zeros)
        image_embed_dict = {f'dim_{i+1}': 0.0 for i in range(512)}
        
        # Combine all features
        from predict_service import (
            DENSE_BASE_FEATURES, SPARSE_FEATURES, 
            ALL_FEATURE_NAMES,
            model, mms, mean_view_velocity
        )
        
        raw_data = {**title_feats, **image_feats, **title_embed_dict, **image_embed_dict}
        raw_data['view_velocity_proxy'] = mean_view_velocity
        
        df_new = pd.DataFrame([raw_data])
        
        # Create sparse features
        df_new['has_face'] = (df_new['face_count'] > 0).astype(int)
        df_new['has_emoji'] = (df_new.get('emoji_count', 0) > 0).astype(int)
        df_new['has_question'] = (df_new.get('question_count', 0) > 0).astype(int)
        
        NEGATIVE_THRESHOLD = -0.2
        POSITIVE_THRESHOLD = 0.2
        df_new['sentiment_category'] = np.select(
            [
                df_new['sentiment'] > POSITIVE_THRESHOLD,
                df_new['sentiment'] < NEGATIVE_THRESHOLD
            ],
            ['positive', 'negative'],
            default='neutral'
        )
        
        # Ensure all features exist
        for col in DENSE_BASE_FEATURES:
            if col not in df_new.columns:
                df_new[col] = 0.0
        
        # Scale
        df_new[DENSE_BASE_FEATURES] = mms.transform(df_new[DENSE_BASE_FEATURES])
        
        # Convert sentiment category
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        sent_text = df_new.at[0, 'sentiment_category']
        df_new.at[0, 'sentiment_category'] = mapping.get(sent_text, 1)
        
        # Ensure all features exist
        for name in ALL_FEATURE_NAMES:
            if name not in df_new.columns:
                df_new[name] = 0.0
        
        # Prepare input
        X_new = {}
        for name in ALL_FEATURE_NAMES:
            arr = df_new[name].values
            if name in SPARSE_FEATURES:
                X_new[name] = arr.astype(np.int32)
            else:
                X_new[name] = arr.astype(np.float32)
        
        # Predict
        pred = model.predict(X_new)
        y_pred_proba = float(np.squeeze(pred))
        
        result = {
            "title": title,
            "optimization_score": y_pred_proba,
            "prediction_class": "CAO TIỀM NĂNG" if y_pred_proba > 0.5 else "THẤP/TRUNG BÌNH"
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(
            score=result['optimization_score'],
            thumb_analysis=thumb_analysis,
            title=title
        )
        
        return OptimizationResponse(
            title=result['title'],
            optimization_score=result['optimization_score'],
            prediction_class=result['prediction_class'],
            thumbnail_analysis=thumb_analysis,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "service": "YouTube Video Optimizer",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
