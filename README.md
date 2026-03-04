# YouTube Video Optimizer 🎥

[![Python](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hệ thống AI tối ưu hóa tiêu đề và thumbnail video YouTube, giúp dự đoán CTR (Click-Through Rate) và đưa ra gợi ý cải thiện trước khi đăng tải video.

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Demo](#-demo)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [API Reference](#-api-reference)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Mô hình AI](#-mô-hình-ai)
- [Deploy lên Production](#-deploy-lên-production)
- [Troubleshooting](#-troubleshooting)
- [Đóng góp](#-đóng-góp)

## ✨ Tính năng chính

### 🎯 Phân tích thông minh
- **Dự đoán CTR**: Sử dụng mô hình FiBiNET deep learning để dự đoán khả năng thu hút người xem
- **Phân tích tiêu đề**: Trích xuất 768 đặc trưng embedding từ PhoBERT cho tiêu đề tiếng Việt
- **Phân tích thumbnail**: 
  - Đánh giá độ sáng, độ tương phản
  - Phát hiện khuôn mặt (Face Detection)
  - Tính toán attractiveness score
  - Embedding 512 chiều từ ResNet/VGG

### 💡 Gợi ý tối ưu
- Đề xuất cải thiện tiêu đề (độ dài, từ khóa, câu hỏi, số liệu)
- Gợi ý chỉnh sửa thumbnail (độ sáng, tương phản, khuôn mặt)
- Phân loại video: "CAO TIỀM NĂNG" hoặc "THẤP/TRUNG BÌNH"
- Điểm tối ưu hóa từ 0-1 (càng cao càng tốt)

### 🚀 Linh hoạt
- Hỗ trợ 2 phương thức: URL YouTube hoặc Upload thumbnail
- API RESTful dễ tích hợp
- Giao diện web thân thiện
- Xử lý đa ngôn ngữ (tiếng Việt, tiếng Anh)

## 🎬 Demo

![Screenshot](docs/screenshot.png)

**Ví dụ sử dụng:**

```bash
# Khởi chạy server
python app.py

# Truy cập giao diện web
http://localhost:8000
```

**API Request:**

```bash
curl -X POST "http://localhost:8000/api/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Hướng dẫn lập trình Python cho người mới bắt đầu",
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }'
```

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│   Web UI        │
│ (HTML/CSS/JS)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         v                  v                  v
┌────────────────┐  ┌──────────────┐  ┌──────────────┐
│ Title Feature  │  │  Thumbnail   │  │   FiBiNET    │
│  Extraction    │  │  Analysis    │  │    Model     │
│  (PhoBERT)     │  │ (CV + DL)    │  │ (TensorFlow) │
└────────────────┘  └──────────────┘  └──────────────┘
```

## 💻 Yêu cầu hệ thống

### Phần cứng
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Ổ cứng**: 2GB dung lượng trống
- **CPU**: Hỗ trợ AVX (cho TensorFlow)
- **GPU**: Không bắt buộc (sử dụng tensorflow-cpu)

### Phần mềm
- **Python**: 3.11.9 (khuyến nghị) hoặc 3.9+
- **pip**: 23.0+
- **Git**: Để clone repository
- **Hệ điều hành**: Windows, macOS, Linux

## 🔧 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/yourusername/YouTube_Optimization_Ultra_Demo.git
cd YouTube_Optimization_Ultra_Demo-main
```

### 2. Tạo môi trường ảo

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý**: Quá trình cài đặt có thể mất 5-10 phút do các gói deep learning khá lớn.

### 4. Tải mô hình (nếu chưa có)

Đảm bảo thư mục `predict/fibinet_model_final/` chứa đầy đủ:
- `saved_model.pb`
- `keras_metadata.pb`
- `variables/variables.data-00000-of-00001`
- `variables/variables.index`

### 5. Kiểm tra cài đặt

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## 🚀 Sử dụng

### Chạy server development

```bash
python app.py
```

Server sẽ chạy tại: `http://localhost:8000`

### Chạy với Uvicorn (production)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Sử dụng giao diện web

1. Mở trình duyệt: `http://localhost:8000`
2. Nhập tiêu đề video
3. Chọn một trong hai:
   - Nhập URL video YouTube
   - Upload file thumbnail
4. Nhấn "Phân tích"
5. Xem kết quả và gợi ý

### Sử dụng API

#### 1. Phân tích từ URL YouTube

**Endpoint:** `POST /api/optimize`

**Request:**
```json
{
  "title": "Cách làm bánh mì Việt Nam chuẩn vị",
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:**
```json
{
  "title": "Cách làm bánh mì Việt Nam chuẩn vị",
  "optimization_score": 0.7234,
  "prediction_class": "CAO TIỀM NĂNG",
  "thumbnail_analysis": {
    "brightness": 0.65,
    "contrast": 0.58,
    "face_count": 1,
    "predicted_score": 72.5,
    "url": "https://i.ytimg.com/vi/VIDEO_ID/maxresdefault.jpg"
  },
  "recommendations": [
    "Điểm CTR dự đoán cao! Tiêu đề và thumbnail này có tiềm năng thu hút người xem tốt.",
    "Có một khuôn mặt trong thumbnail - tốt cho việc tạo kết nối với người xem.",
    "Độ dài tiêu đề phù hợp, dễ đọc trên mọi thiết bị.",
    "Thumbnail có chất lượng tốt với các yếu tố hình ảnh cân đối."
  ]
}
```

#### 2. Phân tích với Upload thumbnail

**Endpoint:** `POST /api/optimize-with-upload`

**Request:** `multipart/form-data`
- `title` (string): Tiêu đề video
- `thumbnail_file` (file): File ảnh thumbnail (JPG, PNG)

#### 3. Health check

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "YouTube Video Optimizer",
  "version": "1.0.0"
}
```

## 📁 Cấu trúc dự án

```
YouTube_Optimization_Ultra_Demo-main/
│
├── app.py                          # FastAPI application chính
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version cho deploy
├── README.md                       # Tài liệu này
│
├── predict/                        # Module dự đoán
│   ├── predict_service.py         # Logic dự đoán chính
│   ├── extract_title_features.py  # Trích xuất features từ tiêu đề
│   ├── title_embedding_for_predict.py  # PhoBERT embedding
│   └── fibinet_model_final/       # Mô hình TensorFlow đã train
│       ├── saved_model.pb
│       ├── keras_metadata.pb
│       └── variables/
│           ├── variables.data-00000-of-00001
│           └── variables.index
│
├── backend/                        # Services phụ trợ
│   └── thumbnail_score_service.py # Chấm điểm thumbnail
│
├── embeddings/                     # Pre-computed embeddings
│   ├── thumbnail_embeddings.npy   # Numpy array embeddings
│   ├── thumbnail_embeddings.csv   # CSV format
│   ├── image_mapping.csv          # Mapping ID ↔ image
│   ├── image_names.json           # Danh sách tên ảnh
│   └── embedding_summary.json     # Metadata
│
├── static/                         # Frontend assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/                      # HTML templates
│   └── index.html
│
└── thumbnail/                      # Thư mục chứa ảnh thumbnail
    └── 1/
```

## 🤖 Mô hình AI

### FiBiNET (Field-aware Bilinear Interaction Network)

**Kiến trúc:**
- **Input Layer**: 1,292 features
  - 4 sparse features (categorical)
  - 10 dense features (numerical)
  - 768 title embedding dimensions (PhoBERT)
  - 512 image embedding dimensions (ResNet/VGG)

- **Feature Interaction**:
  - SENET layer (Squeeze-and-Excitation)
  - Bilinear Interaction
  - Field-aware mechanism

- **Output**: Sigmoid activation (0-1 probability)

**Performance:**
- Training Accuracy: ~85%
- Validation AUC: ~0.82
- Inference time: ~50ms/sample

### Feature Engineering

#### Title Features (10 base + 768 embedding)
- `caps_ratio`: Tỷ lệ chữ hoa
- `question_count`: Số câu hỏi
- `length_chars`: Độ dài ký tự
- `sentiment`: Cảm xúc (-1 đến 1)
- `separator_count`: Số ký tự phân cách
- `has_emoji`: Có emoji không (0/1)
- `has_question`: Có câu hỏi không (0/1)
- `sentiment_category`: Negative/Neutral/Positive
- **768-dim PhoBERT embedding**

#### Thumbnail Features (4 base + 512 embedding)
- `brightness`: Độ sáng (0-1)
- `contrast`: Độ tương phản (0-1)
- `attractiveness_score`: Điểm hấp dẫn (0-1)
- `face_count`: Số khuôn mặt
- `has_face`: Có khuôn mặt không (0/1)
- **512-dim image embedding**

## 🌐 Deploy lên Production

### Deploy với Docker

**Dockerfile:**
```dockerfile
FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t youtube-optimizer .
docker run -p 8000:8000 youtube-optimizer
```

### Deploy lên Heroku

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku ps:scale web=1
heroku open
```

### Deploy lên Railway/Render

1. Kết nối GitHub repository
2. Chọn branch `main`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Biến môi trường

Tạo file `.env`:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Model paths
MODEL_PATH=predict/fibinet_model_final
EMBEDDINGS_PATH=embeddings

# Security (for production)
API_KEY=your-secret-api-key
CORS_ORIGINS=https://yourdomain.com
```

## 🔍 Troubleshooting

### Lỗi thường gặp

#### 1. ImportError: cannot import name 'FiBiNET'

**Nguyên nhân:** Deepctr chưa được cài đặt đúng

**Giải pháp:**
```bash
pip uninstall deepctr
pip install deepctr==0.9.3
```

#### 2. TensorFlow Error: Illegal instruction (core dumped)

**Nguyên nhân:** CPU không hỗ trợ AVX

**Giải pháp:**
```bash
# Sử dụng TensorFlow build không cần AVX
pip install intel-tensorflow
```

#### 3. ModuleNotFoundError: No module named 'cv2'

**Nguyên nhân:** OpenCV chưa được cài

**Giải pháp:**
```bash
pip install opencv-python-headless==4.8.1.78
```

#### 4. Model loading error

**Nguyên nhân:** File model bị thiếu hoặc corrupt

**Giải pháp:**
1. Kiểm tra thư mục `predict/fibinet_model_final/`
2. Đảm bảo có đủ 4 files: saved_model.pb, keras_metadata.pb, variables.data, variables.index
3. Re-download model nếu cần

#### 5. PhoBERT out of memory

**Nguyên nhân:** RAM không đủ cho PhoBERT model

**Giải pháp:**
```python
# Trong title_embedding_for_predict.py, sử dụng lightweight model
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("vinai/phobert-base", 
                                   use_cache=False,
                                   low_cpu_mem_usage=True)
```

#### 6. Face detection không hoạt động

**Nguyên nhân:** Cascade file không tìm thấy

**Giải pháp:**
```bash
# Download Haarcascade manually
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# Đặt vào thư mục predict/
```

### Debug mode

Bật debug để xem log chi tiết:

```bash
# Trong app.py
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="debug")
```

### Kiểm tra dependencies

```bash
pip list | grep -E "tensorflow|torch|fastapi|opencv"
```

## 🧪 Testing

### Unit tests

```bash
pytest tests/
```

### Integration tests

```bash
pytest tests/integration/
```

### Load testing với Locust

```bash
pip install locust
locust -f tests/load_test.py
```

## 📊 Performance Optimization

### Caching

Thêm Redis cache cho predictions:

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="yt-opt")
```

### Batch Processing

Xử lý nhiều videos cùng lúc:

```python
@app.post("/api/batch-optimize")
async def batch_optimize(requests: List[OptimizationRequest]):
    results = []
    for req in requests:
        result = await optimize_video(req)
        results.append(result)
    return results
```

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

### Coding Standards

- PEP 8 cho Python code
- Type hints cho functions
- Docstrings cho modules/classes/functions
- Unit tests cho features mới

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Vietnamese BERT model
- [DeepCTR](https://github.com/shenweichen/DeepCTR) - Deep learning models for CTR prediction
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library

## 📧 Contact

Project Link: [https://github.com/yourusername/YouTube_Optimization_Ultra_Demo](https://github.com/yourusername/YouTube_Optimization_Ultra_Demo)

## 🗺️ Roadmap

- [ ] Thêm hỗ trợ ngôn ngữ tiếng Anh
- [ ] A/B testing cho nhiều biến thể thumbnail
- [ ] Export báo cáo PDF
- [ ] Mobile app (React Native)
- [ ] Real-time analytics dashboard
- [ ] Competitor analysis
- [ ] Trend prediction
- [ ] Video description optimization
- [ ] Tag suggestion

---

**Made with ❤️ for YouTube Creators**
