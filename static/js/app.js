// Handle file upload display
document.getElementById('thumbnail_file').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || '';
    document.getElementById('fileName').textContent = fileName ? `Đã chọn: ${fileName}` : '';
});

// Example functions
function fillExample1() {
    document.getElementById('title').value = '10 Điều Kỳ Diệu Về Vũ Trụ Mà Bạn Chưa Biết!';
    document.getElementById('video_url').value = 'O6tdnuHKgSQ';
    document.getElementById('thumbnail_file').value = '';
    document.getElementById('fileName').textContent = '';
}

function fillExample2() {
    document.getElementById('title').value = 'Bí Quyết Học Tiếng Anh Hiệu Quả Trong 30 Ngày';
    document.getElementById('video_url').value = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ';
    document.getElementById('thumbnail_file').value = '';
    document.getElementById('fileName').textContent = '';
}

// Form submission
document.getElementById('optimizerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const title = document.getElementById('title').value;
    const video_url = document.getElementById('video_url').value;
    const thumbnail_file = document.getElementById('thumbnail_file').files[0];
    
    // Validate input
    if (!video_url && !thumbnail_file) {
        showError('Vui lòng nhập URL video hoặc tải lên thumbnail');
        return;
    }
    
    // Hide previous results and errors
    hideResults();
    hideError();
    showLoading();
    disableSubmit();
    
    try {
        let response;
        
        if (thumbnail_file) {
            // Use uploaded file
            const formData = new FormData();
            formData.append('title', title);
            formData.append('thumbnail_file', thumbnail_file);
            
            response = await fetch('/api/optimize-with-upload', {
                method: 'POST',
                body: formData
            });
        } else {
            // Use video URL
            response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ title, video_url })
            });
        }
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Có lỗi xảy ra');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
        console.error('Error:', error);
    } finally {
        hideLoading();
        enableSubmit();
    }
});

function displayResults(data) {
    // Display title
    document.getElementById('displayTitle').textContent = data.title;
    
    // Score
    const score = (data.optimization_score * 100).toFixed(1);
    const scoreNum = parseFloat(score);
    document.getElementById('scoreValue').textContent = score + '%';
    
    // Reset color
    document.getElementById('scoreValue').style.color = '';
    
    // Add warning for very low scores
    if (scoreNum < 5) {
        document.getElementById('scoreValue').style.color = '#dc3545';
    }
    
    // Prediction class with more granular levels
    const predClass = document.getElementById('predictionClass');
    let classText = '';
    let className = 'prediction-class ';
    
    if (scoreNum >= 60) {
        classText = 'CAO TIỀM NĂNG';
        className += 'class-high';
    } else if (scoreNum >= 40) {
        classText = 'TRUNG BÌNH';
        className += 'class-medium';
    } else if (scoreNum >= 20) {
        classText = 'THẤP';
        className += 'class-low';
    } else {
        classText = 'RẤT THẤP - CẦN TỐI ƯU';
        className += 'class-very-low';
    }
    
    predClass.textContent = classText;
    predClass.className = className;
    
    // Title evaluation - Improved algorithm based on actual title quality
    const titleLen = data.title.length;
    const hasQuestion = data.title.includes('?');
    const hasNumber = /\d+/.test(data.title);
    // Check for emoji - simple heuristic
    const hasEmoji = /[\u{1F300}-\u{1F9FF}]|[\u2600-\u27BF]/u.test(data.title);
    const upperCasePattern = /[A-Z]/g;
    const upperCaseRatio = (data.title.match(upperCasePattern) || []).length / Math.max(titleLen, 1);
    const wordCount = data.title.trim().split(/\s+/).length;
    
    // Start with base score from overall CTR prediction
    let titleScore = Math.round(data.optimization_score * 60); // Use 60% of CTR score as base
    
    // Length optimization (ideal: 40-70 characters)
    if (titleLen >= 40 && titleLen <= 70) {
        titleScore += 15;
    } else if (titleLen >= 30 && titleLen < 40) {
        titleScore += 10;
    } else if (titleLen > 70 && titleLen <= 80) {
        titleScore += 10;
    } else if (titleLen < 30) {
        titleScore += 0; // Too short
    } else {
        titleScore += 5; // Too long
    }
    
    // Word count (ideal: 5-12 words)
    if (wordCount >= 5 && wordCount <= 12) {
        titleScore += 10;
    } else if (wordCount >= 3 && wordCount < 5) {
        titleScore += 5;
    }
    
    // Engagement elements
    if (hasQuestion) titleScore += 8;
    if (hasNumber) titleScore += 8;
    if (hasEmoji) titleScore += 4;
    
    // Penalize excessive caps (clickbait indicator)
    if (upperCaseRatio > 0.5) {
        titleScore -= 15;
    } else if (upperCaseRatio > 0.3) {
        titleScore -= 8;
    }
    
    // Cap at 0-100
    titleScore = Math.max(0, Math.min(100, titleScore));
    
    document.getElementById('titleScore').textContent = titleScore + '/100';
    
    let titleAssessment = '';
    if (titleScore >= 75) titleAssessment = 'Rất tốt - Tiêu đề hấp dẫn và tối ưu';
    else if (titleScore >= 55) titleAssessment = 'Tốt - Tiêu đề ổn, có thể cải thiện';
    else if (titleScore >= 35) titleAssessment = 'Trung bình - Nên tối ưu thêm';
    else titleAssessment = 'Yếu - Cần viết lại tiêu đề';
    document.getElementById('titleAssessment').textContent = titleAssessment;
    
    // Thumbnail evaluation
    const thumbScore = data.thumbnail_analysis.predicted_score;
    document.getElementById('thumbnailScore').textContent = thumbScore.toFixed(0) + '/100';
    
    let thumbAssessment = '';
    if (thumbScore >= 70) thumbAssessment = 'Rất tốt - Thumbnail chất lượng cao';
    else if (thumbScore >= 50) thumbAssessment = 'Tốt - Thumbnail ổn, có thể tốt hơn';
    else thumbAssessment = 'Cần cải thiện - Hãy thiết kế lại thumbnail';
    document.getElementById('thumbnailAssessment').textContent = thumbAssessment;
    
    // Thumbnail preview
    const thumbnailImg = document.getElementById('thumbnailPreview');
    const thumbnailError = document.getElementById('thumbnailError');
    
    if (data.thumbnail_analysis.url) {
        thumbnailImg.src = data.thumbnail_analysis.url;
        thumbnailImg.style.display = 'block';
        thumbnailError.style.display = 'none';
        
        // Handle image load error
        thumbnailImg.onerror = function() {
            thumbnailImg.style.display = 'none';
            thumbnailError.style.display = 'block';
        };
    } else {
        thumbnailImg.style.display = 'none';
        thumbnailError.style.display = 'block';
    }
    
    // Analysis grid
    const faceCount = data.thumbnail_analysis.face_count;
    const faceText = faceCount > 0 ? 'Có' : 'Không';
    const faceColor = faceCount > 0 ? '#28a745' : '#6c757d';
    
    const analysisGrid = document.getElementById('analysisGrid');
    analysisGrid.innerHTML = `
        <div class="analysis-item">
            <div class="analysis-label">Độ sáng</div>
            <div class="analysis-value">${(data.thumbnail_analysis.brightness * 100).toFixed(0)}%</div>
        </div>
        <div class="analysis-item">
            <div class="analysis-label">Độ tương phản</div>
            <div class="analysis-value">${(data.thumbnail_analysis.contrast * 100).toFixed(0)}%</div>
        </div>
        <div class="analysis-item">
            <div class="analysis-label">Khuôn mặt</div>
            <div class="analysis-value" style="color: ${faceColor}">${faceText}</div>
        </div>
        <div class="analysis-item">
            <div class="analysis-label">Điểm thumbnail</div>
            <div class="analysis-value">${data.thumbnail_analysis.predicted_score.toFixed(0)}</div>
        </div>
    `;
    
    // Recommendations
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = data.recommendations
        .map(rec => `<div class="recommendation-item">${rec}</div>`)
        .join('');
    
    // Show results
    showResults();
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Helper functions
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showResults() {
    document.getElementById('results').style.display = 'block';
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function disableSubmit() {
    document.getElementById('submitBtn').disabled = true;
}

function enableSubmit() {
    document.getElementById('submitBtn').disabled = false;
}
