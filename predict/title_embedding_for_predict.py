import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ----- 2. Load mô hình -----
model_name = "xlm-roberta-base"
# Nếu dùng tiếng Việt tốt hơn: model_name = "vinai/phobert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# -----  Hàm lấy embedding -----
def get_title_embedding(text, max_length=50):
    """Trả về vector embedding 768 chiều từ tiêu đề."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Hidden state: (1, seq_len, 768)
    hidden_states = outputs.last_hidden_state

    # Dùng vector [CLS]
    cls_vec = hidden_states[:, 0, :].squeeze().numpy()
    return cls_vec