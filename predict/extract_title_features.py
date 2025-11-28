import re
import emoji
from underthesea import word_tokenize 
from transformers import pipeline, AutoModelForSequenceClassification, XLMRobertaTokenizer

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Load slow tokenizer (không dùng fast)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)


# Kiểm tra emojis: 
def extract_emojis(text):
    emojis_found = [c for c in text if c in emoji.EMOJI_DATA]
    return emojis_found, len(emojis_found)

def get_sentiment(text: str):
    try:
        res = classifier(text)[0]
        # map thành số 
        label = res["label"]
        score = res["score"]

        if label == "positive":
            return score
        elif label == "negative":
            return -score
        return 0
    except:
        return 0

# Nhóm dấu câu 
separator_PUNCT = r"[\|\/\\\-\–\—\~\*\.·]+"
hype_PUNCT = r"[!?]{1,}" 

def count_pattern(text, pattern):
    return len(re.findall(pattern, text))

def extract_raw_signals(title: str):
    s = title or ""

    # Hiển thị các emoji và đếm
    emojis, count = extract_emojis(title)
    emojis_found = emojis
    emoji_count = count 

    # caps
    total_chars = len(s)
    caps_chars = sum(1 for c in s if c.isupper())
    caps_ratio = caps_chars / max(total_chars,1)
    words = re.findall(r"\b\w+\b", s)
    num_caps_words = sum(1 for w in words if w.isupper() and len(w)>1)

    # Dấu câu 
    question_count = count_pattern(s, r"\?+") # Đếm ký tự thể hiện câu hỏi
    hype_punct_count = count_pattern(s, hype_PUNCT) # Đếm ký tự thể hiện tạo cảm xúc 
    separator_count = count_pattern(s, separator_PUNCT) # Đếm ký tự ngăn cách  

    # Chiều dài
    length_chars = total_chars
    length_words = len(words)

    # tokens
    tokens = word_tokenize(s)
    # Bỏ dấu câu nếu cần
    unigrams = [t for t in tokens if re.match(r'\w+', t)]

    # sentiment từ XLM-R
    sentiment = get_sentiment(s)

    return {
        "emoji_list": emojis_found,
        "emoji_count": emoji_count,
        "caps_ratio": caps_ratio,
        "num_caps_words": num_caps_words,
        "question_count": question_count,
        "hype_count": hype_punct_count,
        "separator_count": separator_count,
        "length_chars": length_chars,
        "sentiment": sentiment,  # -1 → 1
    }

