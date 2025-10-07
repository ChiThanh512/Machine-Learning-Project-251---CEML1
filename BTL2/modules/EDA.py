# ============================================================
# modules/data_preprocessing.py
# Exploratory Data Analysis (EDA) & Preprocessing cho Fake News
# ============================================================

import pandas as pd
import numpy as np
import re, string
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from IPython.display import display

# ============================================================
# 1. EDA
# ============================================================

def run_eda(df, STOPWORDS):
    print("==============================================")
    print("📊 TỔNG QUAN DỮ LIỆU")
    print("==============================================")
    print("Tổng số mẫu:", len(df))
    print("Tổng số tin giả:", df["label"].value_counts()[0])
    print("Tổng số tin thật:", df["label"].value_counts()[1])
    print()
    df.info()

    # Missing value
    print("\n🔹 Missing value:")
    print(df.isna().sum())

    # Duplicate
    print("\n🔹 Kiểm tra trùng lặp:")
    print("Dữ liệu bị trùng lặp:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print("✅ Sau khi loại bỏ trùng lặp:", df.shape)

    # Thêm cột độ dài văn bản
    df["text_len"] = df["text"].apply(lambda x: len(str(x).split()))

    # Thống kê độ dài
    print("\n--- Thống kê độ dài văn bản ---")
    print("Min:", df["text_len"].min())
    print("Max:", df["text_len"].max())
    print("Mean:", df["text_len"].mean())
    print("Median:", df["text_len"].median())
    print("80%:", np.quantile(df["text_len"], 0.8))

    # Biểu đồ histogram
    plt.figure(figsize=(8,5))
    sns.histplot(df["text_len"], bins=50, kde=True)
    plt.title("Phân phối độ dài văn bản")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8,2))
    sns.boxplot(x=df["text_len"])
    plt.title("Boxplot độ dài văn bản")
    plt.show()

    # CountVectorizer unigram
    vectorizer_uni = CountVectorizer(stop_words="english", ngram_range=(1,1))
    X_uni = vectorizer_uni.fit_transform(df["text"])
    vocab_size = len(vectorizer_uni.vocabulary_)
    print(f"📌 Vocabulary size: {vocab_size}")

    # Top 20 unigrams
    unigrams_freq = zip(vectorizer_uni.get_feature_names_out(), X_uni.sum(axis=0).tolist()[0])
    top_unigrams = sorted(unigrams_freq, key=lambda x: x[1], reverse=True)[:20]

    top_uni_df = pd.DataFrame(top_unigrams, columns=["word", "count"])
    sns.barplot(x="count", y="word", data=top_uni_df)
    plt.title("Top 20 Unigrams")
    plt.show()

    # WordCloud
    all_text = " ".join(df["text"].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Từ khóa nổi bật")
    plt.show()

    # Stopword ratio
    def stopword_ratio(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        return sum(1 for w in words if w.lower() in STOPWORDS) / len(words)

    df["stopword_ratio"] = df["text"].apply(stopword_ratio)
    print("📊 Tỷ lệ stopwords trung bình:", df["stopword_ratio"].mean())

    # Ký tự đặc biệt
    df["num_punct"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    df["num_digits"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
    df["num_urls"] = df["text"].apply(lambda x: len(re.findall(r"http[s]?://\S+", str(x))))
    print("🔹 Trung bình số dấu câu:", df["num_punct"].mean())
    print("🔹 Trung bình số chữ số:", df["num_digits"].mean())
    print("🔹 Trung bình số URL:", df["num_urls"].mean())

    # Sentiment
    df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    plt.figure(figsize=(8,5))
    df["sentiment"].hist(bins=50)
    plt.title("Phân phối sentiment")
    plt.show()

    print("📊 Sentiment trung bình:")
    print(df.groupby("label")["sentiment"].mean())

    # Pie chart phân bố nhãn
    df["label"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(5,5))
    plt.title("Phân bố nhãn (Fake vs Real)")
    plt.show()

    # Dọn dẹp cột phụ
    df.drop(["num_punct", "num_digits", "num_urls", "sentiment", "stopword_ratio"], axis=1, inplace=True)
    print("\n✅ Hoàn tất bước EDA.")
    return df


# ============================================================
# 2. Tiền xử lý văn bản & chia tập
# ============================================================

def clean_text(text, STOPWORDS, CONFIG, bert_tokenizer=None):
    if CONFIG["preprocessing"]["lowercase"]:
        text = text.lower()
    if CONFIG["preprocessing"]["remove_brackets"]:
        text = re.sub(r'\[.*?\]', '', text)
    if CONFIG["preprocessing"]["remove_urls"]:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    if CONFIG["preprocessing"]["remove_html"]:
        text = re.sub(r'<.*?>+', '', text)
    if CONFIG["preprocessing"]["remove_punct"]:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if CONFIG["preprocessing"]["remove_newline"]:
        text = text.replace("\n", " ")
    if CONFIG["preprocessing"]["remove_numbers"]:
        text = re.sub(r'\w*\d\w*', '', text)
    if CONFIG["preprocessing"]["remove_symbols"]:
        text = re.sub(r"\\W", " ", text)

    if CONFIG["preprocessing"]["remove_stopwords"]:
        text = " ".join([w for w in text.split() if w not in STOPWORDS])

    tokens = None
    if CONFIG["preprocessing"]["tokenization"] == "word":
        tokens = word_tokenize(text)
        if CONFIG["preprocessing"]["remove_stopwords"]:
            tokens = [w for w in tokens if w not in STOPWORDS]

    elif CONFIG["preprocessing"]["tokenization"] == "bert" and bert_tokenizer is not None:
        tokens = bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=CONFIG["preprocessing"]["max_len"],
            truncation=True,
            padding="max_length" if CONFIG["preprocessing"]["padding"] else False,
            return_tensors="pt"
        )
        return tokens

    if CONFIG["preprocessing"]["padding"] and tokens is not None:
        max_len = CONFIG["preprocessing"]["padding_length"]
        tokens = (tokens + ["<PAD>"] * (max_len - len(tokens)))[:max_len]

    return tokens if tokens is not None else text


def preprocess_texts(df, STOPWORDS, CONFIG, bert_tokenizer=None):
    """
    Làm sạch văn bản và chuẩn bị dữ liệu trước khi chia tập.
    """
    print("🧹 Đang tiền xử lý dữ liệu ...")

    # Bỏ các cột không cần thiết (nếu có)
    df = df.drop(["title", "subject", "date", "text_len"], axis=1, errors="ignore")
    print("Văn bản trước khi được xử lý")
    print(df["text"].iloc[0][:1000])

    df["text"] = df["text"].apply(lambda x: clean_text(x, STOPWORDS, CONFIG, bert_tokenizer)) 

    print("\nVăn bản sau khi được xử lý")
    print(df["text"].iloc[0][:1000])
    
    print("✅ Hoàn tất tiền xử lý.")
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập huấn luyện và kiểm thử.
    """
    print("✂️ Đang chia dữ liệu train/test ...")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"✅ Kích thước X_train: {X_train.shape}")
    print(f"✅ Kích thước X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

