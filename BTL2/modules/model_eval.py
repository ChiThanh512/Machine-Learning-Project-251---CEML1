# ============================================================
# 1. XỬ LÝ DỮ LIỆU CƠ BẢN & TRỰC QUAN HÓA
# ============================================================
import pandas as pd
import numpy as np
import re, string

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from textblob import TextBlob
import itertools
# ============================================================
# 2. TIỀN XỬ LÝ VĂN BẢN
# ============================================================
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))   # Khởi tạo stopwords


# ============================================================
# 3. TOKENIZER & PADDING (CHO ML / DL)
# ============================================================
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from keras.preprocessing.sequence import pad_sequences  # padding cho RNN/LSTM

# ============================================================
# 4. CHUẨN BỊ TẬP DỮ LIỆU & VECTOR HÓA
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ============================================================
# 5. EMBEDDINGS (Word2Vec, GloVe, BERT)
# ============================================================
import torch
import subprocess
import sys
try:
    import gensim
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim"])
# Word2Vec
from gensim.models import Word2Vec

# GloVe (chuyển sang word2vec format)
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# ============================================================
# 6. MÔ HÌNH MACHINE LEARNING TRUYỀN THỐNG
# ============================================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


# ============================================================
# 7. ĐÁNH GIÁ MÔ HÌNH
# ============================================================
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# ================================
#  Hàm tiền xử lý
# ================================
def preprocess_texts(texts, config):
    processed = []
    for text in texts:
        if config.get("lowercase", False):
            text = text.lower()
        if config.get("remove_stopwords", False):
            text = " ".join([w for w in text.split() if w not in STOPWORDS])
        if config.get("remove_numbers", False):
            text = ''.join(ch for ch in text if not ch.isdigit())
        if config.get("remove_punct", False):
            import string
            text = ''.join(ch for ch in text if ch not in string.punctuation)
        processed.append(text)
    return processed

# ================================
#  Hàm embedding
# ================================
def get_embeddings(X_train, method):
    if method == "bow":
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
    elif method == "word2vec":
        from gensim.models import Word2Vec
        tokenized = [t.split() for t in X_train]
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
        def sent2vec(tokens):
            return np.mean([model.wv[w] for w in tokens if w in model.wv] or [np.zeros(model.vector_size)], axis=0)
        X_train_vec = np.array([sent2vec(t) for t in tokenized])
        return X_train_vec, model
    else:
        raise ValueError("Unknown embedding method")
    return X_train_vec, vectorizer

# ================================
# Hàm tạo classifier
# ================================
def make_clf(name):
    if name == "nb":
        return MultinomialNB()
    elif name == "logreg":
        return LogisticRegression(max_iter=1000)
    elif name == "svm":
        return LinearSVC()
    elif name == "decisiontree":
        return DecisionTreeClassifier()
    else:
        raise ValueError("Unknown classifier")

# ================================
#  Grid Search qua các lựa chọn
# ================================
def run_grid_search(X_train, y_train, X_val, y_val, X_test, y_test):
    preprocessing_opts = [
        {"lowercase": True, "remove_punct": True,"remove_stopwords": True}
    ]
    embedding_opts = ["bow", "tfidf", "word2vec"]
    classifier_opts = ["nb", "logreg", "svm", "decisiontree"]

    results = []

    for pre_cfg, emb, clf_name in itertools.product(preprocessing_opts, embedding_opts, classifier_opts):
        if emb == "word2vec" and clf_name == "nb":
            print(f"⏩ Skipping invalid combination: {emb} + {clf_name}")
            continue
        print(f"\n=== Testing: preprocessing={pre_cfg}, embedding={emb}, classifier={clf_name} ===")

        # 1. Preprocess
        X_train_p = preprocess_texts(X_train, pre_cfg)
        X_val_p   = preprocess_texts(X_val, pre_cfg)
        X_test_p  = preprocess_texts(X_test, pre_cfg)

        # 2. Embedding
        X_train_feat, embedder = get_embeddings(X_train_p, emb)
        if emb in ["bow", "tfidf"]:
            X_val_feat  = embedder.transform(X_val_p)
            X_test_feat = embedder.transform(X_test_p)
        else:  # word2vec
            def sent2vec(tokens):
                return np.mean([embedder.wv[w] for w in tokens if w in embedder.wv] or [np.zeros(embedder.vector_size)], axis=0)
            X_val_feat  = np.array([sent2vec(t.split()) for t in X_val_p])
            X_test_feat = np.array([sent2vec(t.split()) for t in X_test_p])

        # 3. Classifier
        clf = make_clf(clf_name)
        clf.fit(X_train_feat, y_train)

        # 4. Eval
        val_pred  = clf.predict(X_val_feat)
        test_pred = clf.predict(X_test_feat)

        # 🔹 Thước đo chi tiết
        val_metrics = {
            "f1": f1_score(y_val, val_pred, average="macro"),
            "precision": precision_score(y_val, val_pred, average="macro"),
            "recall": recall_score(y_val, val_pred, average="macro"),
            "accuracy": accuracy_score(y_val, val_pred)
        }
        test_metrics = {
            "f1": f1_score(y_test, test_pred, average="macro"),
            "precision": precision_score(y_test, test_pred, average="macro"),
            "recall": recall_score(y_test, test_pred, average="macro"),
            "accuracy": accuracy_score(y_test, test_pred)
        }

        # 5️⃣ Lưu kết quả
        results.append({
            "preprocessing": pre_cfg,
            "embedding": emb,
            "classifier": clf_name,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()}
        })

        print(f"✅ Val F1={val_metrics['f1']:.4f} | Test F1={test_metrics['f1']:.4f}")

    return results
def print_model_info(CONFIG):
    print("=== Thông tin cấu hình ===")
    print(f"Embedding:   {CONFIG['embedding']}")
    print(f"Classifier:  {CONFIG['classifier']}")
    print("\n--- Preprocessing ---")
    for key, val in CONFIG["preprocessing"].items():
        print(f"{key}: {val}")
    print("======================\n")

def evaluate_model(y_test, y_pred):
    """In kết quả đánh giá cơ bản của mô hình"""
    print("📊 Kết quả đánh giá mô hình:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)


def split_data(df, test_size=0.3, val_size=0.5, random_state=42):
    """Chia dữ liệu thành train / val / test"""
    print("🔀 Đang chia tập dữ liệu ...")

    df_prepared = df.copy()
    X = df_prepared['text']
    y = df_prepared['label']

    # Chia train/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Chia tiếp test/val
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print(f"✅ Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    print("-" * 40)
    return X_train, X_val, X_test, y_train, y_val, y_test