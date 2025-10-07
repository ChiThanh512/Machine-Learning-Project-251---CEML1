"""
feature_extraction.py
---------------------
Module dùng để trích xuất đặc trưng văn bản (BOW, TF-IDF, Word2Vec)
và lưu ra file .npy để tái sử dụng sau.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import os


def extract_features(X_train, X_test, CONFIG):
    """
    Trích xuất đặc trưng từ văn bản theo phương pháp embedding:
    - bow (Bag of Words)
    - tfidf (TF-IDF)
    - word2vec (Word2Vec trung bình)
    """
    print("🧩 Đang trích xuất đặc trưng với embedding =", CONFIG["embedding"])

    if CONFIG["embedding"] == "bow":
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    elif CONFIG["embedding"] == "tfidf":
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    elif CONFIG["embedding"] == "word2vec":
        from gensim.models import Word2Vec

        # Chuyển text thành list từ
        X_train_tok = [str(x).lower().split() for x in X_train]
        X_test_tok = [str(x).lower().split() for x in X_test]

        # Huấn luyện Word2Vec trên dữ liệu train
        w2v_model = Word2Vec(
            sentences=X_train_tok, vector_size=100, window=5, min_count=1, workers=4
        )

        def get_w2v_embeddings(tokenized_texts, model, vector_size=100):
            all_embeddings = []
            for tokens in tokenized_texts:
                vectors = [model.wv[w] for w in tokens if w in model.wv]
                if len(vectors) == 0:
                    all_embeddings.append(np.zeros(vector_size))
                else:
                    all_embeddings.append(np.mean(vectors, axis=0))
            return np.vstack(all_embeddings)

        X_train_vec = get_w2v_embeddings(X_train_tok, w2v_model)
        X_test_vec = get_w2v_embeddings(X_test_tok, w2v_model)

    else:
        raise ValueError("❌ Embedding không hợp lệ! (bow | tfidf | word2vec)")

    print("✅ Trích xuất xong. Shapes:", X_train_vec.shape, X_test_vec.shape)
    return X_train_vec, X_test_vec




def save_features(X_train_vec, X_test_vec, prefix="X"):
    """
    💾 Lưu đặc trưng xuống file (.npz cho sparse, .npy cho dense)
    """
    print("💾 Đang lưu đặc trưng ...")

    # Tạo thư mục lưu
    os.makedirs("features", exist_ok=True)

    # Tạo đường dẫn file
    train_path = os.path.join("features", f"{prefix}_train_vec")
    test_path = os.path.join("features", f"{prefix}_test_vec")

    # Kiểm tra dạng dữ liệu
    if sparse.issparse(X_train_vec):
        sparse.save_npz(train_path + ".npz", X_train_vec)
        sparse.save_npz(test_path + ".npz", X_test_vec)
        print(f"✅ Đã lưu sparse matrix: {train_path}.npz, {test_path}.npz")
    else:
        np.save(train_path + ".npy", X_train_vec)
        np.save(test_path + ".npy", X_test_vec)
        print(f"✅ Đã lưu dense array: {train_path}.npy, {test_path}.npy")

    print(f"🎯 Shapes khi lưu: {X_train_vec.shape}, {X_test_vec.shape}")


def load_features(prefix="X"):
    """
    📂 Load đặc trưng đã lưu (.npz hoặc .npy)
    """
    print("📂 Đang load lại đặc trưng ...")

    npz_train = os.path.join("features", f"{prefix}_train_vec.npz")
    npz_test = os.path.join("features", f"{prefix}_test_vec.npz")
    npy_train = os.path.join("features", f"{prefix}_train_vec.npy")
    npy_test = os.path.join("features", f"{prefix}_test_vec.npy")

    if os.path.exists(npz_train):
        X_train_load = sparse.load_npz(npz_train)
        X_test_load = sparse.load_npz(npz_test)
        print(f"✅ Đã load dạng sparse (.npz): {prefix}")
    elif os.path.exists(npy_train):
        X_train_load = np.load(npy_train, allow_pickle=True)
        X_test_load = np.load(npy_test, allow_pickle=True)
        print(f"✅ Đã load dạng dense (.npy): {prefix}")
    else:
        raise FileNotFoundError("❌ Không tìm thấy file đặc trưng cần load.")

    print("🎯 Shapes sau khi load:", X_train_load.shape, X_test_load.shape)
    return X_train_load, X_test_load
