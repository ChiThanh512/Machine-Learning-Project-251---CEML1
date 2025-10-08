from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(CONFIG, X_train_vec, y_train, X_test_vec, y_test):
    """
    Huấn luyện mô hình dựa trên lựa chọn trong CONFIG["classifier"].
    Trả về model đã huấn luyện và y_pred.
    """

    model_name = CONFIG.get("classifier", "").lower()
    print(f"🚀 Đang huấn luyện mô hình: {model_name.upper()}")

    if model_name == "nb":
        clf = MultinomialNB()
    elif model_name == "logreg":
        clf = LogisticRegression(max_iter=300)
    elif model_name == "svm":
        clf = SVC(kernel="linear")
    elif model_name == "decisiontree":
        clf = DecisionTreeClassifier()
    elif model_name == "randomforest":
        clf = RandomForestClassifier(random_state=0)
    elif model_name == "gbc":
        clf = GradientBoostingClassifier(random_state=0)
    else:
        raise ValueError("❌ Model không hợp lệ! (Chọn: nb | logreg | svm | decisiontree | randomforest | gbc)")

    # Huấn luyện mô hình
    clf.fit(X_train_vec, y_train)

    # Dự đoán
    y_pred = clf.predict(X_test_vec)

    print("✅ Huấn luyện hoàn tất.")
    return clf, y_pred
