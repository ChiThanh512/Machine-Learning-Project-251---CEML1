import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def train_and_evaluate_hmm(X_train, X_test, y_train, y_test, class_names):
    """
    Huấn luyện 10 mô hình HMM trên dữ liệu đã được chia sẵn và đánh giá.
    
    Hàm này không còn tự tải hay xử lý dữ liệu nữa.
    
    Args:
        X_train, X_test, y_train, y_test: Dữ liệu đã được chia.
        class_names (list): Danh sách tên các lớp để hiển thị kết quả.
    """
    # 1. HUẤN LUYỆN 10 MÔ HÌNH HMM
    print("\n--- Bắt đầu huấn luyện 10 mô hình HMM... ---")
    hmm_models = []
    for i in range(len(class_names)):
        # Lấy ra danh sách các chuỗi của lớp hiện tại
        X_class_list = [X_train[j] for j, label in enumerate(y_train) if label == i]
        
        # Nối tất cả các chuỗi lại thành một mảng lớn
        X_class_concatenated = np.vstack(X_class_list)
        # Tạo mảng lengths để cho HMM biết độ dài của từng chuỗi
        lengths = [len(x) for x in X_class_list]
        
        # Khởi tạo mô hình GaussianHMM
        # n_components: số trạng thái ẩn (hyperparameter cần tinh chỉnh)
        # covariance_type: "diag" là lựa chọn phổ biến cho MFCC
        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
        
        # Huấn luyện mô hình với dữ liệu nối và mảng lengths
        model.fit(X_class_concatenated, lengths=lengths)
        hmm_models.append(model)
        print(f"Đã huấn luyện xong mô hình cho lớp: '{class_names[i]}'")

    # 2. ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ
    print("\n--- Đang đánh giá trên tập kiểm thử... ---")
    y_pred = []
    # Bây giờ X_test là một danh sách các chuỗi
    for test_sequence in X_test:
        log_likelihoods = []
        for model in hmm_models:
            # Chấm điểm cho từng chuỗi
            score = model.score(test_sequence)
            log_likelihoods.append(score)
        
        # Tìm chỉ số (lớp) của mô hình có log-likelihood cao nhất
        predicted_class = np.argmax(log_likelihoods)
        y_pred.append(predicted_class)

    # 3. HIỂN THỊ KẾT QUẢ
    print("\n--- Kết quả đánh giá ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác (Accuracy): {accuracy:.4f}")

    # Vẽ ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.xlabel('Nhãn dự đoán (Predicted Label)')
    plt.ylabel('Nhãn thật (True Label)')
    plt.show()