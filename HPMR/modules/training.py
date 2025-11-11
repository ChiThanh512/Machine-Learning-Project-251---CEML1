import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from .HMM import continueHMM


warnings.filterwarnings("ignore", category=DeprecationWarning)

def _init_params(num_states, seq_list):
    """Khởi tạo A, pi, means, covariances đơn giản cho một lớp."""
    X = np.vstack(seq_list)              # (T_total, D)
    D = X.shape[1]
    T_total = X.shape[0]

    # pi: phân bố đều
    pi = np.full(num_states, 1.0/num_states)

    # A: gần như tuần tự (left-to-right nhẹ)
    A = np.zeros((num_states, num_states))
    for i in range(num_states):
        stay = 0.6
        move = 0.4
        if i == num_states - 1:
            A[i, i] = 1.0
        else:
            A[i, i] = stay
            A[i, i+1] = move
    # chuẩn hóa
    A /= A.sum(axis=1, keepdims=True)

    # Gán frame vào state theo tỷ lệ vị trí thời gian (simple segmentation)
    cumulative_lengths = np.cumsum([len(s) for s in seq_list])
    # chỉ số bắt đầu mỗi sequence không cần thiết ở đây; dùng vị trí tương đối
    idx = np.arange(T_total)
    rel = idx / (T_total + 1e-9)
    state_ids = np.minimum((rel * num_states).astype(int), num_states-1)

    means = np.zeros((num_states, D))
    covariances = np.zeros((num_states, D, D))
    for s in range(num_states):
        frames = X[state_ids == s]
        if len(frames) == 0:
            # fallback nếu rỗng
            means[s] = X[np.random.randint(0, T_total)]
            covariances[s] = np.diag(np.var(X, axis=0) + 1e-2)
        else:
            means[s] = frames.mean(axis=0)
            var = frames.var(axis=0) + 1e-2
            covariances[s] = np.diag(var)

    return A, pi, means, covariances

def train_and_evaluate_continue_hmm(X_train, X_test, y_train, y_test, class_names, num_states=5,
                                    n_loop=30, tol=1e-3):
    print("\n--- Huấn luyện 10 mô hình continueHMM ---")
    models = []
    for cls_id, cls_name in enumerate(class_names):
        seq_list = [X_train[i] for i, y in enumerate(y_train) if y == cls_id]
        A, pi, means, covs = _init_params(num_states, seq_list)
        model = continueHMM(A=A, means=means, covariances=covs, pi=pi).fit(seq_list, n_loop=n_loop, bound_learning=tol)
        models.append(model)
        print(f"Done: {cls_name}")

    print("\n--- Đánh giá ---")
    y_pred = []
    for seq in X_test:
        scores = [m.forward(seq)[0] for m in models]  # log_prob từ forward
        y_pred.append(int(np.argmax(scores)))

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (continueHMM)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return models, y_pred, acc









from hmmlearn import hmm
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
        model = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=100)
        
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
    return hmm_models, y_pred, accuracy