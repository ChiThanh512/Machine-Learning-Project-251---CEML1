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