from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def make_pipe(base_estimator, use_pca=True, pca_keep=0.95):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_keep, svd_solver="full")))
    steps.append(("clf", base_estimator))
    return SkPipeline(steps)

def evaluate_clf(name, clf, Xtr, ytr, Xva, yva, Xte, yte):
    clf.fit(Xtr, ytr)
    va_pred = clf.predict(Xva)
    te_pred = clf.predict(Xte)
    va_acc = accuracy_score(yva, va_pred); va_f1 = f1_score(yva, va_pred, average="macro")
    te_acc = accuracy_score(yte, te_pred); te_f1 = f1_score(yte, te_pred, average="macro")
    print(f"[{name}] Val Acc {va_acc:.4f} | Val F1 {va_f1:.4f} || Test Acc {te_acc:.4f} | Test F1 {te_f1:.4f}")
    return te_pred, (va_f1, te_f1)

def train_classical_models(train_feat, train_lbs, val_feat, val_lbs, test_feat, test_lbs, class_names, use_pca=True, pca_keep=0.95, seed=42):
    lr  = make_pipe(LogisticRegression(max_iter=500, C=1.0), use_pca, pca_keep)
    svm = make_pipe(SVC(C=2.0, gamma="scale", kernel="rbf"), use_pca, pca_keep)
    rf  = make_pipe(RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed), use_pca, pca_keep)
    pred_lr, sc_lr  = evaluate_clf("LogReg", lr, train_feat, train_lbs, val_feat, val_lbs, test_feat, test_lbs)
    pred_svm, sc_svm = evaluate_clf("SVM-RBF", svm, train_feat, train_lbs, val_feat, val_lbs, test_feat, test_lbs)
    pred_rf, sc_rf  = evaluate_clf("RandomForest", rf, train_feat, train_lbs, val_feat, val_lbs, test_feat, test_lbs)
    return {"LogReg": sc_lr, "SVM-RBF": sc_svm, "RandomForest": sc_rf}

def evaluate_best_model(models_scores, train_feat, train_lbs, val_feat, val_lbs, 
                        test_feat, test_lbs, class_names):

    best_name = max(models_scores, key=lambda k: models_scores[k][1])  # index 1 là F1
    best_model = models_scores[best_name][0]

    print(f"\n=> Best on VAL (macro-F1): {best_name}")

    # Huấn luyện lại mô hình trên toàn bộ train + val
    X_all = np.vstack([train_feat, val_feat])
    y_all = np.concatenate([train_lbs, val_lbs])
    best_model.fit(X_all, y_all)

    # Dự đoán trên test set
    test_pred = best_model.predict(test_feat)

    # In báo cáo
    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    report = classification_report(test_lbs, test_pred, target_names=class_names)
    print(report)

    return best_name, test_pred, report

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", normalize=False, figsize=(8,8)):

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = f"Normalized Confusion Matrix - {model_name}"
    else:
        fmt = "d"
        title = f"Confusion Matrix - {model_name}"

    # Vẽ heatmap bằng matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()

    # Gắn nhãn cho trục
    tick = np.arange(len(class_names))
    plt.xticks(tick, class_names, rotation=90)
    plt.yticks(tick, class_names)

    # Ghi giá trị lên ô
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()




