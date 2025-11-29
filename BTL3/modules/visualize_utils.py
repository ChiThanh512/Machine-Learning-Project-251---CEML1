import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch
import math

def show_sample_images(X_train, y_train, n=9):
    plt.figure(figsize=(8,8))
    rows, cols = 3, 3
    for i in range(1, rows*cols+1):
        idx = random.randint(0, len(X_train)-1)
        img = Image.open(X_train[idx]).convert("RGB")
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(f"Label: {y_train[idx]}")
        plt.axis("off")
    plt.suptitle("Sample training images", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_feature_hist(features, backbone):
    plt.figure(figsize=(6,4))
    plt.hist(features.flatten(), bins=100, color='steelblue')
    plt.title(f"Phân bố giá trị đặc trưng ({backbone})")
    plt.xlabel("Giá trị đặc trưng")
    plt.ylabel("Tần suất")
    plt.show()

def plot_feature_heatmap(features, backbone, n_samples=10):
    n_samples = min(n_samples, len(features))
    plt.figure(figsize=(10,5))
    sns.heatmap(features[:n_samples], cmap="viridis", cbar=True)
    plt.title(f"Heatmap đặc trưng ({backbone}) - {n_samples} mẫu")
    plt.xlabel("Chiều các vector đặc trưng")
    plt.ylabel("Mẫu ảnh")
    plt.show()

def plot_pca_tsne(train_feat, train_lbs, backbone):
    print("PCA & t-SNE visualization:")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(train_feat)
    plt.figure(figsize=(6,5))
    for lbl in np.unique(train_lbs):
        idx = np.where(train_lbs == lbl)
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=str(lbl), alpha=0.6)
    plt.title(f"PCA 2D projection of {backbone} features")
    plt.legend(); plt.show()

    X_embedded = TSNE(n_components=2, init='pca', perplexity=30, learning_rate='auto').fit_transform(train_feat[:1000])
    plt.figure(figsize=(6,5))
    for lbl in np.unique(train_lbs[:1000]):
        idx = np.where(train_lbs[:1000] == lbl)
        plt.scatter(X_embedded[idx,0], X_embedded[idx,1], label=str(lbl), alpha=0.6)
    plt.title(f"t-SNE 2D projection ({backbone})")
    plt.legend(); plt.show()

def denorm_img(t):
    IM_MEAN = np.array([0.485, 0.456, 0.406])
    IM_STD  = np.array([0.229, 0.224, 0.225])
    x = t.detach().cpu().numpy()
    # x shape: (C,H,W)
    x = x.transpose(1,2,0) * IM_STD + IM_MEAN
    x = np.clip(x, 0, 1)
    return x

def show_predictions_dl(dataset, model, DEVICE, class_names, n=8):
    model.eval()
    n = min(n, len(dataset))
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    cols = 4
    rows = int(math.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3.5*rows))

    with torch.inference_mode():
        for i,ix in enumerate(idxs, 1):
            img, y = dataset[ix]
            logits = model(img.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(probs.argmax()); conf = float(probs[pred])

            plt.subplot(rows, cols, i)
            plt.imshow(denorm_img(img))
            title = f"Pred: {class_names[pred]} ({conf:.2f})\nTrue: {class_names[y]}"
            color = "green" if pred==y else "red"
            plt.title(title, color=color, fontsize=11)
            plt.axis("off")
    plt.tight_layout(); plt.show()












