import numpy as np
import matplotlib.pyplot as plt
import math
import torch

IM_MEAN = np.array([0.485, 0.456, 0.406])
IM_STD  = np.array([0.229, 0.224, 0.225])

def denorm_img(t):
    x = t.detach().cpu().numpy().transpose(1,2,0)
    x = x * IM_STD + IM_MEAN
    return np.clip(x, 0, 1)

def show_predictions_dl(dataset, model, class_names, device, n=8):
    model.eval()
    n = min(n, len(dataset))
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    cols = 4
    rows = int(math.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3.5*rows))
    with torch.inference_mode():
        for i,ix in enumerate(idxs, 1):
            img, y = dataset[ix]
            logits = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(probs.argmax()); conf = float(probs[pred])
            plt.subplot(rows, cols, i)
            plt.imshow(denorm_img(img))
            color = "green" if pred==y else "red"
            plt.title(f"Pred: {class_names[pred]} ({conf:.2f})\nTrue: {class_names[y]}", color=color)
            plt.axis("off")
    plt.tight_layout(); plt.show()









