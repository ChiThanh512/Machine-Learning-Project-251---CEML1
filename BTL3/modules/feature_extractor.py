import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from modules.data_utils import FileListDS, fe_tfm

def build_feature_extractor(backbone, device):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
    elif backbone == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        dim = m.heads.head.in_features
        m.heads.head = nn.Identity()
    else:
        raise ValueError("Unknown backbone:", backbone)
    return m.to(device).eval(), dim


@torch.no_grad()
def extract_numpy_features(paths, labels, backbone, device, batch=64, num_workers=2):
    ds = FileListDS(paths, labels, fe_tfm)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    fe, dim = build_feature_extractor(backbone, device)
    feats = np.zeros((len(ds), dim), dtype=np.float32)
    lbs   = np.zeros((len(ds),), dtype=np.int64)
    idx = 0
    for imgs, gts in dl:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(True):
            v = fe(imgs)
        v = v.detach().float().cpu().numpy()
        b = v.shape[0]
        feats[idx:idx+b] = v
        lbs[idx:idx+b]   = gts.numpy()
        idx += b
    return feats, lbs

