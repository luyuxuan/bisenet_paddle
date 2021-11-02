import torch
import torch.nn.functional as F
import numpy as np

def miou(logits,label,n_classes):
    N,H,W = label.shape
    news = []
    hist = torch.zeros(n_classes, n_classes)
    probs = torch.zeros((N, n_classes, H, W), dtype=torch.float32)
    logits = F.interpolate(logits, size=(H,W),
                        mode='bilinear', align_corners=True)
    probs += torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    keep = label != 255
    # print(keep.shape)
    # print(preds.shape)
    hist += torch.bincount(label[keep] * n_classes + preds[keep],minlength=n_classes ** 2).view(n_classes, n_classes)
    ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
    # print(ious)
    ious = np.array(ious)
    for i in range(len(ious)):
        # print(ious[i].dtype)
        if ious[i]>0:
            news.append(ious[i])
        else:
            news.append(float(0))
    # ious = np.array(ious)
    # print("iou_class",news)
    miou = np.mean(news)
    # print(miou)
    return format(miou,'.8f')