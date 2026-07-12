"""IoU metrics from a confusion matrix."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data import CLASS_NAMES, SegmentationDataset


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    mask = target != ignore_index
    pred, target = pred[mask].view(-1), target[mask].view(-1)
    idx = num_classes * target + pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes).float()


def compute_metrics(cm: torch.Tensor) -> dict:
    tp = torch.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / union.clamp(min=1.0)
    valid = union > 0

    total = cm.sum()
    support = cm.sum(0)
    metrics = {
        "mIoU": iou[valid].mean().item() if valid.any() else 0.0,
        "pixel_accuracy": (tp.sum() / total.clamp(min=1.0)).item() if total > 0 else 0.0,
    }
    freq = valid[:3]
    metrics["mIoU_frequent"] = iou[:3][freq].mean().item() if freq.any() else 0.0
    if valid.any():
        weights = support[valid] / support[valid].sum().clamp(min=1.0)
        metrics["mIoU_weighted"] = (weights * iou[valid]).sum().item()
    else:
        metrics["mIoU_weighted"] = 0.0
    metrics["per_class"] = {name: iou[i].item() for i, name in enumerate(CLASS_NAMES[: cm.shape[0]])}
    for name, value in metrics["per_class"].items():
        metrics[f"IoU_{name}"] = value
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int, ignore_index: int = 255, amp: bool = False) -> dict:
    from utils import amp_autocast

    model.eval()
    cm = torch.zeros(num_classes, num_classes, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_autocast(device, amp):
            logits = model(images)
        cm += confusion_matrix(logits.argmax(1), labels, num_classes, ignore_index)
    return compute_metrics(cm.cpu())


def evaluate_split(model, samples, cfg, device, amp: bool = False) -> dict:
    from utils import dataloader_kwargs

    ds = SegmentationDataset(samples, cfg["image_size"], cfg["ignore_index"])
    loader = DataLoader(ds, shuffle=False, **dataloader_kwargs(cfg))
    return evaluate_model(model, loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
