"""Device, seed, dataloaders, checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from model import NavSegmenter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    if cfg.get("use_cuda", True) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_device(cfg: dict) -> torch.device:
    if torch.cuda.is_available() and cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    device = get_device(cfg)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return device


def use_amp(cfg: dict) -> bool:
    return bool(cfg.get("amp", False) and torch.cuda.is_available())


def amp_autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device.type, enabled=enabled and device.type == "cuda")


def grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def dataloader_kwargs(cfg: dict) -> dict:
    kw = {"batch_size": cfg["batch_size"], "num_workers": cfg.get("num_workers", 0)}
    if cfg.get("pin_memory") and torch.cuda.is_available():
        kw["pin_memory"] = True
    if cfg.get("num_workers", 0) > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def save_checkpoint(model: NavSegmenter, path: str, extra: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, p)


def load_checkpoint(path: str, num_classes: int, device: torch.device) -> NavSegmenter:
    model = NavSegmenter(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device)
