"""Training loops and CLI for all methods."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptation import (
    DomainDiscriminator,
    GradientReversal,
    adversarial_encoder_loss,
    clone_teacher,
    coral_loss,
    domain_discriminator_loss,
    feat_map,
    gap_features,
    grl_lambda,
    mmd_loss,
    update_ema,
)
from data import SegmentationDataset, UnlabeledDataset
from metrics import evaluate_model
from model import NavSegmenter
from utils import amp_autocast, dataloader_kwargs, grad_scaler, load_checkpoint, save_checkpoint, use_amp


def _loader(ds, cfg, shuffle):
    return DataLoader(ds, shuffle=shuffle, **dataloader_kwargs(cfg))


def _seg_loader(samples, cfg, shuffle):
    return _loader(SegmentationDataset(samples, cfg["image_size"], cfg["ignore_index"]), cfg, shuffle)


def _unlabeled_loader(samples, cfg, shuffle):
    return _loader(UnlabeledDataset(samples, cfg["image_size"]), cfg, shuffle)


def _uda_lr(cfg: dict) -> float:
    return cfg.get("lr_decoder", cfg["lr"]) * cfg.get("uda_lr_factor", 0.1)


def _confidence_threshold(cfg: dict, epoch: int, total_epochs: int) -> float:
    start = cfg["uda_confidence_threshold"]
    end = cfg.get("uda_confidence_threshold_end")
    if end is None or total_epochs <= 1:
        return start
    ramp = cfg.get("uda_confidence_ramp_epochs") or total_epochs
    t = min(1.0, epoch / max(ramp - 1, 1))
    return start + (end - start) * t


def _freeze_encoder(model: NavSegmenter, freeze: bool) -> None:
    for mod in (model.stem, model.pool, model.enc1, model.enc2, model.enc3, model.enc4):
        for p in mod.parameters():
            p.requires_grad = not freeze


def _freeze_model(model: NavSegmenter, freeze: bool) -> None:
    for p in model.parameters():
        p.requires_grad = not freeze


def _encoder_params(model: NavSegmenter):
    return [p for m in (model.stem, model.pool, model.enc1, model.enc2, model.enc3, model.enc4) for p in m.parameters()]


def _decoder_params(model: NavSegmenter):
    return [p for m in (model.up3, model.up2, model.up1, model.up0, model.head) for p in m.parameters()]


@torch.no_grad()
def generate_pseudo_labels(model, images, threshold, ignore_index=255, amp=False) -> torch.Tensor:
    """Per-pixel pseudo labels; low-confidence pixels set to ignore."""
    model.eval()
    with amp_autocast(images.device, amp):
        logits = model(images)
    conf, pseudo = torch.softmax(logits.float(), dim=1).max(dim=1)
    pseudo = pseudo.clone()
    pseudo[conf < threshold] = ignore_index
    return pseudo


def _best_state(model, val_miou, best):
    if val_miou > best[0]:
        return val_miou, deepcopy(model.state_dict())
    return best


def train_supervised(train_samples, val_samples, cfg, device, model=None):
    """MSL pretraining; best checkpoint by msl_val mIoU."""
    model = (model or NavSegmenter(cfg["num_classes"], pretrained=True)).to(device)
    enc = [p for m in (model.stem, model.pool, model.enc1, model.enc2, model.enc3, model.enc4) for p in m.parameters()]
    dec = [p for m in (model.up3, model.up2, model.up1, model.up0, model.head) for p in m.parameters()]
    optim = torch.optim.AdamW(
        [{"params": enc, "lr": cfg["lr_encoder"]}, {"params": dec, "lr": cfg["lr_decoder"]}],
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    amp = use_amp(cfg)
    scaler = grad_scaler(amp)

    train_loader = _seg_loader(train_samples, cfg, shuffle=True)
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)

    history = {"val_miou": []}
    best = (-1.0, None)
    patience, stale = cfg.get("early_stop_patience", 0), 0

    for epoch in range(cfg["epochs_supervised"]):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        avg_loss = total / max(len(train_loader), 1)
        print("E1 epoch", epoch + 1, "loss", round(avg_loss, 4), "val_mIoU", round(val["mIoU"], 4))

        prev = best[0]
        best = _best_state(model, val["mIoU"], best)
        if patience > 0:
            stale = 0 if val["mIoU"] > prev else stale + 1
            if stale >= patience:
                print("early stop epoch", epoch + 1, "best val mIoU", round(best[0], 4))
                break

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_feature_uda(model, method, src_samples, tgt_samples, val_samples, cfg, device):
    """CORAL, MMD, or DANN on enc4; checkpoint by mer_val mIoU."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr_decoder"] * 0.5, weight_decay=cfg["weight_decay"])
    disc = opt_disc = None
    if method == "dann":
        disc = DomainDiscriminator(512).to(device)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4)

    amp = use_amp(cfg)
    scaler = grad_scaler(amp)
    src_loader = _seg_loader(src_samples, cfg, shuffle=True)
    tgt_loader = _unlabeled_loader(tgt_samples, cfg, shuffle=True)
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)

    epochs = cfg["uda_epochs"]
    lam, sigma, lambda_max = cfg["uda_lambda"], cfg["mmd_sigma"], cfg["dann_lambda_max"]
    history = {"val_miou": []}
    best = (-1.0, None)

    for epoch in range(epochs):
        model.train()
        if disc:
            disc.train()
        tgt_iter = cycle(tgt_loader)
        grl = grl_lambda(epoch, epochs, lambda_max) if method == "dann" else 0.0

        for src_img, src_lbl in src_loader:
            tgt_img = next(tgt_iter).to(device)
            src_img, src_lbl = src_img.to(device), src_lbl.to(device)
            optim.zero_grad(set_to_none=True)
            if opt_disc:
                opt_disc.zero_grad(set_to_none=True)

            with amp_autocast(device, amp):
                e4_s, skips = model.encode(src_img)
                seg_loss = criterion(model.decode(e4_s, skips, src_img.shape[-2:]), src_lbl)
                e4_t, _ = model.encode(tgt_img)

                if method == "coral":
                    loss = seg_loss + lam * coral_loss(feat_map(e4_s), feat_map(e4_t))
                elif method == "mmd":
                    loss = seg_loss + lam * mmd_loss(feat_map(e4_s), feat_map(e4_t), sigma=sigma)
                else:
                    gap_s, gap_t = e4_s.mean(dim=(2, 3)), e4_t.mean(dim=(2, 3))
                    dom_logits = disc(torch.cat([GradientReversal.apply(gap_s, grl), GradientReversal.apply(gap_t, grl)]))
                    dom_labels = torch.cat([torch.zeros(gap_s.size(0), device=device), torch.ones(gap_t.size(0), device=device)])
                    loss = seg_loss + F.binary_cross_entropy_with_logits(dom_logits, dom_labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            if opt_disc:
                scaler.step(opt_disc)
            scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print(method.upper(), "epoch", epoch + 1, "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best_state(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_adda(model, src_samples, tgt_samples, val_samples, cfg, device):
    """Two-phase ADDA: train D frozen, then adapt encoder with seg + adv + CORAL."""
    model = model.to(device)
    disc = DomainDiscriminator(512).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    amp = use_amp(cfg)
    scaler = grad_scaler(amp)
    src_loader = _seg_loader(src_samples, cfg, shuffle=True)
    tgt_loader = _unlabeled_loader(tgt_samples, cfg, shuffle=True)
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)

    lam_seg = cfg["adda_lambda_seg"]
    lam_adv = cfg["adda_lambda_adv"]
    lam_coral = cfg["adda_lambda_coral"]
    train_decoder = cfg.get("adda_train_decoder", True)
    adv_ramp = cfg.get("adda_adv_ramp", True)

    history = {"val_miou": [], "phase1_epochs": cfg["adda_epochs_disc"], "phase2_epochs": cfg["adda_epochs_adapt"]}
    best = (-1.0, None)

    # Phase 1: train discriminator with frozen segmenter
    _freeze_model(model, True)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=cfg["adda_disc_lr"])

    for epoch in range(cfg["adda_epochs_disc"]):
        disc.train()
        model.eval()
        tgt_iter = cycle(tgt_loader)
        for src_img, _ in src_loader:
            tgt_img = next(tgt_iter).to(device)
            src_img = src_img.to(device)
            opt_disc.zero_grad(set_to_none=True)
            with torch.no_grad(), amp_autocast(device, amp):
                gap_s = gap_features(model.encode(src_img)[0])
                gap_t = gap_features(model.encode(tgt_img)[0])
            with amp_autocast(device, amp):
                loss_d = domain_discriminator_loss(disc(gap_s), True) + domain_discriminator_loss(disc(gap_t), False)
            scaler.scale(loss_d).backward()
            scaler.step(opt_disc)
            scaler.update()
        print("ADDA phase1 epoch", epoch + 1, "of", cfg["adda_epochs_disc"])

    # Phase 2: alternate D updates and encoder/decoder updates
    _freeze_model(model, False)
    enc = _encoder_params(model)
    dec = _decoder_params(model)
    opt_groups = [{"params": enc, "lr": cfg["adda_encoder_lr"]}]
    if train_decoder:
        opt_groups.append({"params": dec, "lr": cfg["adda_decoder_lr"]})
    opt_model = torch.optim.AdamW(opt_groups, weight_decay=cfg["weight_decay"])

    adapt_epochs = cfg["adda_epochs_adapt"]
    for epoch in range(adapt_epochs):
        model.train()
        disc.train()
        lam_adv_eff = lam_adv * grl_lambda(epoch, adapt_epochs, 1.0) if adv_ramp else lam_adv
        tgt_iter = cycle(tgt_loader)

        for src_img, src_lbl in src_loader:
            tgt_img = next(tgt_iter).to(device)
            src_img, src_lbl = src_img.to(device), src_lbl.to(device)

            with amp_autocast(device, amp):
                e4_s, skips = model.encode(src_img)
                e4_t, _ = model.encode(tgt_img)
                gap_s = gap_features(e4_s)
                gap_t = gap_features(e4_t)

            opt_disc.zero_grad(set_to_none=True)
            with amp_autocast(device, amp):
                loss_d = domain_discriminator_loss(disc(gap_s.detach()), True) + domain_discriminator_loss(disc(gap_t.detach()), False)
            scaler.scale(loss_d).backward()
            scaler.step(opt_disc)

            opt_model.zero_grad(set_to_none=True)
            with amp_autocast(device, amp):
                e4_s, skips = model.encode(src_img)
                e4_t, _ = model.encode(tgt_img)
                seg_loss = criterion(model.decode(e4_s, skips, src_img.shape[-2:]), src_lbl)
                gap_t = gap_features(e4_t)
                adv = adversarial_encoder_loss(disc(gap_t))
                coral = coral_loss(feat_map(e4_s), feat_map(e4_t))
                loss = lam_seg * seg_loss + lam_adv_eff * adv + lam_coral * coral
            scaler.scale(loss).backward()
            scaler.step(opt_model)
            scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("ADDA phase2 epoch", epoch + 1, "lam_adv", round(lam_adv_eff, 3), "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best_state(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_pseudolabel(model, uda_samples, val_samples, cfg, device, use_ema=False):
    """Self-training on unlabeled MER; optional EMA teacher."""
    model = model.to(device)
    teacher = clone_teacher(model).to(device) if use_ema else None
    ema_decay = cfg.get("ema_decay") or 0.99
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(model.parameters(), lr=_uda_lr(cfg), weight_decay=cfg["weight_decay"])
    amp = use_amp(cfg)
    scaler = grad_scaler(amp)

    uda_loader = _unlabeled_loader(uda_samples, cfg, shuffle=True)
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)
    epochs = cfg["epochs_uda"]
    min_px = cfg.get("uda_min_pseudo_pixels", 100)
    freeze_epochs = cfg.get("uda_freeze_encoder_epochs", 0)

    history = {"val_miou": []}
    best = (-1.0, None)
    tag = "PL-EMA" if use_ema else "PL"

    for epoch in range(epochs):
        _freeze_encoder(model, epoch < freeze_epochs)
        threshold = _confidence_threshold(cfg, epoch, epochs)
        model.train()
        for images in uda_loader:
            images = images.to(device)
            pseudo = generate_pseudo_labels(teacher or model, images, threshold, cfg["ignore_index"], amp=amp)
            if (pseudo != cfg["ignore_index"]).sum() < min_px:
                continue
            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss = criterion(model(images), pseudo)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if teacher is not None:
                update_ema(teacher, model, ema_decay)

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print(tag, "epoch", epoch + 1, "thr", round(threshold, 2), "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best_state(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_semisup(model, labeled_samples, uda_samples, val_samples, cfg, device):
    """Two-phase: labeled MER epochs, then pseudo-label epochs."""
    if not labeled_samples:
        raise ValueError("mer_labeled_adapt is empty")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr_decoder"] * cfg.get("semisup_lr_factor", 0.1),
        weight_decay=cfg["weight_decay"],
    )
    amp = use_amp(cfg)
    scaler = grad_scaler(amp)

    labeled_loader = _seg_loader(labeled_samples, cfg, shuffle=True)
    uda_loader = _unlabeled_loader(uda_samples, cfg, shuffle=True) if uda_samples else None
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)
    epochs = cfg["epochs_semisup"]
    threshold = cfg["uda_confidence_threshold"]
    lam = cfg.get("semisup_lambda_uda", 1.0)

    history = {"val_miou": []}
    best = (-1.0, None)

    for epoch in range(epochs):
        model.train()
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        if uda_loader is not None and lam > 0:
            for images in uda_loader:
                images = images.to(device)
                pseudo = generate_pseudo_labels(model, images, threshold, cfg["ignore_index"], amp=amp)
                if (pseudo != cfg["ignore_index"]).sum() < 100:
                    continue
                optim.zero_grad(set_to_none=True)
                with amp_autocast(images.device, amp):
                    loss = lam * criterion(model(images), pseudo)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("Semisup epoch", epoch + 1, "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best_state(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_semisup_joint(model, labeled_samples, uda_samples, val_samples, cfg, device):
    """Labeled + pseudo loss in one step; optional EMA teacher."""
    if not labeled_samples:
        raise ValueError("mer_labeled_adapt is empty")
    model = model.to(device)
    ema_decay = cfg.get("ema_decay")
    teacher = clone_teacher(model).to(device) if ema_decay else None
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(model.parameters(), lr=_uda_lr(cfg), weight_decay=cfg["weight_decay"])
    amp = use_amp(cfg)
    scaler = grad_scaler(amp)

    labeled_loader = _seg_loader(labeled_samples, cfg, shuffle=True)
    uda_loader = _unlabeled_loader(uda_samples, cfg, shuffle=True) if uda_samples else None
    val_loader = _seg_loader(val_samples, cfg, shuffle=False)

    epochs = cfg["epochs_semisup"]
    lambda_uda = cfg.get("semisup_lambda_uda", 1.0)
    ramp = cfg.get("semisup_lambda_ramp_epochs", 0)
    warmup = cfg.get("semisup_pseudo_warmup_epochs", 1)
    min_px = cfg.get("uda_min_pseudo_pixels", 100)

    history = {"val_miou": []}
    best = (-1.0, None)

    for epoch in range(epochs):
        lam = lambda_uda * min(1.0, (epoch + 1) / ramp) if ramp > 0 else lambda_uda
        threshold = _confidence_threshold(cfg, epoch, epochs)
        model.train()
        uda_iter = cycle(uda_loader) if uda_loader and lam > 0 and epoch >= warmup else None
        pseudo_model = teacher if teacher is not None else model

        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            with amp_autocast(images.device, amp):
                loss = criterion(model(images), labels)
                if uda_iter is not None:
                    uda_img = next(uda_iter).to(device)
                    pseudo = generate_pseudo_labels(pseudo_model, uda_img, threshold, cfg["ignore_index"], amp=amp)
                    if (pseudo != cfg["ignore_index"]).sum() >= min_px:
                        loss = loss + lam * criterion(model(uda_img), pseudo)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if teacher is not None:
                update_ema(teacher, model, ema_decay)

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("SS-joint epoch", epoch + 1, "lam", round(lam, 2), "thr", round(threshold, 2),
              "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best_state(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


def train_combo(model, labeled_samples, uda_samples, val_samples, cfg, device):
    """Joint semi-supervision plus optional extra pseudo-label polish."""
    joint_cfg = {**cfg, "ema_decay": cfg.get("ema_decay", 0.99)}
    model, hist = train_semisup_joint(model, labeled_samples, uda_samples, val_samples, joint_cfg, device)

    extra = cfg.get("combo_extra_pseudo_epochs", 0)
    if extra > 0 and uda_samples:
        polish_cfg = {
            **cfg,
            "epochs_uda": extra,
            "ema_decay": cfg.get("ema_decay", 0.99),
            "uda_lr_factor": cfg.get("combo_extra_pseudo_lr_factor", 0.05),
        }
        model, hist_polish = train_pseudolabel(model, uda_samples, val_samples, polish_cfg, device, use_ema=True)
        hist["best_miou"] = max(hist["best_miou"], hist_polish["best_miou"])
    return model, hist


FEATURE_METHODS = ("coral", "mmd", "dann")
NEEDS_INIT = ("coral", "mmd", "dann", "adda", "pseudolabel", "pseudolabel_ema", "semisup", "semisup_joint", "combo")


def run_method(method, splits, cfg, device, init_ckpt=None):
    """Dispatch to the training loop for method."""
    if method == "supervised":
        return train_supervised(splits["msl_train"], splits["msl_val"], cfg, device)

    model = load_checkpoint(init_ckpt, cfg["num_classes"], device)
    if method in FEATURE_METHODS:
        return train_feature_uda(model, method, splits["msl_train"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "adda":
        return train_adda(model, splits["msl_train"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "pseudolabel":
        return train_pseudolabel(model, splits["mer_uda"], splits["mer_val"], cfg, device, use_ema=False)
    if method == "pseudolabel_ema":
        return train_pseudolabel(model, splits["mer_uda"], splits["mer_val"], cfg, device, use_ema=True)
    if method == "semisup":
        return train_semisup(model, splits["mer_labeled_adapt"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "semisup_joint":
        return train_semisup_joint(model, splits["mer_labeled_adapt"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "combo":
        return train_combo(model, splits["mer_labeled_adapt"], splits["mer_uda"], splits["mer_val"], cfg, device)
    raise ValueError("unknown method:", method)


def main():
    from config import get_config
    from data import load_splits
    from utils import set_seed, setup_device

    p = argparse.ArgumentParser(description="Train a single Mars segmentation method.")
    p.add_argument("--method", required=True,
                   choices=["supervised", *FEATURE_METHODS, "adda", "pseudolabel", "pseudolabel_ema",
                            "semisup", "semisup_joint", "combo"])
    p.add_argument("--preset", default="budget", choices=["budget", "followup", "adda"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--out-ckpt", default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.preset, data_root=args.data_root, output_dir=args.output_dir)
    if args.epochs is not None:
        cfg["epochs_supervised"] = cfg["epochs_uda"] = cfg["epochs_semisup"] = cfg["uda_epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = load_splits(cfg["output_dir"])

    if args.method in NEEDS_INIT and not (args.init_ckpt and Path(args.init_ckpt).exists()):
        raise FileNotFoundError("--init-ckpt required for method", args.method)

    model, hist = run_method(args.method, splits, cfg, device, args.init_ckpt)
    out = args.out_ckpt or str(Path(cfg["output_dir"]) / (args.method + ".pt"))
    save_checkpoint(model, out, extra={"method": args.method, "history": hist})
    print("saved", args.method, "to", out, "best val mIoU", round(hist["best_miou"], 4))


if __name__ == "__main__":
    main()
