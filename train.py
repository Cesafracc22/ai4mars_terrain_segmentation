"""Training loops and CLI (report methods only)."""

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
    consistency_mse,
    coral_loss,
    domain_discriminator_loss,
    feat_map,
    gap_features,
    grl_lambda,
    mmd_loss,
    sigmoid_rampup,
    update_ema,
)
from data import SegmentationDataset, UnlabeledDataset
from metrics import evaluate_model
from model import NavSegmenter
from utils import amp_autocast, dataloader_kwargs, grad_scaler, load_checkpoint, save_checkpoint, use_amp

METHODS = ("supervised", "coral", "mmd", "dann", "adda", "pseudolabel", "mean_teacher")
NEEDS_INIT = ("coral", "mmd", "dann", "adda", "pseudolabel", "mean_teacher")


def _seg_loader(samples, cfg, shuffle):
    return DataLoader(
        SegmentationDataset(samples, cfg["image_size"], cfg["ignore_index"]),
        shuffle=shuffle,
        **dataloader_kwargs(cfg),
    )


def _uda_loader(samples, cfg, shuffle):
    return DataLoader(UnlabeledDataset(samples, cfg["image_size"]), shuffle=shuffle, **dataloader_kwargs(cfg))


def _lr(cfg):
    return cfg["lr_decoder"] * cfg.get("uda_lr_factor", 0.1)


def _best(model, miou, best):
    return (miou, deepcopy(model.state_dict())) if miou > best[0] else best


@torch.no_grad()
def hard_pseudo(model, images, threshold, ignore=255, amp=False):
    """Argmax labels; low-confidence pixels → ignore."""
    model.eval()
    with amp_autocast(images.device, amp):
        conf, pseudo = torch.softmax(model(images).float(), dim=1).max(dim=1)
    pseudo = pseudo.clone()
    pseudo[conf < threshold] = ignore
    return pseudo


def train_supervised(train_samples, val_samples, cfg, device, model=None):
    """MSL pretrain; keep best msl_val mIoU."""
    model = (model or NavSegmenter(cfg["num_classes"], pretrained=True)).to(device)
    enc = [p for m in (model.stem, model.pool, model.enc1, model.enc2, model.enc3, model.enc4) for p in m.parameters()]
    dec = [p for m in (model.up3, model.up2, model.up1, model.up0, model.head) for p in m.parameters()]
    optim = torch.optim.AdamW(
        [{"params": enc, "lr": cfg["lr_encoder"]}, {"params": dec, "lr": cfg["lr_decoder"]}],
        weight_decay=cfg["weight_decay"],
    )
    ce = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    amp, scaler = use_amp(cfg), grad_scaler(use_amp(cfg))
    train_loader = _seg_loader(train_samples, cfg, True)
    val_loader = _seg_loader(val_samples, cfg, False)
    history, best, stale = {"val_miou": []}, (-1.0, None), 0
    patience = cfg.get("early_stop_patience", 0)

    for epoch in range(cfg["epochs_supervised"]):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss = ce(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("E1 epoch", epoch + 1, "loss", round(total / max(len(train_loader), 1), 4),
              "val_mIoU", round(val["mIoU"], 4))
        prev = best[0]
        best = _best(model, val["mIoU"], best)
        if patience > 0:
            stale = 0 if val["mIoU"] > prev else stale + 1
            if stale >= patience:
                print("early stop epoch", epoch + 1, "best", round(best[0], 4))
                break

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


# feature uda coral mmd dann on enc4
# dann path imported and adapted from https://github.com/fungtion/DANN
# ganin et al grl and domain discriminator classification to dense segmentation
def train_feature_uda(model, method, src_samples, tgt_samples, val_samples, cfg, device):
    """CORAL / MMD / DANN on enc4."""
    model = model.to(device)
    ce = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr_decoder"] * 0.5, weight_decay=cfg["weight_decay"])
    disc = opt_disc = None
    if method == "dann":
        disc = DomainDiscriminator(512).to(device)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4)

    amp, scaler = use_amp(cfg), grad_scaler(use_amp(cfg))
    src_loader = _seg_loader(src_samples, cfg, True)
    tgt_loader = _uda_loader(tgt_samples, cfg, True)
    val_loader = _seg_loader(val_samples, cfg, False)
    epochs = cfg["uda_epochs"]
    lam, sigma, lmax = cfg["uda_lambda"], cfg["mmd_sigma"], cfg["dann_lambda_max"]
    history, best = {"val_miou": []}, (-1.0, None)

    for epoch in range(epochs):
        model.train()
        if disc:
            disc.train()
        tgt_iter = cycle(tgt_loader)
        grl = grl_lambda(epoch, epochs, lmax) if method == "dann" else 0.0

        for src_img, src_lbl in src_loader:
            tgt_img = next(tgt_iter).to(device)
            src_img, src_lbl = src_img.to(device), src_lbl.to(device)
            optim.zero_grad(set_to_none=True)
            if opt_disc:
                opt_disc.zero_grad(set_to_none=True)

            with amp_autocast(device, amp):
                e4_s, skips = model.encode(src_img)
                seg = ce(model.decode(e4_s, skips, src_img.shape[-2:]), src_lbl)
                e4_t, _ = model.encode(tgt_img)
                if method == "coral":
                    loss = seg + lam * coral_loss(feat_map(e4_s), feat_map(e4_t))
                elif method == "mmd":
                    loss = seg + lam * mmd_loss(feat_map(e4_s), feat_map(e4_t), sigma=sigma)
                else:
                    gap_s, gap_t = e4_s.mean(dim=(2, 3)), e4_t.mean(dim=(2, 3))
                    logits = disc(torch.cat([
                        GradientReversal.apply(gap_s, grl),
                        GradientReversal.apply(gap_t, grl),
                    ]))
                    labels = torch.cat([
                        torch.zeros(gap_s.size(0), device=device),
                        torch.ones(gap_t.size(0), device=device),
                    ])
                    loss = seg + F.binary_cross_entropy_with_logits(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            if opt_disc:
                scaler.step(opt_disc)
            scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print(method.upper(), "epoch", epoch + 1, "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


# adda imported and adapted from https://github.com/ayushtues/ADDA_pytorch
# tzeng et al shared encoder variant for u-net and dense coral on enc4
def train_adda(model, src_samples, tgt_samples, val_samples, cfg, device):
    """ADDA: train D (frozen seg), then adapt encoder with seg + adv + CORAL."""
    model = model.to(device)
    disc = DomainDiscriminator(512).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    amp, scaler = use_amp(cfg), grad_scaler(use_amp(cfg))
    src_loader = _seg_loader(src_samples, cfg, True)
    tgt_loader = _uda_loader(tgt_samples, cfg, True)
    val_loader = _seg_loader(val_samples, cfg, False)
    history, best = {"val_miou": []}, (-1.0, None)

    for p in model.parameters():
        p.requires_grad = False
    opt_disc = torch.optim.Adam(disc.parameters(), lr=cfg["adda_disc_lr"])

    for epoch in range(cfg["adda_epochs_disc"]):
        disc.train()
        model.eval()
        tgt_iter = cycle(tgt_loader)
        for src_img, _ in src_loader:
            src_img = src_img.to(device)
            tgt_img = next(tgt_iter).to(device)
            opt_disc.zero_grad(set_to_none=True)
            with torch.no_grad(), amp_autocast(device, amp):
                gap_s = gap_features(model.encode(src_img)[0])
                gap_t = gap_features(model.encode(tgt_img)[0])
            with amp_autocast(device, amp):
                loss_d = domain_discriminator_loss(disc(gap_s), True) + domain_discriminator_loss(disc(gap_t), False)
            scaler.scale(loss_d).backward()
            scaler.step(opt_disc)
            scaler.update()
        print("ADDA phase1 epoch", epoch + 1)

    for p in model.parameters():
        p.requires_grad = True
    enc = [p for m in (model.stem, model.pool, model.enc1, model.enc2, model.enc3, model.enc4) for p in m.parameters()]
    dec = [p for m in (model.up3, model.up2, model.up1, model.up0, model.head) for p in m.parameters()]
    opt_model = torch.optim.AdamW(
        [{"params": enc, "lr": cfg["adda_encoder_lr"]}, {"params": dec, "lr": cfg["adda_decoder_lr"]}],
        weight_decay=cfg["weight_decay"],
    )
    adapt_epochs = cfg["adda_epochs_adapt"]
    lam_seg, lam_adv, lam_coral = cfg["adda_lambda_seg"], cfg["adda_lambda_adv"], cfg["adda_lambda_coral"]

    for epoch in range(adapt_epochs):
        model.train()
        disc.train()
        lam_adv_eff = lam_adv * grl_lambda(epoch, adapt_epochs, 1.0) if cfg.get("adda_adv_ramp", True) else lam_adv
        tgt_iter = cycle(tgt_loader)

        for src_img, src_lbl in src_loader:
            src_img, src_lbl = src_img.to(device), src_lbl.to(device)
            tgt_img = next(tgt_iter).to(device)

            with amp_autocast(device, amp):
                e4_s, _ = model.encode(src_img)
                e4_t, _ = model.encode(tgt_img)
                gap_s, gap_t = gap_features(e4_s), gap_features(e4_t)

            opt_disc.zero_grad(set_to_none=True)
            with amp_autocast(device, amp):
                loss_d = domain_discriminator_loss(disc(gap_s.detach()), True) + domain_discriminator_loss(
                    disc(gap_t.detach()), False
                )
            scaler.scale(loss_d).backward()
            scaler.step(opt_disc)

            opt_model.zero_grad(set_to_none=True)
            with amp_autocast(device, amp):
                e4_s, skips = model.encode(src_img)
                e4_t, _ = model.encode(tgt_img)
                seg = ce(model.decode(e4_s, skips, src_img.shape[-2:]), src_lbl)
                adv = adversarial_encoder_loss(disc(gap_features(e4_t)))
                coral = coral_loss(feat_map(e4_s), feat_map(e4_t))
                loss = lam_seg * seg + lam_adv_eff * adv + lam_coral * coral
            scaler.scale(loss).backward()
            scaler.step(opt_model)
            scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], cfg["ignore_index"], amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("ADDA phase2 epoch", epoch + 1, "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


# pseudo labeling imported and adapted from
# https://github.com/iBelieveCJM/pseudo_label-pytorch
# lee 2013 hard labels and confidence ignore for dense prediction
def train_pseudolabel(model, labeled_samples, uda_samples, val_samples, cfg, device):
    """Lee PL: CE(labeled MER) + α·CE(hard unlabeled masks)."""
    model = model.to(device)
    ce = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(model.parameters(), lr=_lr(cfg), weight_decay=cfg["weight_decay"])
    amp, scaler = use_amp(cfg), grad_scaler(use_amp(cfg))
    lab_loader = _seg_loader(labeled_samples, cfg, True)
    uda_loader = _uda_loader(uda_samples, cfg, True)
    val_loader = _seg_loader(val_samples, cfg, False)

    epochs = cfg["epochs_semisup"]
    lambda_u = cfg.get("semisup_lambda_uda", 1.0)
    ramp = cfg.get("semisup_lambda_ramp_epochs", 0)
    thr = cfg["uda_confidence_threshold"]
    min_px = cfg.get("uda_min_pseudo_pixels", 100)
    ignore = cfg["ignore_index"]
    history, best = {"val_miou": []}, (-1.0, None)

    for epoch in range(epochs):
        alpha = lambda_u * min(1.0, (epoch + 1) / ramp) if ramp > 0 else lambda_u
        model.train()
        uda_iter = cycle(uda_loader)
        for images, labels in lab_loader:
            images, labels = images.to(device), labels.to(device)
            uda_img = next(uda_iter).to(device)
            pseudo = hard_pseudo(model, uda_img, thr, ignore, amp)

            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss = ce(model(images), labels)
                if alpha > 0 and (pseudo != ignore).sum() >= min_px:
                    loss = loss + alpha * ce(model(uda_img), pseudo)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        val = evaluate_model(model, val_loader, device, cfg["num_classes"], ignore, amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("PL epoch", epoch + 1, "α", round(alpha, 2), "mer_val_mIoU", round(val["mIoU"], 4))
        best = _best(model, val["mIoU"], best)

    if best[1] is not None:
        model.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return model, history


# mean teacher imported and adapted from https://github.com/CuriousAI/mean-teacher
# tarvainen and valpola ema teacher and mse consistency on softmax maps
def train_mean_teacher(model, labeled_samples, uda_samples, val_samples, cfg, device):
    """Mean Teacher: CE(labeled) + λ·MSE(softmax student, softmax EMA teacher)."""
    student = model.to(device)
    teacher = clone_teacher(student).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optim = torch.optim.AdamW(student.parameters(), lr=_lr(cfg), weight_decay=cfg["weight_decay"])
    amp, scaler = use_amp(cfg), grad_scaler(use_amp(cfg))
    lab_loader = _seg_loader(labeled_samples, cfg, True)
    uda_loader = _uda_loader(uda_samples, cfg, True)
    val_loader = _seg_loader(val_samples, cfg, False)

    decay = cfg.get("mt_ema_decay", 0.99)
    w_max = cfg.get("mt_consistency_weight", 1.0)
    noise = cfg.get("mt_noise_std", 0.1)
    ramp_steps = max(int(cfg.get("mt_rampup_epochs", 5) * max(len(lab_loader), 1)), 1)
    ignore = cfg["ignore_index"]
    history, best, step = {"val_miou": []}, (-1.0, None), 0

    for epoch in range(cfg["epochs_semisup"]):
        student.train()
        teacher.eval()
        uda_iter = cycle(uda_loader)
        for images, labels in lab_loader:
            images, labels = images.to(device), labels.to(device)
            uda_img = next(uda_iter).to(device)
            lam = w_max * sigmoid_rampup(step, ramp_steps)

            optim.zero_grad(set_to_none=True)
            with amp_autocast(images.device, amp):
                loss_sup = ce(student(images), labels)
                s_u = student(uda_img + torch.randn_like(uda_img) * noise)
                with torch.no_grad():
                    t_u = teacher(uda_img + torch.randn_like(uda_img) * noise)
                loss = loss_sup + lam * consistency_mse(s_u, t_u)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            update_ema(teacher, student, decay)
            step += 1

        val = evaluate_model(teacher, val_loader, device, cfg["num_classes"], ignore, amp=amp)
        history["val_miou"].append(val["mIoU"])
        print("MeanTeacher epoch", epoch + 1, "λ", round(lam, 3), "mer_val_mIoU", round(val["mIoU"], 4))
        if val["mIoU"] > best[0]:
            best = (val["mIoU"], deepcopy(teacher.state_dict()))

    if best[1] is not None:
        teacher.load_state_dict(best[1])
    history["best_miou"] = best[0]
    return teacher, history


def run_method(method, splits, cfg, device, init_ckpt=None):
    if method == "supervised":
        return train_supervised(splits["msl_train"], splits["msl_val"], cfg, device)

    model = load_checkpoint(init_ckpt, cfg["num_classes"], device)
    if method in ("coral", "mmd", "dann"):
        return train_feature_uda(model, method, splits["msl_train"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "adda":
        return train_adda(model, splits["msl_train"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "pseudolabel":
        return train_pseudolabel(model, splits["mer_labeled_adapt"], splits["mer_uda"], splits["mer_val"], cfg, device)
    if method == "mean_teacher":
        return train_mean_teacher(model, splits["mer_labeled_adapt"], splits["mer_uda"], splits["mer_val"], cfg, device)
    raise ValueError("unknown method:", method)


def main():
    from config import get_config
    from data import load_splits
    from utils import set_seed, setup_device

    p = argparse.ArgumentParser(description="Train one report method.")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--preset", default="budget", choices=["budget", "followup", "adda"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--out-ckpt", default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.preset, data_root=args.data_root, output_dir=args.output_dir)
    if args.epochs is not None:
        cfg["epochs_supervised"] = cfg["epochs_semisup"] = cfg["uda_epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = load_splits(cfg["output_dir"])

    if args.method in NEEDS_INIT and not (args.init_ckpt and Path(args.init_ckpt).exists()):
        raise FileNotFoundError("--init-ckpt required for", args.method)

    model, hist = run_method(args.method, splits, cfg, device, args.init_ckpt)
    out = args.out_ckpt or str(Path(cfg["output_dir"]) / (args.method + ".pt"))
    save_checkpoint(model, out, extra={"method": args.method, "history": hist})
    print("saved", args.method, "to", out, "best val mIoU", round(hist["best_miou"], 4))


if __name__ == "__main__":
    main()
