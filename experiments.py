"""Experiment protocols: e1-search, pipeline, followup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_config
from data import CLASS_NAMES, create_splits, load_splits
from metrics import evaluate_split
from train import run_method, train_supervised
from utils import load_checkpoint, save_checkpoint, set_seed, setup_device


def _ensure_splits(cfg: dict) -> dict:
    if (Path(cfg["output_dir"]) / "splits.json").exists():
        return load_splits(cfg["output_dir"])
    return create_splits(cfg)


def _eval(model, splits, keys, cfg, device):
    amp = cfg.get("amp", False)
    out = {}
    for key in keys:
        if splits.get(key):
            m = evaluate_split(model, splits[key], cfg, device, amp=amp)
            out[key] = m
            parts = [c + "=" + str(round(m["IoU_" + c], 3)) for c in CLASS_NAMES]
            print(" ", key, "mIoU", round(m["mIoU"], 4), "wIoU", round(m["mIoU_weighted"], 4), "|", " ".join(parts))
    return out


def _save(summary, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved summary to", path)


E1_CANDIDATES = [
    {"name": "E1A_base", "lr_encoder": 1e-5, "lr_decoder": 1e-4, "weight_decay": 1e-4},
    {"name": "E1B_low_lr", "lr_encoder": 5e-6, "lr_decoder": 5e-5, "weight_decay": 1e-4},
    {"name": "E1C_low_wd", "lr_encoder": 1e-5, "lr_decoder": 1e-4, "weight_decay": 1e-5},
    {"name": "E1D_higher_dec", "lr_encoder": 1e-5, "lr_decoder": 2e-4, "weight_decay": 1e-4},
]
E1_DEFAULT = ["E1A_base", "E1B_low_lr", "E1D_higher_dec"]


def cmd_e1_search(args):
    cfg = get_config("followup", data_root=args.data_root, output_dir=args.output_dir)
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = _ensure_splits(cfg)
    out_dir = Path(cfg["output_dir"]) / "e1_search"

    wanted = args.candidates or E1_DEFAULT
    summary = {"candidates": [], "best": None, "baseline_mer_test": None}
    best = (-1.0, "", "")

    for cand in [c for c in E1_CANDIDATES if c["name"] in wanted]:
        name = cand["name"]
        print("\n==========", name, "==========")
        cfg_c = {**cfg, **{k: v for k, v in cand.items() if k != "name"}}
        model, _ = train_supervised(splits["msl_train"], splits["msl_val"], cfg_c, device)
        ckpt = out_dir / (name + ".pt")
        save_checkpoint(model, str(ckpt), extra={"hparams": cand})

        msl_val = evaluate_split(model, splits["msl_val"], cfg_c, device, amp=cfg.get("amp", False))
        mer_val = evaluate_split(model, splits["mer_val"], cfg_c, device, amp=cfg.get("amp", False))
        print(" ", name, "msl_val", round(msl_val["mIoU"], 4), "mer_val", round(mer_val["mIoU"], 4))
        summary["candidates"].append(
            {"name": name, "ckpt": str(ckpt), "msl_val_mIoU": msl_val["mIoU"], "mer_val_mIoU": mer_val["mIoU"]}
        )
        if msl_val["mIoU"] > best[0]:
            best = (msl_val["mIoU"], name, str(ckpt))

    summary["candidates"].sort(key=lambda r: r["msl_val_mIoU"], reverse=True)
    summary["best"] = {"name": best[1], "ckpt": best[2], "msl_val_mIoU": best[0]}
    print("\nbest by msl_val:", best[1], round(best[0], 4))

    if best[2]:
        print("baseline test of winner:")
        m = load_checkpoint(best[2], cfg["num_classes"], device)
        summary["baseline_mer_test"] = _eval(m, splits, ["mer_test"], cfg, device)
    _save(summary, out_dir / "e1_hparam_summary.json")
    print("winner checkpoint:", best[2])


def cmd_pipeline(args):
    cfg = get_config("budget", data_root=args.data_root, output_dir=args.output_dir)
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = _ensure_splits(cfg)
    run = Path(cfg["output_dir"]) / "pipeline"
    summary = {"experiments": {}, "final_test": {}}

    print("\n========== E1: supervised pretraining ==========")
    model, hist = train_supervised(splits["msl_train"], splits["msl_val"], cfg, device)
    e1_ckpt = run / "E1.pt"
    save_checkpoint(model, str(e1_ckpt))
    summary["experiments"]["E1"] = {"best_msl_val_mIoU": hist["best_miou"], "ckpt": str(e1_ckpt)}

    print("\n========== E2: baseline (val) ==========")
    summary["experiments"]["E2_baseline_val"] = _eval(model, splits, ["msl_val", "mer_val", "m2020_val"], cfg, device)

    ckpts = {"baseline": str(e1_ckpt)}
    stages = [
        ("coral", "coral"),
        ("mmd", "mmd"),
        ("dann", "dann"),
        ("adda", "adda"),
        ("finetune", "finetune"),
        ("pseudolabel", "pseudolabel"),
        ("mean_teacher", "mean_teacher"),
    ]
    for name, method in stages:
        print("\n==========", method, "(from E1) ==========")
        if method == "adda":
            cfg_m = get_config("adda", **{k: cfg[k] for k in ("data_root", "output_dir")})
        elif method in ("finetune", "pseudolabel", "mean_teacher"):
            cfg_m = get_config("followup", **{k: cfg[k] for k in ("data_root", "output_dir")})
        else:
            cfg_m = cfg
        m, h = run_method(method, splits, cfg_m, device, init_ckpt=str(e1_ckpt))
        ckpt = run / (name + ".pt")
        save_checkpoint(m, str(ckpt))
        ckpts[name] = str(ckpt)
        summary["experiments"][name] = {"best_mer_val_mIoU": h["best_miou"], "ckpt": str(ckpt)}

    print("\n========== FINAL TEST (once) ==========")
    for name, ckpt in ckpts.items():
        if not Path(ckpt).exists():
            continue
        print(" --", name, "--")
        m = load_checkpoint(ckpt, cfg["num_classes"], device)
        summary["final_test"][name] = _eval(m, splits, ["msl_test", "mer_test", "m2020_val"], cfg, device)

    _save(summary, run / "summary.json")


FOLLOWUP = [
    ("F0_finetune", "finetune", {}),
    ("F1_pseudolabel", "pseudolabel", {}),
    ("F2_mean_teacher", "mean_teacher", {}),
]


def cmd_followup(args):
    cfg = get_config("followup", data_root=args.data_root, output_dir=args.output_dir)
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = _ensure_splits(cfg)
    run = Path(cfg["output_dir"]) / "followup"

    init_ckpt = args.init_ckpt
    if not (init_ckpt and Path(init_ckpt).exists()):
        raise FileNotFoundError("--init-ckpt (E1 winner) is required and must exist")

    summary = {"init_ckpt": init_ckpt, "experiments": {}, "ranking": [], "finalists": [], "final_test": {}}
    for name, method, overrides in FOLLOWUP:
        print("\n==========", name, method, "==========")
        cfg_e = {**cfg, **overrides}
        model, hist = run_method(method, splits, cfg_e, device, init_ckpt=init_ckpt)
        ckpt = run / (name + ".pt")
        save_checkpoint(model, str(ckpt))
        mer_val = evaluate_split(model, splits["mer_val"], cfg, device, amp=cfg.get("amp", False))
        summary["experiments"][name] = {"ckpt": str(ckpt), "mer_val_mIoU": mer_val["mIoU"]}
        print(" ", name, "mer_val mIoU", round(mer_val["mIoU"], 4))

    ranking = sorted(summary["experiments"].items(), key=lambda kv: kv[1]["mer_val_mIoU"], reverse=True)
    summary["ranking"] = [{"name": n, "mer_val_mIoU": e["mer_val_mIoU"]} for n, e in ranking]
    finalists = [n for n, _ in ranking[: max(1, args.topk)]]
    summary["finalists"] = finalists
    print("\n========== RANKING (mer_val) ==========")
    for row in summary["ranking"]:
        print(" ", row["name"], "mer_val", round(row["mer_val_mIoU"], 4))

    print("\n========== FINAL TEST (finalists only) ==========")
    for name in finalists:
        print(" --", name, "--")
        m = load_checkpoint(summary["experiments"][name]["ckpt"], cfg["num_classes"], device)
        summary["final_test"][name] = _eval(m, splits, ["mer_test", "msl_test", "m2020_val"], cfg, device)

    _save(summary, run / "followup_summary.json")


def main():
    p = argparse.ArgumentParser(description="Reproduce the report experiment protocols.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("e1-search")
    s.add_argument("--candidates", nargs="*", default=None)
    s.set_defaults(func=cmd_e1_search)

    s = sub.add_parser("pipeline")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("followup")
    s.add_argument("--init-ckpt", required=True)
    s.add_argument("--topk", type=int, default=2)
    s.set_defaults(func=cmd_followup)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
