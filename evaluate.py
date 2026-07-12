"""Evaluate a checkpoint on one or more splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_config
from data import CLASS_NAMES, load_splits
from metrics import evaluate_split
from utils import load_checkpoint, set_seed, setup_device


def main():
    p = argparse.ArgumentParser(description="Evaluate a Mars segmentation checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", nargs="+", default=["mer_test"])
    p.add_argument("--preset", default="budget", choices=["budget", "followup", "adda"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--save", default=None)
    args = p.parse_args()

    cfg = get_config(args.preset, data_root=args.data_root, output_dir=args.output_dir)
    set_seed(cfg["seed"])
    device = setup_device(cfg)
    splits = load_splits(cfg["output_dir"])
    model = load_checkpoint(args.checkpoint, cfg["num_classes"], device)
    amp = cfg.get("amp", False)

    results = {}
    for name in args.splits:
        if not splits.get(name):
            print("skip empty split:", name)
            continue
        m = evaluate_split(model, splits[name], cfg, device, amp=amp)
        results[name] = m
        parts = [c + "=" + str(round(m["IoU_" + c], 3)) for c in CLASS_NAMES]
        print(name, "mIoU", round(m["mIoU"], 4), "wIoU", round(m["mIoU_weighted"], 4),
              "freq", round(m["mIoU_frequent"], 4), "acc", round(m["pixel_accuracy"], 4), "|", " ".join(parts))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print("saved metrics to", args.save)


if __name__ == "__main__":
    main()
