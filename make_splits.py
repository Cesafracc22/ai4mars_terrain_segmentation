"""Build splits.json from the AI4MARS dataset."""

from __future__ import annotations

import argparse

from config import get_config
from data import create_splits


def main():
    p = argparse.ArgumentParser(description="Create reproducible AI4Mars splits.")
    p.add_argument("--preset", default="budget", choices=["budget", "followup"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    cfg = get_config(args.preset, data_root=args.data_root, output_dir=args.output_dir)
    splits = create_splits(cfg)
    print("splits written to", args.output_dir + "/splits.json")
    for key in ("msl_train", "msl_val", "msl_test", "mer_test", "mer_val",
                "mer_labeled_adapt", "mer_uda", "m2020_val"):
        print(" ", key, len(splits[key]))


if __name__ == "__main__":
    main()
