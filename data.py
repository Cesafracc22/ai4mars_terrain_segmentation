"""Load AI4MARS images/masks and build train/val/test splits."""

from __future__ import annotations

import csv
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_label_array(arr: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """RGB or grayscale mask to class indices 0-3; invalid -> ignore."""
    if arr.ndim == 2:
        out = arr.astype(np.int64, copy=True)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = (arr[:, :, i].astype(np.int64) for i in range(3))
        out = r.copy()
        null = (r == 255) & (g == 255) & (b == 255)
        mismatch = (r != g) | (r != b)
        out[null] = ignore_index
        out[mismatch & ~null] = ignore_index
    else:
        raise ValueError("unexpected label shape:", arr.shape)
    out[(out < 0) | (out > 3)] = ignore_index
    return out


def _resolve_msl_aux_mask(image_path: Path, kind: str) -> Path | None:
    """Find mxy (rover) or rng (>30m) aux mask for an MSL image."""
    if "images" not in image_path.parts:
        return None
    images_idx = image_path.parts.index("images")
    msl_root = Path(*image_path.parts[:images_idx])
    stem = image_path.stem
    subdirs = ("mxy",) if kind == "mxy" else ("rng-30m", "rng")
    names = (
        (stem + ".png", stem + "_mxy.png", stem + "_MXY.png")
        if kind == "mxy"
        else (stem + ".png", stem + "_rng-30m.png", stem + "_RNG.png")
    )
    for sub in subdirs:
        base = msl_root / "images" / sub
        for name in names:
            p = base / name
            if p.exists():
                return p
    return None


def apply_aux_ignore(label: np.ndarray, image_path: Path, ignore_index: int = 255) -> np.ndarray:
    """MSL only: ignore rover body and distant pixels."""
    if "/msl/" not in image_path.as_posix():
        return label
    out = label.copy()
    for kind in ("mxy", "rng"):
        aux_path = _resolve_msl_aux_mask(image_path, kind)
        if aux_path is None:
            continue
        aux = np.array(Image.open(aux_path))
        if aux.ndim == 3:
            aux = aux[:, :, 0]
        if aux.shape != label.shape:
            aux = np.array(Image.fromarray(aux).resize((label.shape[1], label.shape[0]), Image.NEAREST))
        out[aux > 0] = ignore_index
    return out


def load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(img)


def load_label(path: str, size: int, image_path: str, ignore_index: int = 255) -> torch.Tensor:
    lbl = Image.open(path)
    resized = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)(lbl)
    arr = decode_label_array(np.array(resized), ignore_index=ignore_index)
    arr = apply_aux_ignore(arr, Path(image_path), ignore_index=ignore_index)
    return torch.from_numpy(arr)


class SegmentationDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int = 512, ignore_index: int = 255):
        self.samples = samples
        self.image_size = image_size
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = load_image(s["image"], self.image_size)
        label = load_label(s["label"], self.image_size, s["image"], self.ignore_index)
        return image, label


class UnlabeledDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int = 512):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_image(self.samples[idx]["image"], self.image_size)


def _list_pairs(image_dir: Path, label_dir: Path, resolver, exts=(".JPG", ".jpg", ".jpeg", ".png")):
    if not image_dir.is_dir() or not label_dir.is_dir():
        return []
    pairs = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix not in exts:
            continue
        lbl = resolver(label_dir, img.stem)
        if lbl is not None and lbl.exists():
            pairs.append((img, lbl))
    return pairs


def _msl_dirs(root: Path):
    candidates = (
        (root / "msl/ncam/images/edr", root / "msl/ncam/labels/train",
         root / "msl/ncam/labels/test/masked-gold-min3-100agree"),
        (root / "msl/images/edr", root / "msl/labels/train",
         root / "msl/labels/test/masked-gold-min3-100agree"),
    )
    for image_dir, train_dir, test_dir in candidates:
        if image_dir.is_dir() and train_dir.is_dir():
            return image_dir, train_dir, test_dir
    return candidates[0]


def list_msl_train_pairs(root: Path):
    image_dir, train_dir, _ = _msl_dirs(root)
    return _list_pairs(image_dir, train_dir, lambda d, s: d / (s + ".png"))


def list_msl_test_pairs(root: Path):
    image_dir, _, test_dir = _msl_dirs(root)
    return _list_pairs(image_dir, test_dir, lambda d, s: d / (s + "_merged.png"))


def _mer_train_resolver(label_dir: Path, stem: str) -> Path | None:
    matches = sorted(label_dir.glob(stem + "_merged*.png"))
    return matches[-1] if matches else None


def list_mer_train_pairs(root: Path):
    return _list_pairs(root / "mer/images/eff", root / "mer/labels/train/merged-unmasked", _mer_train_resolver)


@lru_cache(maxsize=1)
def _mer_test_csv(root_str: str) -> dict:
    csv_path = Path(root_str) / "mer/labels/test/masked-gold-min3-100agree/test.csv"
    mapping: dict = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            stem, img_name = row[0].strip(), row[1].strip()
            lbl = next((p.name for p in csv_path.parent.glob(stem + "*_merged.png")), None)
            if lbl:
                mapping[stem] = (lbl, img_name)
    return mapping


def list_mer_test_pairs(root: Path):
    mapping = _mer_test_csv(str(root))
    pairs = []
    for _, (lbl_name, img_name) in mapping.items():
        img = root / "mer/images/test" / img_name
        lbl = root / "mer/labels/test/masked-gold-min3-100agree" / lbl_name
        if img.exists() and lbl.exists():
            pairs.append((img, lbl))
    return sorted(pairs, key=lambda x: x[0].name)


def list_m2020_nav_pairs(root: Path):
    img_dir, lbl_dir = root / "m2020/images/ncam", root / "m2020/labels/NAV"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return []
    pairs = []
    for lbl in sorted(lbl_dir.glob("*.png")):
        base = lbl.stem.split("_merged")[0]
        for ext in (".jpeg", ".jpg", ".JPG"):
            img = img_dir / (base + ext)
            if img.exists():
                pairs.append((img, lbl))
                break
    return pairs


def _to_records(pairs) -> list[dict]:
    return [{"image": str(img), "label": str(lbl)} for img, lbl in pairs]


def create_splits(cfg: dict) -> dict:
    """Write splits.json under output_dir."""
    root = Path(cfg["data_root"])
    rng = random.Random(cfg["seed"])

    msl_pool = list_msl_train_pairs(root)
    msl_test = list_msl_test_pairs(root)
    mer_train = list_mer_train_pairs(root)
    mer_test = list_mer_test_pairs(root)
    m2020 = list_m2020_nav_pairs(root)
    for pool in (msl_pool, msl_test, mer_train, mer_test, m2020):
        rng.shuffle(pool)

    msl_train = msl_pool[: cfg["msl_train_size"]]
    msl_val = msl_pool[cfg["msl_train_size"]: cfg["msl_train_size"] + cfg["msl_val_size"]]
    mer_adapt = mer_train[: cfg["mer_labeled_adapt_size"]]
    mer_val = mer_train[cfg["mer_labeled_adapt_size"]: cfg["mer_labeled_adapt_size"] + cfg["mer_val_size"]]

    eff = sorted({p for ext in ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG")
                  for p in (root / "mer/images/eff").glob(ext)})
    exclude = {p.stem for p, _ in mer_test}
    exclude |= {Path(r["image"]).stem for r in _to_records(mer_val)}
    exclude |= {Path(r["image"]).stem for r in _to_records(mer_adapt)}
    mer_uda = [p for p in eff if p.stem not in exclude]
    rng.shuffle(mer_uda)

    splits = {
        "msl_train": _to_records(msl_train),
        "msl_val": _to_records(msl_val),
        "msl_test": _to_records(msl_test[: cfg["msl_test_size"]]),
        "mer_test": _to_records(mer_test[: cfg["mer_test_size"]]),
        "mer_val": _to_records(mer_val),
        "mer_labeled_adapt": _to_records(mer_adapt),
        "mer_uda": [{"image": str(p)} for p in mer_uda[: cfg["mer_uda_size"]]],
        "m2020_val": _to_records(m2020[: cfg["m2020_val_size"]]),
        "meta": {"seed": cfg["seed"]},
    }

    out = Path(cfg["output_dir"]) / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(splits, f, indent=2)
    return splits


def load_splits(output_dir: str) -> dict:
    with (Path(output_dir) / "splits.json").open() as f:
        return json.load(f)
