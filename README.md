# Mars terrain segmentation (MSL → MER)

ResNet34-U-Net pretrained on MSL (Curiosity) and adapted to MER (Spirit/Opportunity) with feature-level UDA and semi-supervised training on [AI4MARS](https://github.com/ai4mars/ai4mars-dataset).

Classes: soil, bedrock, sand, big_rock.

Full write-up: [report.pdf](report.pdf) (Cross-Rover Mars NavCam Terrain Segmentation with Domain Adaptation).

## Results (MER test)

| Method | mIoU | wIoU |
|--------|------|------|
| Zero-shot (MSL pretrain) | 0.590 | 0.769 |
| CORAL | 0.623 | 0.810 |
| MMD | 0.639 | 0.821 |
| DANN | 0.637 | 0.820 |
| ADDA | 0.647 | 0.846 |
| Pseudo-label | 0.127 | 0.203 |
| Two-phase semi-sup | 0.393 | 0.535 |
| **Joint semi-supervision** | **0.658** | **0.844** |

Best overall: joint semi-supervision (200 labeled + 2000 unlabeled MER images). Best pure UDA wIoU: ADDA.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU (CUDA 12.4 example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Requires Python 3.10+ and a CUDA GPU for training (batch size 16, ~12 GB VRAM).

## Dataset

1. Download **AI4MARS** merged dataset v0.6 from the [ai4mars-dataset repo](https://github.com/ai4mars/ai4mars-dataset) (see releases / download instructions there).
2. Unpack so the root folder is named `ai4mars-dataset-merged-0.6/` and sits next to `train.py`:

```
mars-segmentation/
├── report.pdf                      # final report (PDF)
├── ai4mars-dataset-merged-0.6/   # not in git
│   ├── msl/
│   ├── mer/
│   └── m2020/
├── train.py
└── ...
```

Or pass another path with `--data-root /path/to/ai4mars-dataset-merged-0.6`.

## Reproduce

```bash
# 1. Build splits (once)
python make_splits.py

# 2. MSL pretraining
python train.py --method supervised --out-ckpt outputs/E1.pt

# 3. Evaluate zero-shot on MER
python evaluate.py --checkpoint outputs/E1.pt --splits msl_test mer_test

# 4. Feature UDA (pick one)
python train.py --method mmd --init-ckpt outputs/E1.pt --out-ckpt outputs/mmd.pt
python train.py --method adda --preset adda --init-ckpt outputs/E1.pt --out-ckpt outputs/adda.pt

# 5. Joint semi-supervision (best result)
python train.py --method semisup_joint --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/joint.pt
python evaluate.py --checkpoint outputs/joint.pt --splits mer_test msl_test
```

Full automated pipeline (E1–E7 + test eval):

```bash
python experiments.py pipeline
```

E1 hyperparameter search + follow-up semi-supervised runs:

```bash
python experiments.py e1-search
python experiments.py followup --init-ckpt outputs/e1_search/E1B_low_lr.pt --topk 2
```

## Methods

`supervised`, `coral`, `mmd`, `dann`, `adda`, `pseudolabel`, `pseudolabel_ema`, `semisup`, `semisup_joint`, `combo`

Presets: `budget` (main pipeline), `followup` (EMA + joint semi-sup), `adda` (batch 32).

## Layout

```
adaptation.py    CORAL, MMD, DANN, ADDA, EMA
config.py        defaults and presets
data.py          dataset loading and splits
model.py         ResNet34 + U-Net
train.py         training CLI
evaluate.py      metrics CLI
experiments.py   full pipeline scripts
make_splits.py   build outputs/splits.json
metrics.py       mIoU / wIoU
utils.py         device, checkpoints
report.pdf       final report (PDF)
```

Outputs go to `outputs/` (checkpoints, splits, JSON summaries). Not tracked in git.
