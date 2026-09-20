# Mars terrain segmentation (MSL → MER)

ResNet34-U-Net pretrained on MSL (Curiosity) and adapted to MER (Spirit/Opportunity) with
feature-level UDA, Lee-style pseudo-labeling, and Mean Teacher on [AI4MARS](https://github.com/ai4mars/ai4mars-dataset).

Classes: soil, bedrock, sand, big_rock.

Full write-up: [FRACCAROLI_CESARE_REPORT_CVDL.pdf](FRACCAROLI_CESARE_REPORT_CVDL.pdf).

## Results (final report)

### Feature UDA (MER test)

| Method | mIoU | wIoU |
|--------|------|------|
| Baseline (MSL pretrain) | 0.590 | 0.769 |
| CORAL | 0.623 | 0.810 |
| MMD | 0.639 | 0.821 |
| DANN | 0.637 | 0.820 |
| ADDA | 0.647 | **0.846** |

### Semi-supervised (MER test)

| Method | mIoU | wIoU |
|--------|------|------|
| Baseline | 0.590 | 0.769 |
| Fine-tune 200 | 0.692 | 0.849 |
| Mean Teacher | 0.588 | 0.769 |
| **Pseudo-label** | **0.724** | **0.869** |

Best overall: **pseudo-labeling** (200 labeled + 2000 unlabeled MER). Best pure UDA: ADDA.

Primary metric in the report is **wIoU** (frequency-weighted IoU); mIoU is secondary.

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

1. Download **AI4MARS** merged dataset v0.6 from the [ai4mars-dataset repo](https://github.com/ai4mars/ai4mars-dataset).
2. Unpack so the root folder is named `ai4mars-dataset-merged-0.6/` and sits next to `train.py`:

```
ai4mars_terrain_segmentation/
├── FRACCAROLI_CESARE_REPORT_CVDL.pdf
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

# 3. Baseline on MER
python evaluate.py --checkpoint outputs/E1.pt --splits msl_test mer_test

# 4. Feature UDA (pick one)
python train.py --method mmd --init-ckpt outputs/E1.pt --out-ckpt outputs/mmd.pt
python train.py --method adda --preset adda --init-ckpt outputs/E1.pt --out-ckpt outputs/adda.pt

# 5. Semi-supervised (from E1)
python train.py --method finetune --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/finetune.pt
python train.py --method mean_teacher --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/mean_teacher.pt
python train.py --method pseudolabel --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/pseudolabel.pt
python evaluate.py --checkpoint outputs/pseudolabel.pt --splits mer_test msl_test
```

Full automated pipeline (E1 + UDA + SSL + test eval):

```bash
python experiments.py pipeline
```

E1 hyperparameter search + SSL follow-up:

```bash
python experiments.py e1-search
python experiments.py followup --init-ckpt outputs/e1_search/E1B_low_lr.pt --topk 2
```

## Methods

| CLI | Role |
|-----|------|
| `supervised` | MSL pretrain (baseline init) |
| `coral` / `mmd` / `dann` / `adda` | Feature UDA on `enc4` |
| `finetune` | CE on 200 labeled MER only |
| `pseudolabel` | Lee hard pseudo-labels + labeled CE |
| `mean_teacher` | EMA teacher + MSE consistency + labeled CE |

Presets: `budget` (main pipeline), `followup` (SSL), `adda` (batch 32).

### Semi-supervised losses (short)

- **Pseudo-label:** `L = CE(x_L, y_L) + α(t) · CE(x_U, ŷ)` with `ŷ = argmax(student(x_U))`, low-conf pixels ignored.
- **Mean Teacher:** `L = CE(x_L, y_L) + λ(t) · MSE(softmax(s(x_U+η)), softmax(t(x_U+η)))`; report the EMA teacher.

## Layout

```
adaptation.py    CORAL, MMD, DANN, ADDA helpers, Mean Teacher EMA / MSE
config.py        defaults and presets
data.py          dataset loading and splits
model.py         ResNet34 + U-Net
train.py         training CLI
evaluate.py      metrics CLI
experiments.py   full pipeline scripts
make_splits.py   build outputs/splits.json
metrics.py       mIoU / wIoU
utils.py         device, checkpoints
FRACCAROLI_CESARE_REPORT_CVDL.pdf
```

Outputs go to `outputs/` (checkpoints, splits, JSON summaries). Not tracked in git.
