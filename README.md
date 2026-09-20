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
| Mean Teacher | 0.588 | 0.769 |
| **Pseudo-label** | **0.724** | **0.869** |

Best overall: **pseudo-labeling** (200 labeled + 2000 unlabeled MER). Best pure UDA: ADDA.

Primary metric: **wIoU**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Python 3.10+, CUDA GPU recommended (batch 16, ~12 GB).

## Dataset

Download AI4MARS merged v0.6 and place `ai4mars-dataset-merged-0.6/` next to `train.py`
(or pass `--data-root`).

## Reproduce

```bash
python make_splits.py

python train.py --method supervised --out-ckpt outputs/E1.pt
python evaluate.py --checkpoint outputs/E1.pt --splits msl_test mer_test

# UDA
python train.py --method coral --init-ckpt outputs/E1.pt --out-ckpt outputs/coral.pt
python train.py --method mmd --init-ckpt outputs/E1.pt --out-ckpt outputs/mmd.pt
python train.py --method dann --init-ckpt outputs/E1.pt --out-ckpt outputs/dann.pt
python train.py --method adda --preset adda --init-ckpt outputs/E1.pt --out-ckpt outputs/adda.pt

# Semi-supervised
python train.py --method mean_teacher --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/mean_teacher.pt
python train.py --method pseudolabel --preset followup \
  --init-ckpt outputs/E1.pt --out-ckpt outputs/pseudolabel.pt
python evaluate.py --checkpoint outputs/pseudolabel.pt --splits mer_test
```

Full pipeline:

```bash
python experiments.py pipeline
```

## Methods

| CLI | Role |
|-----|------|
| `supervised` | MSL pretrain (baseline) |
| `coral` / `mmd` / `dann` / `adda` | Feature UDA on `enc4` |
| `pseudolabel` | Hard pseudo-labels + labeled CE |
| `mean_teacher` | EMA teacher + MSE consistency + labeled CE |

Presets: `budget`, `followup` (SSL), `adda` (batch 32).

## Code references

Training loops and helpers are adapted from public PyTorch repos (classification → dense
segmentation on ResNet34-U-Net):

| Method | Paper | Reference implementation |
|--------|-------|--------------------------|
| ResNet34-U-Net | Ronneberger et al. / He et al. | [gyb357/UNet-Segmentation](https://github.com/gyb357/UNet-Segmentation) (encoder layout; see `model.py`) |
| DANN | Ganin et al., 2016 | [fungtion/DANN](https://github.com/fungtion/DANN) |
| ADDA | Tzeng et al., 2017 | [ayushtues/ADDA_pytorch](https://github.com/ayushtues/ADDA_pytorch) (shared-encoder variant here) |
| Pseudo-label | Lee, 2013 | [iBelieveCJM/pseudo_label-pytorch](https://github.com/iBelieveCJM/pseudo_label-pytorch) |
| Mean Teacher | Tarvainen & Valpola, 2017 | [CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher) |

CORAL / MMD follow Sun & Saenko (2016) and Gretton et al. (2012); no separate reference repo was used.
See comments in `model.py`, `train.py`, and `adaptation.py`.

## Layout

```
adaptation.py   CORAL, MMD, DANN, ADDA, Mean Teacher helpers
config.py       defaults / presets
data.py         loading and splits
model.py        ResNet34-U-Net
train.py        training CLI
evaluate.py     metrics CLI
experiments.py  pipeline / e1-search / followup
make_splits.py  outputs/splits.json
metrics.py      mIoU / wIoU
utils.py        device, checkpoints
```

Outputs go to `outputs/` (not in git).
