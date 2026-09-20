"""Training config and presets (budget, followup, adda)."""

from __future__ import annotations

import copy

DEFAULT_DATA_ROOT = "ai4mars-dataset-merged-0.6"

DEFAULTS: dict = {
    "data_root": DEFAULT_DATA_ROOT,
    "output_dir": "outputs",
    "num_classes": 4,
    "ignore_index": 255,
    "image_size": 512,
    "seed": 42,

    "msl_train_size": 3000,
    "msl_val_size": 200,
    "msl_test_size": 322,
    "mer_labeled_adapt_size": 200,
    "mer_val_size": 200,
    "mer_uda_size": 2000,
    "mer_test_size": 204,
    "m2020_val_size": 200,

    "batch_size": 16,
    "num_workers": 4,
    "pin_memory": True,
    "amp": True,
    "use_cuda": True,
    "cudnn_benchmark": True,

    "lr": 1.0e-4,
    "lr_encoder": 1.0e-5,
    "lr_decoder": 1.0e-4,
    "weight_decay": 1.0e-4,

    "epochs_supervised": 15,
    "early_stop_patience": 5,

    "uda_epochs": 8,
    "uda_lambda": 0.1,
    "dann_lambda_max": 0.1,
    "mmd_sigma": 1.0,

    "epochs_semisup": 10,
    "uda_lr_factor": 0.1,

    # Pseudo-labeling (Lee): CE on labeled MER + α(t)·CE on hard unlabeled masks.
    "semisup_lambda_uda": 1.0,
    "semisup_lambda_ramp_epochs": 4,
    "uda_confidence_threshold": 0.9,
    "uda_min_pseudo_pixels": 100,

    # Mean Teacher: CE on labeled MER + λ(t)·MSE(softmax student, softmax teacher).
    "mt_ema_decay": 0.99,
    "mt_consistency_weight": 1.0,
    "mt_rampup_epochs": 5,
    "mt_noise_std": 0.1,

    "adda_epochs_disc": 5,
    "adda_epochs_adapt": 10,
    "adda_disc_lr": 1.0e-4,
    "adda_encoder_lr": 2.0e-5,
    "adda_decoder_lr": 5.0e-5,
    "adda_lambda_seg": 1.0,
    "adda_lambda_adv": 0.1,
    "adda_lambda_coral": 0.1,
    "adda_adv_ramp": True,
}

PRESET_BUDGET: dict = {}

PRESET_ADDA: dict = {
    "batch_size": 32,
}

PRESET_FOLLOWUP: dict = {
    "epochs_semisup": 12,
    "uda_lr_factor": 0.05,
    "semisup_lambda_uda": 1.0,
    "semisup_lambda_ramp_epochs": 4,
    "uda_confidence_threshold": 0.9,
    "uda_min_pseudo_pixels": 100,
}

PRESETS = {"budget": PRESET_BUDGET, "followup": PRESET_FOLLOWUP, "adda": PRESET_ADDA}


def get_config(preset: str = "budget", **overrides) -> dict:
    if preset not in PRESETS:
        raise ValueError("unknown preset:", preset, "choose from", list(PRESETS))
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(PRESETS[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
