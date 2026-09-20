"""Domain adaptation losses and helpers (CORAL, MMD, DANN, Mean Teacher)."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def feat_map(feat: torch.Tensor, l2norm: bool = True) -> torch.Tensor:
    """Flatten enc4 to per-pixel features [B*H*W, C]."""
    b, c, _, _ = feat.shape
    v = feat.permute(0, 2, 3, 1).reshape(-1, c)
    return F.normalize(v, dim=1) if l2norm else v


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align source/target feature covariances (CORAL)."""
    ns, nt = source.shape[0], target.shape[0]
    if ns < 2 or nt < 2:
        return source.new_tensor(0.0)
    d = source.shape[1]
    xs = source - source.mean(0, keepdim=True)
    xt = target - target.mean(0, keepdim=True)
    cov_s = (xs.t() @ xs) / max(ns - 1, 1)
    cov_t = (xt.t() @ xt) / max(nt - 1, 1)
    return ((cov_s - cov_t) ** 2).sum() / d


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True)
    dist = xx + yy.t() - 2 * (x @ y.t())
    return torch.exp(-dist.clamp(min=0) / (2 * sigma ** 2))


def mmd_loss(source: torch.Tensor, target: torch.Tensor, sigma: float = 1.0, max_n: int = 2048) -> torch.Tensor:
    """RBF-kernel MMD between source and target features."""
    ns, nt = source.shape[0], target.shape[0]
    if ns < 2 or nt < 2:
        return source.new_tensor(0.0)
    if ns > max_n:
        source = source[torch.randperm(ns, device=source.device)[:max_n]]
    if nt > max_n:
        target = target[torch.randperm(nt, device=target.device)[:max_n]]
    k_ss = _rbf_kernel(source, source, sigma)
    k_tt = _rbf_kernel(target, target, sigma)
    k_st = _rbf_kernel(source, target, sigma)
    return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()


class GradientReversal(Function):
    """Reverse gradients through the discriminator path (DANN)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def grl_lambda(epoch: int, max_epochs: int, lambda_max: float) -> float:
    """Ganin schedule: ramp GRL strength from 0 to lambda_max."""
    p = epoch / max(max_epochs, 1)
    return float(lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def gap_features(e4: torch.Tensor) -> torch.Tensor:
    return e4.mean(dim=(2, 3))


def domain_discriminator_loss(logits: torch.Tensor, is_source: bool) -> torch.Tensor:
    """BCE: source=0, target=1."""
    target = torch.zeros_like(logits) if is_source else torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def adversarial_encoder_loss(logits: torch.Tensor) -> torch.Tensor:
    """Push target features toward source (label 0)."""
    return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """EMA weights; also copy BN buffers (Mean Teacher)."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


def clone_teacher(model: nn.Module) -> nn.Module:
    """Frozen EMA teacher copy."""
    teacher = deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """Mean Teacher consistency weight: exp(-5(1-t)^2)."""
    if rampup_length <= 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def consistency_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE between student and teacher softmax maps."""
    return F.mse_loss(F.softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
