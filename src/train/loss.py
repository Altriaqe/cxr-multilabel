"""损失函数。

三档选择：
  * `bce`        —— 稳定基线，AUC 任务上通常已经很强；
  * `focal`      —— 抑制大量易分负样本的梯度；
  * `asymmetric` —— ASL，正负样本分别设置 gamma / clip，多标签场景常用。

ASL 的实现按原论文（Ben-Baruch et al., 2020）的公式写，
不采用旧文件里那种"分别算正负项"的近似写法。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification.

    Args:
        gamma_pos: 正样本聚焦参数（通常 0~1，保留正样本梯度）
        gamma_neg: 负样本聚焦参数（通常 2~5，压制易分负样本）
        clip: 负样本概率下界，进一步丢弃极简负样本
        eps: 数值稳定项
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(logits)
        x_pos = x
        x_neg = 1.0 - x
        if self.clip > 0:
            x_neg = (x_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(x_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(x_neg.clamp(min=self.eps))

        pt = x_pos * targets + x_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = -(loss_pos + loss_neg) * pt.pow(gamma)
        return loss.mean() if self.reduction == "mean" else loss


class BCEWithLogits(nn.Module):
    """可选的正样本加权 BCE（pos_weight 由训练集真实正负比估计）。"""

    def __init__(self, pos_weight: Optional[Sequence[float]] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw.to(logits.device) if pw is not None else None
        )


def build_loss(name: str = "bce", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "bce":
        pw = kwargs.get("pos_weight")
        return BCEWithLogits(pw) if pw is not None else nn.BCEWithLogitsLoss()
    if name == "focal":
        return BinaryFocalLoss(**{k: v for k, v in kwargs.items() if k in ("alpha", "gamma")})
    if name == "asymmetric" or name == "asl":
        return AsymmetricLoss(**{k: v for k, v in kwargs.items() if k in ("gamma_pos", "gamma_neg", "clip")})
    raise ValueError(f"未知损失: {name}")
