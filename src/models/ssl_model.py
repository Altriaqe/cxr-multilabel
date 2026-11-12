"""SimCLR 模型：骨干 + 投影头。

## 为什么选 ConvNeXt-T

* ConvNeXt 用 **LayerNorm，没有 BatchNorm**，因此不存在 SimCLR 的经典坑
  ——BN 在跨样本统计时会泄漏信息、让"同一张图的两个视图"互相可见。
  用 BN 网络做对比学习必须加 sync-BN 或改用其他策略，成本高。
* 384 输入比微调用的 512 便宜 1.8 倍，又保留足够细节。
* 复用现有的 576 缓存：RandomResizedCrop 到 384，裁剪余量充足。

## 初始化方式

`pretrained=True`（ImageNet 起始），这是方案文档 §4 的建议：
* 规则明确允许 ImageNet 初始化（非医学领域开源权重）
* 从随机初始化起步需要 20+ 轮才收敛，成本翻倍，而我们时间只有 2.5 小时

所以这里的 SSL 本质是**领域自适应预训练**：把 ImageNet 特征往胸片域再推一把，
然后用 train 的标签微调。真正的对照基线是"ImageNet 直接微调"。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    """骨干 + 三层投影头（feat → hidden → proj_dim）。

    forward(x) -> (z_normalized, features)
        z 用于 NT-Xent 损失（单位球面上的向量）
        features 用于后续检查特征是否退化
    """

    def __init__(self, arch: str = "convnext_tiny", proj_dim: int = 128,
                 hidden: int = 2048, pretrained: bool = True,
                 grad_checkpointing: bool = False):
        super().__init__()
        import timm

        # num_classes=0 -> timm 返回池化后的特征向量，不带分类头
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        # 梯度检查点：用重算换显存，384/320 输入 + 两个视图时能省 30~40%
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
            self.grad_checkpointing = True
        feat = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )
        self.arch = arch
        self.feat_dim = feat
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        z = self.projector(f)
        return F.normalize(z, dim=1), f

    def backbone_state_dict(self) -> dict:
        """只取骨干（去掉 `backbone.` 前缀名不动，交给 init_from_ssl 处理）。"""
        return {k: v for k, v in self.state_dict().items() if k.startswith("backbone.")}


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """NT-Xent 对比损失（SimCLR）。

    Args:
        z1, z2: 同一批样本的两个视图，形状 (B, D)，**已 L2 归一化**。
        temp: 温度，越小越锐利。batch 小的时候适当降低（负样本少）。

    构造 2B×2B 相似度矩阵，正样本是"同一样本的另一视图"。
    """
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temp

    n = z1.size(0)
    eye = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)          # 对角线是自己和自己，屏蔽

    targets = torch.cat([
        torch.arange(n, 2 * n, device=z.device),   # 前一半 -> 后一半
        torch.arange(0, n, device=z.device),       # 后一半 -> 前一半
    ])
    return F.cross_entropy(sim, targets)
