"""模型定义（供 predict.py / train.py / cost.py 共用）。

这里只放**推理需要的最小模型定义**，不依赖项目里的 src/ 包，
保证材料包解压后即可运行。
"""
from __future__ import annotations

import torch


class XrayClassifier(torch.nn.Module):
    """ConvNeXt 分类器：timm 骨干 + 10 类线性头。

    权重通过 timm 的 ImageNet 预训练初始化（训练时 pretrained=True），
    推理时由 `predict.load_models` 载入微调后的 state_dict。
    """

    def __init__(self, arch: str, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        import timm
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained,
                                          num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
