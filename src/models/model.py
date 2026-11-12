"""模型定义。

用 timm 统一构建 backbone + 分类头，避免为每个骨干手写池化逻辑
（旧实现用 `global_pool=''` 再自己判断张量维度，对 ViT 类模型会算错特征维度）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # timm 是主要路径；缺失时给出可读的报错
    import timm
except ImportError as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc


SUPPORTED_BACKBONES = [
    "resnet34", "resnet50", "resnet101",
    "densenet121",
    "efficientnet_b0", "efficientnet_b3", "efficientnet_b4",
    "convnext_tiny", "convnext_small", "convnext_base",
    "swin_tiny_patch4_window7_224",
]


class XrayClassifier(nn.Module):
    """timm backbone + dropout + linear 的 10 类多标签分类器。"""

    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        drop: float = 0.4,
        drop_path: float = 0.1,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise ImportError("需要安装 timm: pip install timm") from _TIMM_IMPORT_ERROR

        kwargs = dict(
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )
        try:
            self.backbone = timm.create_model(arch, pretrained=pretrained, **kwargs)
        except TypeError:
            # 部分模型不接受 drop_path_rate
            kwargs.pop("drop_path_rate")
            self.backbone = timm.create_model(arch, pretrained=pretrained, **kwargs)

        if grad_checkpointing:
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        self.arch = arch
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


MED_ARCH_PREFIX = "xrv_"


def is_med_arch(arch: str) -> bool:
    """是否为 CXR 预训练（torchxrayvision）骨干。"""
    return arch.startswith(MED_ARCH_PREFIX)


def build_model(arch: str = "resnet50", num_classes: int = 10, pretrained: bool = True, **kw) -> nn.Module:
    """统一入口。

    `arch` 以 `xrv_` 开头时走 torchxrayvision 的 CXR 预训练权重
    （输入约定不同：单通道、[-1024,1024]，需配套 `build_med_*_transform`）。
    其余走 timm 的 ImageNet 骨干。
    """
    if is_med_arch(arch):
        from src.models.med_model import (
            CXRDenseNetClassifier,
            CXRResNetClassifier,
            XRV_WEIGHTS,
        )

        if arch not in XRV_WEIGHTS:
            raise ValueError(f"未知的医学预训练骨干: {arch}，可选 {list(XRV_WEIGHTS)}")
        weights = XRV_WEIGHTS[arch]
        # pretrained=False 时仍要建同构网络（推理时用我们自己的 state_dict 覆盖）
        w = weights if pretrained else None
        if "densenet" in weights:
            return CXRDenseNetClassifier(num_classes, w, drop=kw.get("drop", 0.2))
        return CXRResNetClassifier(num_classes, w, drop=kw.get("drop", 0.4))

    return XrayClassifier(arch=arch, num_classes=num_classes, pretrained=pretrained, **kw)


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


if __name__ == "__main__":
    for arch in SUPPORTED_BACKBONES:
        m = build_model(arch, pretrained=False)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{arch:<40} params={n_params/1e6:7.2f}M  fp32={model_size_mb(m):7.1f}MB")
