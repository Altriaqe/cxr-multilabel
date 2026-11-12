"""基于 torchxrayvision 的 CXR 预训练模型封装。

## 为什么用

ImageNet 预训练学到的是自然图像的纹理/边缘；胸部 X 光片是单通道、低对比、
以解剖结构为主的影像，域差距很大。torchxrayvision 提供的权重是在
CheXpert / NIH / RSNA / SIIM / PadChest / VIN 等**真实 CXR 数据集**上训练出来的，
特征天然贴近本任务。

其中 `resnet50-res512-all` 尤其合适：**ResNet50 + 512 输入**，与我们的管线完全对齐。

## 输入约定（与 ImageNet 管线不同，务必注意）

| 项 | ImageNet 管线 | 本管线 |
|---|---|---|
| 通道 | 3（灰度复制） | **1** |
| 归一化 | ImageNet mean/std → 约 [-2.1, 2.6] | **x/255*2048 − 1024 → [-1024, 1024]** |

torchxrayvision 的模型内部假定输入是这个 [-1024, 1024] 量级（它自带
`utils.warn_normalization` 检查）。喂错量级不会报错，但精度会崩，所以两套
预处理必须严格配套使用。

## 可用权重（torchxrayvision 1.5.7）

    densenet121-res224-chex / -nih / -pc / -rsna / -mimic_nb / -mimic_ch / -all
    resnet50-res512-all          <-- 我们用的这个
"""

from __future__ import annotations

import torch
import torch.nn as nn


XRV_WEIGHTS = {
    "xrv_resnet50": "resnet50-res512-all",
    "xrv_densenet121_chex": "densenet121-res224-chex",
    "xrv_densenet121_all": "densenet121-res224-all",
}


class CXRResNetClassifier(nn.Module):
    """CXR 预训练 ResNet50 + 自定义 10 类头。

    Args:
        num_classes: 输出类别数
        weights: torchxrayvision 的权重名；None 表示不加载预训练（只用结构，
                 例如推理时用自己的 checkpoint 覆盖）
        drop: 分类头前的 dropout
        freeze_backbone: 是否冻结骨干（先训头用）
    """

    def __init__(self, num_classes: int = 10, weights: str | None = "resnet50-res512-all",
                 drop: float = 0.4, freeze_backbone: bool = False):
        super().__init__()
        self.arch = f"xrv:{weights or 'scratch'}"
        self.num_classes = num_classes
        self.in_channels = 1
        self.input_resolution = 512

        if weights is not None:
            try:
                import torchxrayvision as xrv
            except ImportError as exc:  # pragma: no cover
                raise ImportError("需要 torchxrayvision: pip install torchxrayvision") from exc
            base = xrv.models.ResNet(weights=weights)
            inner = base.model  # torchvision ResNet，conv1 已改成单通道，fc 是 18 类
            self.feature_dim = inner.fc.in_features
        else:
            # 不依赖 torchxrayvision 构建同构网络：torchvision resnet50 + 单通道 conv1。
            # 键名与 xrv 版本完全一致，因此 state_dict 可以互通（下方有断言测试）。
            import torchvision.models as tvm

            inner = tvm.resnet50(weights=None, num_classes=18)
            inner.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = inner.fc.in_features

        # 去掉最后的 fc，保留到 avgpool：输出 (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(inner.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.feature_dim, num_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = torch.flatten(f, 1)
        return self.classifier(f)

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = flag


class CXRDenseNetClassifier(nn.Module):
    """CXR 预训练 DenseNet121（224 输入）+ 自定义头。

    DenseNet 版本在 torchxrayvision 里输入是 224，训练时会被 `fix_resolution` 缩小，
    所以要在 224 上训练，不能直接用 512（否则自动降采样，白白多花算力）。
    """

    def __init__(self, num_classes: int = 10, weights: str = "densenet121-res224-chex",
                 drop: float = 0.2, freeze_backbone: bool = False):
        super().__init__()
        try:
            import torchxrayvision as xrv
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 torchxrayvision: pip install torchxrayvision") from exc

        base = xrv.models.DenseNet(weights=weights)
        # features2 = features + relu + adaptive_avg_pool + flatten
        self.backbone = base.features2
        dim = base.classifier.in_features
        self.feature_dim = dim
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim, num_classes),
        )
        self.arch = f"xrv:{weights}"
        self.num_classes = num_classes
        self.input_resolution = getattr(base, "input_resolution", 224)
        self.in_channels = 1

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return self.classifier(f)
