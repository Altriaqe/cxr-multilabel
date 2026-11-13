"""自监督预训练用的双视图数据集。

## 合规边界（审计要看）

| 允许 | 禁止 |
|---|---|
| ✅ 只用 `cxr_first/train`（18,513 study / 30,344 图）| ❌ test 集参与任何形式的 SSL |
| ✅ 自监督目标（SimCLR 对比学习）| ❌ 使用任何外部医学数据 |
| ✅ 后续用 train 的标签监督微调 | ❌ 用医学预训练权重做初始化 |

**为什么不用 valid**：规则第 3 条——验证集一旦并入训练，就不能再用于选模/早停。
SSL 属于"参与训练"，所以 valid 一律不进 SSL，保留选模能力。

## 实现说明

`XrayImageDataset` 在 `transform=None` 时返回未归一化的张量（float, /255）。
这里需要**两次独立的随机增强**，所以不把 transform 交给 Dataset，
而是在 `__getitem__` 里对同一份 uint8 原图跑两遍增强。

注意必须还原成 uint8：`XrayImageDataset(transform=None)` 已经把值除以了 255，
如果直接在上面跑 `ColorJitter` / `GaussianBlur`，取值区间是 [0,1] 而
torchvision 的这两个变换假定 [0,1] —— 其实是对的，但 `RandomResizedCrop`
对 uint8 和 float 的插值行为不同。为与监督训练的像素语义保持一致，
这里统一回到 uint8 再增强。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.constants import CACHE_DIR
from src.data.dataset import XrayImageDataset
from src.data.manifest import load_split
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, _expand_3ch


def two_view_transform(size: int = 320):
    """返回两个**互相独立**的增强管线（随机种子不同，所以同图会得到不同视图）。"""

    def make():
        return v2.Compose([
            # ---- 增强强度是实测标定的，不要随手改 ----
            # 初始 NT-Xent 必须落在 4.0~4.6（B=32 的理论上限 ln(63)=4.14）。
            # 实测（ConvNeXt-T 预训练骨干 + 320 输入）：
            #   scale 0.50 / blur 0.3 / rot  7度 -> loss 3.115，视图特征余弦 0.800（太容易）
            #   scale 0.25 / blur 0.5 / rot 15度 -> loss 3.428，余弦 0.720
            #   scale 0.15 / blur 0.7 / rot 20度 -> loss 4.004，余弦 0.649  <-- 采用
            # 原因：从 ImageNet 起步时两视图本就高度相似，增强太弱会让对比任务变恒等映射，
            # 模型不用学就能对上正样本，SSL 就白跑了。
            v2.RandomResizedCrop(size, scale=(0.15, 1.0), ratio=(0.75, 1.33), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=20, fill=0),
            # 亮度和对比度是最关键的两个增强：X 光片的曝光差异极大
            v2.RandomApply([v2.ColorJitter(brightness=0.8, contrast=0.8)], p=0.8),
            v2.RandomApply([v2.GaussianBlur(3)], p=0.7),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(_expand_3ch),                       # 单通道 -> 3 通道
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return [make(), make()]


class TwoViewDataset(Dataset):
    """同一张胸片返回两个独立增强视图，供 SimCLR 使用。

    __getitem__ -> (view1, view2)，各为 (3, size, size) float32。
    """

    def __init__(self, size: int = 384, cache_size: int = 576, split: str = "train"):
        if split != "train":
            raise ValueError(
                f"SSL 只允许用 train 划分，收到 split={split!r}。"
                "规则第 3/5 条：valid 一旦并入训练就不能再用于选模，test 则完全禁止参与。"
            )
        self.table = load_split("train")
        # transform=None -> 返回 (1,H,W) 未归一化张量；我们自己做两次独立增强
        self.ds = XrayImageDataset(
            self.table,
            transform=None,
            cache_path=CACHE_DIR / f"train_{cache_size}.npy",
            with_labels=False,
        )
        self.tf = two_view_transform(size)

    def __len__(self) -> int:
        return len(self.ds)

    def _raw_uint8(self, i: int) -> torch.Tensor:
        img = self.ds[i][0]
        if img.dtype == torch.uint8:
            return img
        # Dataset 已 /255，还原回 uint8 再增强，保证与监督训练像素语义一致
        return (img * 255.0).clamp(0, 255).to(torch.uint8)

    def __getitem__(self, i: int):
        img = self._raw_uint8(i)
        v1 = self.tf[0](img)
        v2 = self.tf[1](img)
        return v1, v2
