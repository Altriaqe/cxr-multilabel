"""数据增强与预处理。

刻意不使用 albumentations：本项目的输入已经缓存成 576x576 的 uint8 数组，
解码不再是瓶颈，而 albumentations 1.x / 2.x 的 API 差异（`var_limit` -> `std_range`、
`fill_value` 等）会带来无谓的兼容成本。torchvision.transforms.v2 随 torchvision 一起安装，
版本永远对齐，且在本规模下足够快。

约定：网络输入 3 通道（灰度复制 3 份），以复用 ImageNet 预训练权重。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _expand_3ch(x: torch.Tensor) -> torch.Tensor:
    """(H, W) 或 (1, H, W) -> (3, H, W)。"""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


class ToModelInput:
    """uint8 灰度 -> float32 三通道 ImageNet 归一化张量。"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = v2.functional.to_dtype(x, torch.float32, scale=True)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        x = _expand_3ch(x)
        return v2.functional.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False)


class ToMedInput:
    """uint8 灰度 -> float32 **单通道** [−1024, 1024] 张量。

    这是 torchxrayvision 的输入约定（`xrv.datasets.normalize(img, 255)`）：
        x = (img / 255) * 2048 - 1024

    与 ImageNet 管线**不可混用**：喂错量级不会报错，但精度会崩。
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = v2.functional.to_dtype(x, torch.float32, scale=True)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 3:                 # 兜底：万一传进来是 3 通道，取均值回单通道
            x = x.mean(dim=0, keepdim=True)
        return x * 2048.0 - 1024.0


def build_med_train_transform(
    image_size: int = 512,
    scale: Tuple[float, float] = (0.80, 1.0),
    rotate_deg: float = 7.0,
    jitter: float = 0.2,
    erasing_p: float = 0.25,
) -> v2.Compose:
    """CXR 预训练模型用的训练增强（输出单通道 [-1024,1024]）。"""
    return v2.Compose([
        v2.RandomResizedCrop(image_size, scale=scale, ratio=(0.90, 1.10), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=rotate_deg, fill=0),
        v2.ColorJitter(brightness=jitter, contrast=jitter),
        ToMedInput(),
        # 归一化后取值范围是 [-1024,1024]，填 0 等价于填中灰
        v2.RandomErasing(p=erasing_p, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0.0),
    ])


def build_med_eval_transform(image_size: int = 512) -> v2.Compose:
    """CXR 预训练模型用的验证/推理变换（输出单通道 [-1024,1024]）。"""
    return v2.Compose([
        v2.Resize(image_size, antialias=True),
        ToMedInput(),
    ])


def build_train_transform(
    image_size: int = 512,
    scale: Tuple[float, float] = (0.80, 1.0),
    rotate_deg: float = 7.0,
    jitter: float = 0.2,
    erasing_p: float = 0.25,
    ra_ops: int = 0,
    ra_mag: int = 0,
    crop_ratio: Tuple[float, float] = (0.90, 1.10),
) -> v2.Compose:
    """训练增强：保守的几何 + 光度扰动。

    医学影像上的禁忌：垂直翻转、大角度旋转、颜色抖动（无色彩信息）、
    强弹性形变（会破坏解剖结构）。这里只保留临床合理范围内的扰动。

    强增强（`ra_ops > 0`）会额外插入 RandAugment，并把裁剪/抖动放宽，
    用于验证"是不是正则不够导致过拟合太早"。
    """
    ops: List = [
        v2.RandomResizedCrop(image_size, scale=scale, ratio=crop_ratio, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=rotate_deg, fill=0),
    ]
    if ra_ops > 0:
        ops.append(v2.RandAugment(num_ops=int(ra_ops), magnitude=int(ra_mag)))
    ops += [
        v2.ColorJitter(brightness=jitter, contrast=jitter),
        ToModelInput(),
        # 归一化后填 0 == 填 ImageNet 均值灰度，等价于温和的 cutout
        v2.RandomErasing(p=erasing_p, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0.0),
    ]
    return v2.Compose(ops)


def build_eval_transform(image_size: int = 512, resize_ratio: float = 1.0) -> v2.Compose:
    """验证/推理变换。resize_ratio != 1 用于多尺度 TTA。"""
    ops: List = []
    if resize_ratio != 1.0:
        s = int(round(image_size * resize_ratio))
        ops += [v2.Resize(s, antialias=True), v2.CenterCrop(image_size)]
    else:
        ops.append(v2.Resize(image_size, antialias=True))
    ops += [ToModelInput()]
    return v2.Compose(ops)


def build_tta_transforms(image_size: int = 512) -> List[Tuple[str, v2.Compose]]:
    """多尺度 TTA 基础变换；水平翻转版本由推理脚本额外叠加。"""
    return [
        ("base", build_eval_transform(image_size, 1.0)),
        ("scale0.9", build_eval_transform(image_size, 0.9)),
        ("scale1.1", build_eval_transform(image_size, 1.1)),
    ]


# ------------------------------------------------------------------ 按骨干自动选预处理
def is_med_arch(arch: str) -> bool:
    return str(arch).startswith("xrv_")


def build_any_train_transform(arch: str, image_size: int = 512, **kw) -> v2.Compose:
    """根据骨干自动选择 ImageNet 或 CXR 预处理。

    **两套预处理的输入约定不同，混用不会报错但精度会崩**：
      * ImageNet 管线：3 通道、ImageNet mean/std
      * CXR 管线    ：1 通道、[−1024, 1024]
    """
    if is_med_arch(arch):
        return build_med_train_transform(image_size, **{k: v for k, v in kw.items()
                                                        if k in ("scale", "rotate_deg", "jitter", "erasing_p")})
    return build_train_transform(image_size, **kw)


def build_any_eval_transform(arch: str, image_size: int = 512) -> v2.Compose:
    return build_med_eval_transform(image_size) if is_med_arch(arch) else build_eval_transform(image_size)


def build_any_tta_transforms(arch: str, image_size: int = 512) -> List[Tuple[str, v2.Compose]]:
    if is_med_arch(arch):
        return [
            ("base", build_med_eval_transform(image_size)),
            ("scale0.9", v2.Compose([v2.Resize(int(round(image_size * 0.9)), antialias=True),
                                     v2.CenterCrop(image_size), ToMedInput()])),
            ("scale1.1", v2.Compose([v2.Resize(int(round(image_size * 1.1)), antialias=True),
                                     v2.CenterCrop(image_size), ToMedInput()])),
        ]
    return build_tta_transforms(image_size)

# ---------------------------------------------------------------- 批量函数式增强
class BatchAugment:
    """在 (B,3,H,W) 上**批量**做几何/光度增强，替代逐样本的 torchvision 变换。

    为什么要批量：逐样本调用 v2.RandomHorizontalFlip / ColorJitter / RandomRotation
    每个样本都要走一次 Python 分支和独立 kernel 启动；批量版本把整个 batch 当张量算，
    翻转/亮度/对比度都是一次 elementwise 运算，旋转用一次 rotate。

    RandomResizedCrop 仍然逐样本（每个样本要独立的裁剪参数），所以它留在 Dataset 里，
    本模块只负责可批量的部分。

    输入/输出都是 [0,1] 的 float 张量，形状 (B,3,H,W)。归一化在批量增强之后做。
    """

    def __init__(self, p_flip: float = 0.5, rotate_deg: float = 7.0, jitter: float = 0.2):
        self.p_flip = float(p_flip)
        self.rotate_deg = float(rotate_deg)
        self.jitter = float(jitter)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        dev = x.device

        # --- 水平翻转：按掩码整批切一次 ---
        if self.p_flip > 0:
            m = torch.rand(B, device=dev) < self.p_flip
            if m.any():
                x = x.clone()
                x[m] = torch.flip(x[m], dims=[-1])

        # --- 亮度 / 对比度：每样本一个因子，一次广播乘 ---
        if self.jitter > 0:
            bf = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * self.jitter
            x = x * bf.view(-1, 1, 1, 1)
            cf = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * self.jitter
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) * cf.view(-1, 1, 1, 1) + mean
            x = x.clamp_(0.0, 1.0)

        # --- 旋转：离散角度分桶 ---
        # 注意：本版 torchvision 的 v2.functional.rotate **不支持批量角度列表**
        # （内部 ngle % 360 对 list 会 TypeError）。所以把连续角度量化成 5 个离散值，
        # 每个桶整批旋转一次 -> 每批只需 5 次 kernel 启动，而不是 B 次。
        # 量化到 5 档对增强强度的影响可以忽略（-7/-3.5/0/+3.5/+7 度）。
        if self.rotate_deg > 0:
            n_bucket = 5
            buckets = torch.randint(0, n_bucket, (B,), device=dev)
            for b in range(n_bucket):
                m = buckets == b
                if not m.any():
                    continue
                ang = (b / (n_bucket - 1) * 2 - 1) * self.rotate_deg
                if abs(ang) > 1e-6:
                    x = x.clone() if x is not getattr(self, '_x_src', None) else x
                    x[m] = v2.functional.rotate(x[m], float(ang), fill=0.0)
        return x


def build_train_transform_batched(
    image_size: int = 512,
    scale: Tuple[float, float] = (0.80, 1.0),
    rotate_deg: float = 7.0,
    jitter: float = 0.2,
    erasing_p: float = 0.25,
) -> Tuple[v2.Compose, BatchAugment]:
    """返回 (逐样本变换, 批量增强)。

    逐样本部分只做 RandomResizedCrop + 转 [0,1]（RRC 需要每样本独立参数）；
    其余统统交给返回的 BatchAugment，在训练循环里对整批做。
    """
    per_sample = v2.Compose([
        v2.RandomResizedCrop(image_size, scale=scale, ratio=(0.90, 1.10), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    batch_aug = BatchAugment(p_flip=0.5, rotate_deg=rotate_deg, jitter=jitter)
    return per_sample, batch_aug


def finalize_batch(x: torch.Tensor) -> torch.Tensor:
    """批量增强之后统一做的事：扩成 3 通道 + ImageNet 归一化。"""
    x = _expand_3ch(x)
    return v2.functional.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False)
