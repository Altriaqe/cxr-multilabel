"""[已废弃 · 未被任何代码引用] 旧版 albumentations 数据增强

保留仅作参考。**当前管线使用的是 `src/data/transforms.py`**（torchvision v2）。

为什么换掉：
1. albumentations 1.x / 2.x 的 API 漂移（`GaussNoise(var_limit=...)` → `std_range=...`、
   `CoarseDropout(fill_value=...)` 等），会带来无谓的兼容成本；
2. 本项目输入已经缓存成 576/768 的 uint8 数组，解码不再是瓶颈，
   albumentations 的速度优势用不上；
3. torchvision.transforms.v2 与 torchvision 版本天然对齐。

另外本文件里的几个假设要留意：
- 依赖 `albumentations`，而它**不在 requirements.txt 里**，所以本文件 `import` 会失败；
- `CoarseDropout(max_height=32, max_width=32)` 相对 512 的输入太小，遮挡效果几乎为零；
- `GaussNoise` 对 X 光片是伪需求（灰度、低噪声、噪声模式与自然图像不同）。

数据增强函数（历史版本，勿用）
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=512):
    """
    训练集数据增强

    Args:
        image_size (int): 目标图像尺寸

    Returns:
        albumentations.Compose: 数据增强pipeline
    """
    return A.Compose([
        # 尺寸调整
        A.Resize(image_size, image_size),

        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),

        # 亮度和对比度
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0
            ),
        ], p=0.5),

        # 噪声和模糊
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),

        # Cutout (遮挡部分区域)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),

        # 归一化（使用ImageNet统计量）
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        # 转换为PyTorch张量
        ToTensorV2(),
    ])


def get_val_transforms(image_size=512):
    """
    验证集/测试集数据变换（不包含随机增强）

    Args:
        image_size (int): 目标图像尺寸

    Returns:
        albumentations.Compose: 数据变换pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_heavy_train_transforms(image_size=512):
    """
    重度数据增强（用于更激进的正则化）

    Args:
        image_size (int): 目标图像尺寸

    Returns:
        albumentations.Compose: 重度增强pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # 更激进的几何变换
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=0,
            p=0.7
        ),

        # 弹性变形
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),

        # 网格扭曲
        A.GridDistortion(p=0.3),

        # 光学畸变
        A.OpticalDistortion(
            distort_limit=0.1,
            shift_limit=0.1,
            p=0.3
        ),

        # 亮度、对比度、饱和度
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),

        # CLAHE（对比度受限自适应直方图均衡）
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

        # 噪声
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.5),

        # 模糊和锐化
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.4),

        # GridMask
        A.CoarseDropout(
            max_holes=12,
            max_height=48,
            max_width=48,
            min_holes=5,
            min_height=16,
            min_width=16,
            fill_value=0,
            p=0.5
        ),

        # 归一化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])


def get_tta_transforms(image_size=512):
    """
    Test Time Augmentation变换列表

    Args:
        image_size (int): 目标图像尺寸

    Returns:
        list: TTA变换列表
    """
    tta_transforms = [
        # 原始图像
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),

        # 水平翻转
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),

        # 稍微缩小
        A.Compose([
            A.Resize(int(image_size * 0.9), int(image_size * 0.9)),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),

        # 稍微放大
        A.Compose([
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

    return tta_transforms
