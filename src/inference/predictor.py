"""推理：图像级概率预测与 TTA。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (  # noqa: F401  (constants 里顺带设置了 HF_ENDPOINT 等环境变量)
    NUM_CLASSES,
)
from src.data.dataset import worker_init_fn


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   amp: bool = True, n_out: Optional[int] = None,
                   return_logits: bool = False):
    """返回 (n_images, C) 概率矩阵，行序与数据集索引一致。

    return_logits=True 时改返回 (probs, logits) —— study 级融合时
    "logit 取均值再 sigmoid" 通常比"概率取均值"更稳（后者会被单视图极端值拖累）。
    """
    model.eval()
    n = len(loader.dataset)
    n_out = n_out or getattr(model, "num_classes", 10)
    out = np.zeros((n, n_out), dtype=np.float32)
    out_logits = np.zeros((n, n_out), dtype=np.float32) if return_logits else None
    use_amp = amp and device.type == "cuda"
    for batch in tqdm(loader, desc="predict", leave=False):
        imgs, idx = batch[0], batch[-1]
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        out[idx.numpy()] = torch.sigmoid(logits).float().cpu().numpy()
        if return_logits:
            out_logits[idx.numpy()] = logits.float().cpu().numpy()
    return (out, out_logits) if return_logits else out


@torch.no_grad()
def predict_dataset_with_tta(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    transforms: Sequence,
    use_hflip: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    amp: bool = True,
    n_views: int = 1,
    return_logits: bool = False,
):
    """多变换 / 水平翻转 TTA 平均。

    Args:
        dataset: `XrayImageDataset`（with_labels=False 也可以）
        transforms: 变换列表。对**所有图像**使用同一组变换做平均。
        return_logits: 同时返回平均 logits，供 study 级 logit_mean 融合使用。

    注意：这里对 **logits 求平均**，再 sigmoid 得到概率。多视图/多增强融合时
    "先平均 logit 再 sigmoid" 比"先 sigmoid 再平均概率"更稳（后者会被单个
    极端视图拖偏）。返回的 probs 已由平均 logits 计算，两者一致。
    """
    n = len(dataset)
    n_out = getattr(model, "num_classes", 10)
    total_logits = np.zeros((n, n_out), dtype=np.float64)
    count = 0
    use_amp = amp and device.type == "cuda"
    model.eval()

    variants: List[Tuple[object, bool]] = [(tf, False) for tf in transforms]
    if use_hflip:
        variants += [(tf, True) for tf in transforms]

    original_tf = dataset.transform
    try:
        for tf, flip in variants:
            dataset.transform = tf
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                worker_init_fn=worker_init_fn)
            for batch in tqdm(loader, desc="tta", leave=False):
                imgs, idx = batch[0], batch[-1]
                if flip:
                    imgs = torch.flip(imgs, dims=[-1])
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                total_logits[idx.numpy()] += logits.float().cpu().numpy()
            count += 1
    finally:
        dataset.transform = original_tf

    mean_logits = (total_logits / max(count, 1)).astype(np.float32)
    probs = (1.0 / (1.0 + np.exp(-mean_logits.astype(np.float64)))).astype(np.float32)
    return (probs, mean_logits) if return_logits else probs
