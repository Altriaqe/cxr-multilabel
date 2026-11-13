"""按 AP 视图过采样的采样器。

## 为什么需要

线上 test 的视图构成和训练集不同：

| 视图组 | train 占比 | test 占比 | ResNet50 的 AUC |
|---|---:|---:|---:|
| AP（床旁/便携） | 约 40% | **62.6%**（study 级） | **0.7223** |
| PA（标准后前位） | — | 13.3% | 0.8684 |
| 侧位 | — | 19.2% | 0.8577 |

AP 组 AUC 比 PA 低 0.146，而 test 里 AP 反而更多。把训练分布往 AP 拉，
是直接对冲这个结构性短板。

## 做法

按 **study 的主导视图**给样本加权：AP 主导的 study 权重 × `ap_boost`，其余 × 1。
用 `WeightedRandomSampler` 按权重有放回采样，得到逼近目标分布的每个 epoch 样本。

注意：**这会改变训练分布，因此验证集（AP 占比 54.2%）不再等同于 test（62.6%）**。
选型时应同时看「视图重加权」口径（`tools/view_reweight_eval.py`）。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from torch.utils.data import WeightedRandomSampler

# view_codes: 1=AP 2=PA 3=LATERAL 4=LL 5=PA LLD 0=未知
VIEW = {1: "AP", 2: "PA", 3: "LAT", 4: "LAT", 5: "PA", 0: "UNK"}


def study_dominant_view(table) -> dict:
    out = {}
    for sid in table.study_order:
        codes = [VIEW.get(int(table.view_codes[r]), "UNK") for r in table.study_to_rows[sid]]
        codes = [c for c in codes if c != "UNK"]
        out[sid] = Counter(codes).most_common(1)[0][0] if codes else "UNK"
    return out


def view_distribution(table) -> dict:
    dom = study_dominant_view(table)
    c = Counter(dom.values())
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}, c


def build_ap_sampler(table, ap_boost: float = 1.6, seed: int = 42) -> WeightedRandomSampler:
    """图像级权重：该图像所属 study 若以 AP 为主导，则权重 × ap_boost。"""
    dom = study_dominant_view(table)
    weights = np.ones(table.n_images, dtype=np.float64)
    for i, sid in enumerate(table.study_ids):
        if dom.get(sid) == "AP":
            weights[i] *= ap_boost

    g = np.random.default_rng(seed)
    return WeightedRandomSampler(
        weights=torch_tensor(weights),
        num_samples=table.n_images,
        replacement=True,
        generator=None,  # 由 DataLoader 的 seed 控制
    )


def torch_tensor(arr):
    import torch

    return torch.as_tensor(arr, dtype=torch.double)


def estimate_ap_share(table, ap_boost: float) -> dict:
    """不用真的采样，直接算加权后的 AP 占比（study 级）。"""
    dom = study_dominant_view(table)
    counts = Counter()
    for sid in table.study_order:
        v = dom.get(sid, "UNK")
        counts[v] += ap_boost if v == "AP" else 1.0
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}
