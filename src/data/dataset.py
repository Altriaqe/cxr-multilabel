"""Dataset 实现。

两个关键设计：
1. **按行索引访问**：预先算好 `image_paths` / 标签矩阵，`__getitem__` 是 O(1)。
   旧实现每取一个样本都要对整张表做一次布尔索引（O(N) x 3 次），这是训练吞吐崩溃的主因。
2. **优先读 memmap 缓存**：缓存缺失时回落到直接解码 JPEG，保证任何环境都能跑。
3. **多段缓存**：train+valid 合训时，两张表拼成一张，但缓存仍是两个文件。
   `cache_segments` 按「累计行数」把全局索引映射到 (文件, 文件内偏移)。
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import SplitTable
from src.utils.imageio import imread_gray

cv2.setNumThreads(0)  # DataLoader 多进程时避免线程超额订阅

_MMAP_CACHE: Dict[Tuple[int, str], np.ndarray] = {}
_SHARED_CACHE: Dict[str, torch.Tensor] = {}


def _to_shared(arr: np.ndarray) -> torch.Tensor:
    """把整块 uint8 缓存搬进共享内存，返回可被所有 worker 直接索引的张量。

    share_memory_() 会把张量移到 Windows 的共享内存段（file-backed shared memory），
    fork/spawn 出来的 worker 通过同一段物理内存访问，不再各自 np.load(mmap)。
    同一进程内重复调用走缓存，避免多数据集重复搬 9.4 GB。
    """
    key = f"{id(arr)}"
    t = _SHARED_CACHE.get(key)
    if t is None:
        t = torch.from_numpy(arr)
        t.share_memory_()
        _SHARED_CACHE[key] = t
    return t


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 每个 worker 的初始化。

    必须把 torch 的线程池压到 1：默认会继承 24 个线程，
    8 个 worker 就是 192 条线程抢 32 个逻辑核，
    实测吞吐会从 ~700 img/s 掉到 ~12 img/s。
    """
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def _get_mmap(path: str) -> np.ndarray:
    key = (os.getpid(), path)
    mm = _MMAP_CACHE.get(key)
    if mm is None:
        mm = np.load(path, mmap_mode="r")
        _MMAP_CACHE[key] = mm
    return mm


def _npy_rows_without_mapping(path: str) -> Optional[int]:
    """只读 .npy 文件头拿到行数，**不建立内存映射**。

    为什么必须这样：主进程如果 `np.load(..., mmap_mode='r')` 了缓存，
    Windows 会把这段文件映射记进进程的提交内存（实测 9.38GB 的 train 缓存
    让主进程提交量从 ~1GB 涨到 9.66GB）。而真正读图的是 DataLoader worker，
    主进程根本不需要这份映射。768px 缓存是 17.9GB，主进程再映射一次就会
    直接触发 `WinError 1455 页面文件太小`。
    """
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            # numpy 2.x 已移除私有接口 _read_array_header，必须按版本分派到公开函数
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(f)
            else:
                return None
        return int(shape[0])
    except Exception:
        return None


def _npy_rows_from_meta(path: str) -> Optional[int]:
    """退路：读缓存构建时写的同名 .json 元信息。"""
    try:
        import json
        from pathlib import Path as _P

        meta = _P(path).with_suffix(".json")
        if meta.exists():
            return int(json.loads(meta.read_text(encoding="utf-8"))["n"])
    except Exception:
        return None
    return None


class XrayImageDataset(Dataset):
    """图像级数据集：一行 CSV = 一个样本。

    Args:
        table: `SplitTable`
        transform: torchvision v2 变换（输入 HxW uint8 张量）
        cache_path: 单文件 memmap 缓存；None 或不存在时直接解码 JPEG
        with_labels: 是否返回标签（test 集为 False）
        with_meta: 是否返回 (view_code, orient_code)
        cache_segments: 多段缓存 `[(npy_path, n_rows), ...]`，顺序必须与 table 行序一致。
            给定时忽略 `cache_path`。用于 train+valid 合训（两张表拼一张、缓存仍是两个文件）。

            索引规则：累计前缀 prefix[i] 之前落在第 i 个文件，文件内偏移 = idx - prefix[i]。
            **顺序错乱会静默错位标签**（不报错、loss 照降），所以构造时会校验总行数。
    """

    def __init__(
        self,
        table: SplitTable,
        transform=None,
        cache_path: Optional[Path] = None,
        with_labels: bool = True,
        with_meta: bool = False,
        cache_segments: Optional[list] = None,
        shared: bool = False,
        row_map: Optional[np.ndarray] = None,
    ):
        self.table = table
        # ★ row_map：子集实验用。表的行数可以小于缓存行数，
        #   第 i 行去缓存里取 row_map[i]。不做行数一致性校验。
        self.row_map = None if row_map is None else np.asarray(row_map, dtype=np.int64)
        self.transform = transform
        self.with_labels = with_labels and table.labels is not None
        self.with_meta = with_meta
        self.cache_path = str(cache_path) if cache_path and Path(cache_path).exists() else None
        self.paths = [str(p) for p in table.image_paths]
        self.labels = table.labels
        self.views = table.view_codes
        self.orients = table.orient_codes

        # ---------------- 多段缓存 ----------------
        self.segments = None
        if cache_segments:
            segs = [(str(pth), int(n)) for pth, n in cache_segments]
            total = sum(n for _, n in segs)
            if total != len(table.image_paths):
                raise ValueError(
                    f"cache_segments 总行数 {total} 与表行数 {len(table.image_paths)} 不一致"
                )
            prefix, acc = [], 0
            for pth, n in segs:
                if not Path(pth).exists():
                    raise FileNotFoundError(f"缓存分段文件不存在: {pth}")
                got = _npy_rows_without_mapping(pth)
                if got is not None and got != n:
                    raise ValueError(f"{pth} 实际 {got} 行，声明 {n} 行")
                prefix.append(acc)
                acc += n
            self.segments = {
                "paths": [p for p, _ in segs],
                "prefix": prefix,
                "sizes": [n for _, n in segs],
            }

        if self.cache_path and not self.segments and self.row_map is None:
            # 只读文件头校验行数，不在主进程建立映射（见 _npy_rows_without_mapping 的说明）
            n = _npy_rows_without_mapping(self.cache_path)
            if n is None:
                n = _npy_rows_from_meta(self.cache_path)
            if n is not None and n != len(self.paths):
                raise ValueError(f"缓存行数 {n} 与 CSV 行数 {len(self.paths)} 不一致")

        # ---------------- 共享内存模式 ----------------
        # memmap：每个 worker 各自建立文件映射；shared：所有 worker 索引同一份物理内存。
        # 代价是主进程要一次性把整块缓存读进 RAM（train_576 = 9.38 GB），默认关闭。
        self.shared = bool(shared)
        self._shared_arr = None
        self._shared_segs = None
        if self.shared:
            if self.segments:
                self._shared_segs = [_to_shared(np.load(p)) for p in self.segments["paths"]]
            elif self.cache_path:
                self._shared_arr = _to_shared(np.load(self.cache_path))

    def __len__(self) -> int:
        return len(self.paths)

    def _read(self, idx: int) -> np.ndarray:
        # ★ 子集模式：数据集第 idx 行对应全量缓存的第 row_map[idx] 行
        if self.row_map is not None:
            idx = int(self.row_map[idx])
        if self._shared_segs is not None:
            i = bisect.bisect_right(self.segments["prefix"], idx) - 1
            return self._shared_segs[i][idx - self.segments["prefix"][i]].numpy().copy()
        if self._shared_arr is not None:
            return self._shared_arr[idx].numpy().copy()
        if self.segments:
            i = bisect.bisect_right(self.segments["prefix"], idx) - 1
            local = idx - self.segments["prefix"][i]
            return np.array(_get_mmap(self.segments["paths"][i])[local], copy=True)
        if self.cache_path:
            # memmap 切片是只读视图，copy 一份可写数组，否则 torch 会反复告警
            return np.array(_get_mmap(self.cache_path)[idx], copy=True)
        return imread_gray(self.paths[idx])

    def __getitem__(self, idx: int):
        # 统一加一个通道维：(H, W) -> (1, H, W)，
        # torchvision 的几何变换在 2 维输入上会走错分支（shape[-3:] 解包失败）。
        img = torch.from_numpy(np.ascontiguousarray(self._read(idx))).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.float().div_(255.0)

        out = [img]
        if self.with_labels:
            out.append(torch.from_numpy(self.labels[idx]))
        if self.with_meta:
            out.append(torch.tensor([int(self.views[idx]), int(self.orients[idx])], dtype=torch.long))
        out.append(torch.tensor(idx, dtype=torch.long))
        return tuple(out) if len(out) > 1 else out[0]


def make_cache_path(split: str, size: int, cache_dir: Optional[Path] = None) -> Optional[Path]:
    from src.constants import CACHE_DIR
    cache_dir = Path(cache_dir or CACHE_DIR)
    p = cache_dir / f"{split}_{size}.npy"
    return p if p.exists() else None
