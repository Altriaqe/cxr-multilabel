"""构建 512/576 像素的 uint8 图像缓存，消除 JPEG 解码瓶颈。

原始图约 300 万像素、单张 1.5MB，每 epoch 重解码 3 万张是训练的主要瓶颈。
预缩放后存成 memmap，随机访问几乎是常数开销。

缓存布局：
    data/cache/<split>_<size>.npy    uint8 memmap, shape (N, size, size)
    data/cache/<split>_<size>.json   元信息

行号与对应 CSV 的行号严格一致，因此可直接用 SplitTable 的行索引访问。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.constants import CACHE_DIR
from src.data.manifest import SplitTable, load_split
from src.utils.imageio import imread_gray

cv2.setNumThreads(1)  # 多进程内部再开线程会互相抢核

_WORKER_PATHS: List[str] = []
_WORKER_SIZE: int = 576


def _init_worker(paths: List[str], size: int) -> None:
    global _WORKER_PATHS, _WORKER_SIZE
    _WORKER_PATHS = paths
    _WORKER_SIZE = size


def _load_one(idx: int) -> tuple:
    path = _WORKER_PATHS[idx]
    img = imread_gray(path)
    if img.shape[0] != _WORKER_SIZE or img.shape[1] != _WORKER_SIZE:
        img = cv2.resize(img, (_WORKER_SIZE, _WORKER_SIZE), interpolation=cv2.INTER_AREA)
    return idx, np.ascontiguousarray(img).tobytes()


def cache_paths(split: str, size: int, cache_dir: Path = CACHE_DIR):
    return cache_dir / f"{split}_{size}.npy", cache_dir / f"{split}_{size}.json"


def build_cache(
    split: str,
    size: int = 576,
    cache_dir: Path = CACHE_DIR,
    workers: int = 16,
    table: Optional[SplitTable] = None,
    force: bool = False,
) -> Path:
    """为某个划分建立 memmap 缓存。返回 .npy 路径。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path, meta_path = cache_paths(split, size, cache_dir)

    if npy_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n") == (table.n_images if table else None) or meta.get("n"):
            print(f"[cache] {split}: 已存在 {npy_path.name} (n={meta['n']})，跳过")
            return npy_path

    table = table or load_split(split)
    paths = [str(p) for p in table.image_paths]
    n = len(paths)

    print(f"[cache] {split}: {n} 张 -> {npy_path} ({n * size * size / 1024**3:.2f} GB)")
    mm = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(n, size, size))

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(paths, size)) as ex:
        for idx, raw in ex.map(_load_one, range(n), chunksize=64):
            mm[idx] = np.frombuffer(raw, dtype=np.uint8).reshape(size, size)
            done += 1
            if done % 2000 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"  {split}: {done}/{n} ({rate:.0f} img/s, eta {eta:.0f}s)", flush=True)

    mm.flush()
    del mm

    meta = {
        "split": split,
        "n": n,
        "size": size,
        "dtype": "uint8",
        "source_csv": str(table.df.attrs.get("source", "")),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "行号与对应 CSV 行号一致；灰度为单通道，等比拉伸到 size x size",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] {split}: 完成，用时 {time.time() - t0:.0f}s")
    return npy_path


def open_cache(split: str, size: int, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    npy_path, _ = cache_paths(split, size, cache_dir)
    if not npy_path.exists():
        raise FileNotFoundError(f"缓存不存在: {npy_path}，请先运行 tools/build_cache.py")
    return np.load(npy_path, mmap_mode="r")


def build_all(size: int = 576, workers: int = 16, splits: Sequence[str] = ("train", "valid", "test")) -> None:
    for sp in splits:
        build_cache(sp, size=size, workers=workers)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="构建图像缓存")
    p.add_argument("--size", type=int, default=576)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sp in args.splits:
        build_cache(sp, size=args.size, workers=args.workers, force=args.force)
