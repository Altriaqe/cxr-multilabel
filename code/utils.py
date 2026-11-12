"""图像读取、预处理与目录扫描工具（predict.py / cost.py 共用）。

关键设计：
* **解码在 GPU（nvJPEG）**、**缩放与归一化也在 GPU** —— 在评测机（8 核 Xeon）
  上比 CPU 解码 + cv2.resize 快约 25ms/study。
* **不做任何跨 study 预取**（官方明令禁止）：每个 study 的解码、前向都在它
  自己的计时窗口内完成；只允许 study 内部多图并行。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

NUM_CLASSES = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_POOL = ThreadPoolExecutor(max_workers=4)
_MEAN_T = None
_STD_T = None


# ------------------------------------------------------------------ 读取
def imread_gray(path) -> np.ndarray:
    """完整解码为灰度图（uint8）。"""
    import cv2
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取文件: {path}")
    return img


def read_raw(path: Path, fast: bool = True) -> np.ndarray:
    """读图。fast=True 时用 1/2 缩小解码（仅 CPU 路径用）。"""
    if fast:
        import cv2
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is not None:
            return img
    return imread_gray(str(path))


def gpu_decode(paths: list[Path]) -> list[torch.Tensor]:
    """用 nvJPEG 在 GPU 上解码整批图（一个 study 的所有图一次解码）。"""
    from torchvision.io import ImageReadMode, decode_jpeg

    bufs = [torch.frombuffer(bytearray(Path(p).read_bytes()), dtype=torch.uint8)
            for p in paths]
    try:
        out = decode_jpeg(bufs if len(bufs) > 1 else bufs[0],
                          mode=ImageReadMode.GRAY, device="cuda")
        if isinstance(out, torch.Tensor):
            out = list(out) if out.dim() == 4 else [out]
        return [t[0] if t.dim() == 3 else t for t in out]
    except Exception:                       # 尺寸不一致等情况：逐张解码
        outs = []
        for b in bufs:
            t = decode_jpeg(b, mode=ImageReadMode.GRAY, device="cuda")
            outs.append(t[0] if t.dim() == 3 else t)
        return outs


# ------------------------------------------------------------------ 预处理
def resize_norm_on_gpu(raws, image_size: int, device) -> torch.Tensor:
    """GPU 上：缩放 → 灰度转 3 通道 → ImageNet 归一化，返回 (N,3,S,S)。"""
    global _MEAN_T, _STD_T
    if _MEAN_T is None or _MEAN_T.device != device:
        _MEAN_T = torch.tensor(MEAN, dtype=torch.float32, device=device).view(3, 1, 1)
        _STD_T = torch.tensor(STD, dtype=torch.float32, device=device).view(3, 1, 1)
    outs = []
    for r in raws:
        if isinstance(r, np.ndarray):
            r = torch.from_numpy(np.ascontiguousarray(r))
        x = r.to(device, non_blocking=True).float().div_(255.0)
        x = x.view(1, 1, x.shape[0], x.shape[1])
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = torch.nn.functional.interpolate(
                x, size=(image_size, image_size), mode="bilinear",
                align_corners=False, antialias=True)
        x = x.view(image_size, image_size).unsqueeze(0).repeat(3, 1, 1)
        outs.append((x - _MEAN_T) / _STD_T)
    return torch.stack(outs)


# ------------------------------------------------------------------ 目录
def list_studies(root: Path) -> list[tuple[str, str, list[Path]]]:
    """扫描目录 → [(study_id, subject_id, [图片路径...]), ...]。

    支持 `root/subject_*/study_*/*.jpg` 与扁平 `root/study_*/*.jpg` 两种结构。
    """
    root = Path(root)
    studies = []
    subs = sorted(p for p in root.iterdir() if p.is_dir())
    for sub in subs:
        inner = sorted(p for p in sub.iterdir() if p.is_dir())
        if not inner:
            continue
        imgs_here = [q for q in sub.iterdir()
                     if q.is_file() and q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs_here:                                   # 扁平结构
            studies.append((sub.name, sub.name, sorted(imgs_here)))
            continue
        for st in inner:                                # subject/study 结构
            imgs = sorted(q for q in st.iterdir()
                          if q.is_file() and q.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if imgs:
                studies.append((st.name, sub.name, imgs))
    return studies


def start_thread_pool():
    """预热线程池（避免首个 study 承担建池开销）。"""
    list(_POOL.map(lambda x: x, range(4)))
