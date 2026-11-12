"""Unicode 安全的图像读写工具。

Windows 上 `cv2.imread` / `cv2.imwrite` 走的是窄字符 API，
遇到含中文的路径（本项目根目录就含"人工智能挑战赛"）会直接返回 None，
且只打印一条容易忽略的 WARN。这里统一改成 `np.fromfile` + `cv2.imdecode`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_gray(path) -> np.ndarray:
    """读取为单通道 uint8 数组。"""
    path = Path(path)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"无法读取文件: {path}") from exc
    if buf.size == 0:
        raise RuntimeError(f"文件为空: {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 极少数非标准 JPEG 交给 PIL 兜底
        from PIL import Image

        with Image.open(path) as im:
            img = np.array(im.convert("L"))
    return img


def imwrite(path, img: np.ndarray) -> None:
    """写入图像（Unicode 安全）。"""
    path = Path(path)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))
