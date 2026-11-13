"""数据层：标注解析、图像缓存、增强、Dataset。

刻意保持 __init__ 为空：`cache.py` / `manifest.py` 不依赖 torch，
空包初始化让"只装了 numpy/cv2 的环境"也能跑缓存构建。
使用方请显式导入具体模块，例如 `from src.data.manifest import load_split`。
"""
