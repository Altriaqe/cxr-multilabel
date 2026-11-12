"""图像缓存构建入口。

只依赖 numpy / opencv / pandas，不需要 torch，
因此可以在 torch 还没装好的时候先把缓存建起来。

用法:
    python tools/build_cache.py --size 576 --workers 16
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CACHE_DIR  # noqa: E402
from src.data.cache import build_cache  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="构建 576px 图像 memmap 缓存")
    p.add_argument("--size", type=int, default=576, help="缓存边长（像素）")
    p.add_argument("--workers", type=int, default=16, help="并行进程数")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--force", action="store_true", help="已有缓存也重建")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"缓存目录: {CACHE_DIR}")
    for sp in args.splits:
        build_cache(sp, size=args.size, workers=args.workers, force=args.force)
    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
