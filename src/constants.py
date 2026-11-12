"""赛题常量定义。

类别 id 采用官方编号（1-10），代码内部一律使用 0-9 索引，
两者只在"读 CSV"和"写 submission"两处发生转换。
"""

from pathlib import Path

# ---------------------------------------------------------------- 环境
# huggingface.co 在本机网络不可达，timm 的预训练权重改走国内镜像。
# 用 setdefault：如果外部已经显式设置了 HF_ENDPOINT（例如官方镜像内），不覆盖。
import os as _os

_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
# 允许用环境变量切换数据集/缓存目录（研究用：把 NIH 数据接进同一套训练管线）
DATA_DIR = Path(_os.environ.get("CXR_DATA_DIR") or (ROOT / "cxr_first"))
CACHE_DIR = Path(_os.environ.get("CXR_CACHE_DIR") or (ROOT / "data" / "cache"))
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "outputs"

SPLITS = {
    "train": DATA_DIR / "train",
    "valid": DATA_DIR / "valid",
    "test": DATA_DIR / "test",
}
CSV_FILES = {
    "train": DATA_DIR / "train.csv",
    "valid": DATA_DIR / "valid.csv",
    "test": DATA_DIR / "test.csv",
}

# ---------------------------------------------------------------- 类别
NUM_CLASSES = 10

# 官方 id 1-10 -> 索引 0-9
CLASS_NAMES_EN = [
    "Enlarged Cardiomediastinum",  # 1
    "Pneumothorax",                # 2
    "Consolidation",               # 3
    "Pneumonia",                   # 4
    "Edema",                       # 5
    "Cardiomegaly",                # 6
    "Atelectasis",                 # 7
    "Lung Opacity",                # 8
    "Pleural Effusion",            # 9
    "No Finding",                  # 10
]
CLASS_NAMES_ZH = [
    "纵隔心影增大", "气胸", "肺部实变", "肺炎", "肺水肿",
    "心影扩大", "肺不张", "肺部阴影", "胸腔积液", "无异常发现",
]

# "No Finding" 在官方标注中与其他 9 类严格互斥（实测共现次数 = 0）
NO_FINDING_IDX = 9
FINDING_INDICES = list(range(9))

# ---------------------------------------------------------------- 字段
VIEW_POSITIONS = ["AP", "PA", "LATERAL", "LL", "PA LLD"]
ORIENTATIONS = ["Erect", "Recumbent"]
