"""CSV 标注解析与 study 级索引构建。

数据事实（已由 tools/audit_data.py 全量核实）：
  * CSV 每行 = 一张图片，`Dicom_id` 即文件名（去扩展名）。
  * 标签列 `Predict_class` 是分号分隔的类别 id 字符串，如 "7;9"；空值表示无标注（仅 test）。
  * 同一 `Study_id` 下所有行的标签完全一致 —— 标签本质是 study 级。
  * 图片路径为 <split>/<Subject_id>/<Study_id>/<Dicom_id>.jpg。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.constants import CSV_FILES, NUM_CLASSES, SPLITS


def parse_label_str(value) -> List[int]:
    """把 "7;9" 解析成 [6, 8]（0-based 索引）。"""
    if pd.isna(value):
        return []
    out = []
    for tok in str(value).split(";"):
        tok = tok.strip()
        if tok:
            cid = int(tok)
            if not 1 <= cid <= NUM_CLASSES:
                raise ValueError(f"类别 id 越界: {cid} in {value!r}")
            out.append(cid - 1)
    return sorted(set(out))


def labels_to_vector(ids: Sequence[int], num_classes: int = NUM_CLASSES) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    for i in ids:
        vec[i] = 1.0
    return vec


@dataclass
class SplitTable:
    """一个划分的完整结构化表示。"""

    name: str
    df: pd.DataFrame                       # 原始表 + 派生列
    image_paths: List[Path]                # 与 df 行序一一对应
    labels: Optional[np.ndarray]           # (n_images, 10) float32；test 为 None
    study_ids: np.ndarray                  # (n_images,) object
    subject_ids: np.ndarray                # (n_images,) object
    view_codes: np.ndarray                 # (n_images,) int8，0=未知
    orient_codes: np.ndarray               # (n_images,) int8，0=未知
    study_order: List[str] = field(default_factory=list)          # 唯一 study，按首次出现排序
    study_to_rows: Dict[str, np.ndarray] = field(default_factory=dict)

    # -------------------------------------------------- study 级视图
    @property
    def n_images(self) -> int:
        return len(self.df)

    @property
    def n_studies(self) -> int:
        return len(self.study_order)

    def study_row_indices(self, study_id: str) -> np.ndarray:
        return self.study_to_rows[study_id]

    def study_labels(self) -> np.ndarray:
        """(n_studies, 10) study 级标签；要求标签在 study 内一致（已核实）。"""
        if self.labels is None:
            raise RuntimeError(f"{self.name} 没有标签")
        return np.stack([self.labels[self.study_to_rows[s][0]] for s in self.study_order])

    def study_label_matrix(self) -> np.ndarray:
        return self.study_labels()


def _encode(values: pd.Series, vocabulary: Sequence[str]) -> np.ndarray:
    lut = {v: i + 1 for i, v in enumerate(vocabulary)}  # 0 保留给"未知/缺失"
    return np.array([lut.get(v, 0) if isinstance(v, str) else 0 for v in values], dtype=np.int8)


def load_split(name: str, csv_path: Optional[Path] = None, image_root: Optional[Path] = None) -> SplitTable:
    """读取并结构化一个划分。"""
    csv_path = Path(csv_path or CSV_FILES[name])
    image_root = Path(image_root or SPLITS[name])
    df = pd.read_csv(csv_path)

    required = {"Dicom_id", "Subject_id", "Study_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    for col in ("Dicom_id", "Subject_id", "Study_id"):
        df[col] = df[col].astype(str)

    # 路径
    image_paths = [
        image_root / s / st / f"{d}.jpg"
        for s, st, d in zip(df["Subject_id"], df["Study_id"], df["Dicom_id"])
    ]

    # 标签
    if "Predict_class" in df.columns:
        label_ids = [parse_label_str(v) for v in df["Predict_class"]]
        labels = np.stack([labels_to_vector(v) for v in label_ids]).astype(np.float32)
    else:
        label_ids = None
        labels = None

    # 元数据编码
    view_codes = (
        _encode(df["ViewPosition"], ["AP", "PA", "LATERAL", "LL", "PA LLD"])
        if "ViewPosition" in df.columns else np.zeros(len(df), dtype=np.int8)
    )
    orient_codes = (
        _encode(df["PatientOrientation"], ["Erect", "Recumbent"])
        if "PatientOrientation" in df.columns else np.zeros(len(df), dtype=np.int8)
    )

    study_ids = df["Study_id"].to_numpy(dtype=object)
    study_order = list(dict.fromkeys(study_ids.tolist()))
    tmp = pd.Series(np.arange(len(study_ids)), index=study_ids)
    grouped = pd.DataFrame({"row": np.arange(len(study_ids)), "study": study_ids}).groupby("study")["row"]
    study_to_rows = {k: v.to_numpy() for k, v in grouped}

    table = SplitTable(
        name=name,
        df=df,
        image_paths=image_paths,
        labels=labels,
        study_ids=study_ids,
        subject_ids=df["Subject_id"].to_numpy(dtype=object),
        view_codes=view_codes,
        orient_codes=orient_codes,
        study_order=study_order,
        study_to_rows=study_to_rows,
    )
    if label_ids is not None:
        table.df["label_ids"] = label_ids
    # ★ 研究用：CSV 若带 cache_row 列，表示这一行对应全量缓存的第几行。
    #   这样多个划分可以共享同一份大缓存，不必各自复制一份。
    if "cache_row" in df.columns:
        table.row_map = df["cache_row"].to_numpy(dtype=np.int64)
    return table


def verify_split(table: SplitTable, check_files: bool = False) -> Dict[str, object]:
    """一致性自检，返回摘要字典。"""
    stats: Dict[str, object] = {
        "split": table.name,
        "n_images": table.n_images,
        "n_studies": table.n_studies,
        "n_subjects": len(set(table.subject_ids.tolist())),
    }
    if check_files:
        missing = [p for p in table.image_paths if not p.is_file()]
        stats["missing_images"] = len(missing)
        stats["missing_examples"] = [str(p) for p in missing[:5]]
    if table.labels is not None:
        # study 内标签一致性
        inconsistent = 0
        for s, rows in table.study_to_rows.items():
            if len(rows) > 1:
                sub = table.labels[rows]
                if not np.allclose(sub, sub[0]):
                    inconsistent += 1
        stats["inconsistent_studies"] = inconsistent
        stats["class_pos_rate"] = table.study_labels().mean(axis=0).round(4).tolist()
    return stats
