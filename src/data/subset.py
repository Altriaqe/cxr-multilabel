"""按 study 数量截取训练集（用于数据量扫描实验）。

用法：
    from src.data.subset import subset_studies
    small = subset_studies(load_split("train"), 5000)
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.data.manifest import SplitTable


def subset_studies(table: SplitTable, n_studies: int) -> SplitTable:
    """只保留前 n_studies 个 study（按 study_order 的顺序）。

    Args:
        table: 原始划分
        n_studies: 保留的 study 数；<=0 或 >= 总数时原样返回

    Returns:
        新的 SplitTable，study_order / study_to_rows 都按子集重建
    """
    if n_studies <= 0 or n_studies >= table.n_studies:
        return table

    keep: List[str] = list(table.study_order[:n_studies])
    rows = np.sort(np.concatenate([table.study_to_rows[s] for s in keep]))

    study_ids = table.study_ids[rows]
    study_to_rows = {}
    for s in keep:
        study_to_rows[s] = np.where(study_ids == s)[0]

    new_table = SplitTable(
        name=f"{table.name}_sub{n_studies}",
        df=table.df.iloc[rows].reset_index(drop=True),
        image_paths=[table.image_paths[i] for i in rows],
        labels=None if table.labels is None else table.labels[rows],
        study_ids=study_ids,
        subject_ids=table.subject_ids[rows],
        view_codes=table.view_codes[rows],
        orient_codes=table.orient_codes[rows],
        study_order=keep,
        study_to_rows=study_to_rows,
    )
    # 子集第 i 行对应全量缓存第 rows[i] 行；供 XrayImageDataset(row_map=...) 使用
    new_table.row_map = rows
    return new_table
