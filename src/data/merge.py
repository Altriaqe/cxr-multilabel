"""把多个 SplitTable 合并成一张表（用于 train+valid 合训）。

## 为什么需要

规则第 3 条允许把 valid 并入训练，但并入后就**不能再拿 valid 选模/早停/调参**。
合训能多 859 个 study（+4.6% 数据），是 §E5 里优先级第 2 的手段。

## 三条不能省的约束（写在这里供审计核对）

1. **行序必须严格是「先全部 train、再全部 valid」**。
   `MergedCacheDataset` 靠 `idx < n_train` 判断该去哪个缓存文件、并减偏移。
   一旦顺序被打乱，标签会静默错位——不报错、loss 照降，只是模型在学错东西。

2. **`study_to_rows` 必须在合并后的表上重建**。
   否则同一个 study 的图像可能落在两个数据集的名字空间里，study 级聚合会串号。

3. **合训后禁止用该表做任何选模**。
   本模块只负责拼表；`train.py` 的 `--train-with-valid` 会同时强制 `no_val`。

## study 名冲突

train 与 valid 是不同患者（已核实交集为 0），`Study_id` 不会重复。
构造时仍做断言，防止将来数据变动后静默出错。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.data.manifest import SplitTable


def merge_tables(tables: Sequence[SplitTable], name: str = "merged") -> SplitTable:
    """按给定顺序拼接多个 SplitTable（行序 = 依次拼接）。

    只做「拼接 + 重建 study 索引」，不做任何重排。
    """
    if not tables:
        raise ValueError("至少需要一个表")

    first = tables[0]
    dfs = [t.df for t in tables]
    image_paths: List = []
    labels: List[np.ndarray] = []
    study_ids: List[np.ndarray] = []
    subject_ids: List[np.ndarray] = []
    view_codes: List[np.ndarray] = []
    orient_codes: List[np.ndarray] = []

    for t in tables:
        if t.labels is None:
            raise ValueError(f"{t.name} 没有标签，不能用于合训")
        image_paths.extend(t.image_paths)
        labels.append(t.labels)
        study_ids.append(t.study_ids)
        subject_ids.append(t.subject_ids)
        view_codes.append(t.view_codes)
        orient_codes.append(t.orient_codes)

    import pandas as pd

    df = pd.concat(dfs, axis=0, ignore_index=True)
    labels_arr = np.concatenate(labels, axis=0)
    study_ids_arr = np.concatenate(study_ids, axis=0)
    subject_ids_arr = np.concatenate(subject_ids, axis=0)
    view_arr = np.concatenate(view_codes, axis=0)
    orient_arr = np.concatenate(orient_codes, axis=0)

    if len(image_paths) != len(df) or len(labels_arr) != len(df):
        raise RuntimeError(
            f"拼接后长度不一致: paths={len(image_paths)} df={len(df)} labels={len(labels_arr)}"
        )

    merged = SplitTable(
        name=name,
        df=df,
        image_paths=image_paths,
        labels=labels_arr,
        study_ids=study_ids_arr,
        subject_ids=subject_ids_arr,
        view_codes=view_arr,
        orient_codes=orient_arr,
    )

    # ---- 重建 study 索引（按首次出现排序，与 SplitTable 的约定一致）----
    order: List[str] = []
    seen = {}
    for i, sid in enumerate(study_ids_arr):
        if sid not in seen:
            seen[sid] = []
            order.append(sid)
        seen[sid].append(i)
    merged.study_order = order
    merged.study_to_rows = {s: np.asarray(seen[s], dtype=np.int64) for s in order}

    # ---- 一致性断言：禁止 study 名跨表重复，否则聚合会串号 ----
    all_sids = [set(t.study_order) for t in tables]
    for i in range(len(all_sids)):
        for j in range(i + 1, len(all_sids)):
            dup = all_sids[i] & all_sids[j]
            if dup:
                raise RuntimeError(
                    f"{tables[i].name} 与 {tables[j].name} 有 {len(dup)} 个重复 Study_id，"
                    f"例：{list(dup)[:3]}。合训前必须先处理，否则标签会串号。"
                )

    return merged


def build_train_with_valid(name: str = "train+valid") -> SplitTable:
    """加载并合并 train 与 valid（顺序：train 全部 → valid 全部）。"""
    from src.data.manifest import load_split

    train = load_split("train")
    valid = load_split("valid")
    return merge_tables([train, valid], name=name)

def build_holdout_split(holdout_studies: int = 1000, seed: int = 42, name: str = "train_minus_holdout"):
    """从 train 里切出 holdout，返回 (训练表, holdout 表)。

    ## 为什么要这样做

    把 valid 并入训练后，就**再也没有干净的验证集**了 —— 今天的 exp015 就是教训：
    合训后本地无法评估，只能靠提交去赌。

    正确做法是切一个内部 holdout：

        train(18,513 study) ──┬── holdout(1,000 study)  ← 不参与训练，只做监控/选模
                              └── 其余 17,513 study
                                    ＋ valid 859 study  ← 并入训练

    这样既拿到了 valid 的 4.6% 额外数据，又保留了一个**训练时完全没见过**的监控集。

    ## 注意

    * holdout 按 study 抽（不是按图像），否则同一 study 的多视图会跨集合泄漏。
    * 训练侧的行序 = 先 train 剩余、后 valid 全部；配合 cache_segments 使用。
    * 抽取用固定 seed，保证可复现（审计要求）。
    """
    import numpy as np

    from src.data.manifest import load_split

    train = load_split("train")
    valid = load_split("valid")

    rs = np.random.default_rng(seed)
    order = list(train.study_order)
    rs.shuffle(order)
    hold_ids = set(order[:holdout_studies])

    keep_rows = np.array([i for i, sid in enumerate(train.study_ids) if sid not in hold_ids])
    hold_rows = np.array([i for i, sid in enumerate(train.study_ids) if sid in hold_ids])

    def subset(rows, nm):
        sub = SplitTable(
            name=nm,
            df=train.df.iloc[rows].reset_index(drop=True),
            image_paths=[train.image_paths[i] for i in rows],
            labels=train.labels[rows],
            study_ids=train.study_ids[rows],
            subject_ids=train.subject_ids[rows],
            view_codes=train.view_codes[rows],
            orient_codes=train.orient_codes[rows],
        )
        o, seen = [], {}
        for i, sid in enumerate(sub.study_ids):
            if sid not in seen:
                seen[sid] = []
                o.append(sid)
            seen[sid].append(i)
        sub.study_order = o
        sub.study_to_rows = {s: np.asarray(seen[s], dtype=np.int64) for s in o}
        return sub

    keep = subset(keep_rows, f"{name}")
    hold = subset(hold_rows, "holdout")
    merged = merge_tables([keep, valid], name=f"{name}+valid")
    return merged, hold, keep, valid
