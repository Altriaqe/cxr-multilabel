"""评估指标：study 级宏平均 AUC / F1。

赛题以 study 为评估单位（`Study_id` 是提交行的主键），因此必须先把
图像级预测聚合成 study 级概率，再计算指标 —— 否则验证集数值与线上不一致。

**不依赖 scikit-learn**：本机 `import sklearn` 实测要 100 秒以上
（杀软逐个扫描 .pyd），而 AUC / F1 都可以用十几行 numpy 写完。
下面 `_rank_average_ties` 的 AUC 与 `sklearn.metrics.roc_auc_score`
在含并列分数时也完全等价（Mann-Whitney U 的秩和形式）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.constants import CLASS_NAMES_ZH, FINDING_INDICES, NO_FINDING_IDX


# ------------------------------------------------------------------ 基础指标
def _rank_average_ties(x: np.ndarray) -> np.ndarray:
    """把分数转成平均秩（1-based），并列取平均秩。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # 处理并列：找到相同分数区间，用区间结束的秩替换
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """单类别 AUC，等价于 sklearn.roc_auc_score。"""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average_ties(y_score)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_pred = np.asarray(y_pred).astype(bool).ravel()
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


# ------------------------------------------------------------------ 聚合
def aggregate_study_probs(
    probs: np.ndarray,
    study_to_rows: Dict[str, np.ndarray],
    study_order: Sequence[str],
    method: str = "mean",
    logits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把 (n_images, C) 的图像级概率聚合为 (n_studies, C)。

    Args:
        probs: 图像级概率 (n_images, C)。
        study_to_rows: study -> 该 study 的图像行索引。
        study_order: 输出顺序。
        method: 'mean' | 'max' | 'lse'(几何平均) | 'logit_mean'
        logits: 仅 method='logit_mean' 时需要。
    """
    n_cls = probs.shape[1]
    out = np.empty((len(study_order), n_cls), dtype=np.float64)
    eps = 1e-6

    for i, sid in enumerate(study_order):
        rows = study_to_rows[sid]
        block = np.asarray(probs[rows], dtype=np.float64)
        if method == "mean":
            out[i] = block.mean(axis=0)
        elif method == "max":
            out[i] = block.max(axis=0)
        elif method == "lse":
            # 概率空间的几何平均：等价于 logit 空间取均值，抗单个视图的极端值
            out[i] = np.exp(np.log(np.clip(block, eps, 1.0)).mean(axis=0))
        elif method == "logit_mean":
            if logits is None:
                raise ValueError("method='logit_mean' 需要传入 logits")
            out[i] = 1.0 / (1.0 + np.exp(-np.asarray(logits[rows], dtype=np.float64).mean(axis=0)))
        else:
            raise ValueError(f"未知聚合方式: {method}")

    return np.clip(out, eps, 1.0 - eps)


def enforce_no_finding_complement(study_probs: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """利用"类别 10 = 其他 9 类的补集"这一实测先验做一致性约束。

    p10' = (1 - weight) * p10 + weight * (1 - max(p1..p9))

    两路证据的线性融合，是否提分必须在验证集上实测。
    """
    out = study_probs.copy()
    max_finding = study_probs[:, FINDING_INDICES].max(axis=1)
    implied = 1.0 - max_finding
    out[:, NO_FINDING_IDX] = (1 - weight) * study_probs[:, NO_FINDING_IDX] + weight * implied
    return out


# ------------------------------------------------------------------ 汇总
# 官方口径（2026-10-24 起）：第 10 类「无异常发现」与其他 9 类互斥，**不计入 macro 平均**。
# 因此线上分数是 9 类的均值，而不是 10 类。历史上我们一直按 10 类算（因为第 10 类 AUC
# 很高、约 0.87~0.88，会把分数抬高约 0.008），这也是 val 与线上系统性偏差的一部分来源。
# 保留 `macro_auc_10` 用于和旧记录对齐，但**选型一律以 9 类为准**。
SCORED_CLASSES = 9


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    return [binary_auc(y_true[:, c], y_prob[:, c]) for c in range(y_true.shape[1])]


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: Optional[int] = None) -> float:
    """宏平均 AUC。

    Args:
        n_classes: 只统计前 n_classes 类。官方口径传 `SCORED_CLASSES`(9)，
                   即排除第 10 类 No Finding；传 None 则统计全部类别。
    """
    aucs = np.array(per_class_auc(y_true, y_prob), dtype=np.float64)
    if n_classes is not None:
        aucs = aucs[:n_classes]
    valid = ~np.isnan(aucs)
    return float(aucs[valid].mean()) if valid.any() else float("nan")


def per_class_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds) -> List[float]:
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)] * y_true.shape[1]
    return [
        binary_f1(y_true[:, c], y_prob[:, c] >= thresholds[c])
        for c in range(y_true.shape[1])
    ]


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds=0.5) -> float:
    return float(np.mean(per_class_f1(y_true, y_prob, thresholds)))


def search_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> tuple:
    """逐类别搜索使 F1 最大的阈值（独立搜索，不改变 AUC）。"""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    best_t = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    for c in range(y_true.shape[1]):
        scores = [binary_f1(y_true[:, c], y_prob[:, c] >= t) for t in grid]
        best_t[c] = float(grid[int(np.argmax(scores))])
    return best_t, macro_f1(y_true, y_prob, best_t)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES_ZH,
    tune_thresholds: bool = True,
    scored_classes: Optional[int] = None,
) -> Dict[str, object]:
    """汇总评估。

    Args:
        scored_classes: 官方口径传 `SCORED_CLASSES`(9)，排除第 10 类 No Finding。
                        `macro_auc` 与 `macro_f1@*` 都按该口径计算；
                        `macro_auc_10` / `macro_f1@0.5` 另存全 10 类结果供与旧记录对比。
    """
    aucs = per_class_auc(y_true, y_prob)
    f1_at_half = per_class_f1(y_true, y_prob, 0.5)
    n = scored_classes
    f1_scored = f1_at_half[:n] if n is not None else f1_at_half
    result: Dict[str, object] = {
        "macro_auc": macro_auc(y_true, y_prob, n),
        "macro_auc_10": macro_auc(y_true, y_prob, None),
        "macro_f1@0.5": float(np.mean(f1_scored)),
        "per_class_auc": aucs,
        "per_class_f1@0.5": f1_at_half,
        "n_studies": int(y_true.shape[0]),
        "scored_classes": n if n is not None else int(y_true.shape[1]),
    }
    if tune_thresholds:
        th, mf1 = search_thresholds(y_true, y_prob)
        result["thresholds"] = th.tolist()
        result["macro_f1@tuned"] = mf1
    return result


def format_report(result: Dict[str, object], class_names: Sequence[str] = CLASS_NAMES_ZH) -> str:
    n = result.get("scored_classes", len(class_names))
    tag = f"{n}类口径" if n != len(class_names) else "全部类别"
    lines = [
        "=" * 74,
        f"  studies={result['n_studies']}   MACRO AUC({tag}) = {result['macro_auc']:.4f}"
        f"   MACRO F1@0.5({tag}) = {result['macro_f1@0.5']:.4f}"
        + (f"   F1@tuned = {result['macro_f1@tuned']:.4f}" if "macro_f1@tuned" in result else ""),
    ]
    if "macro_auc_10" in result:
        lines.append(f"  （参考）10类口径 MACRO AUC = {result['macro_auc_10']:.4f}")
    lines += [
        "=" * 74,
        f"{'id':>3}  {'类别':<12} {'AUC':>8} {'F1@0.5':>8}",
        "-" * 74,
    ]
    for i, name in enumerate(class_names):
        auc = result["per_class_auc"][i]
        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "  n/a "
        skipped = "  (不计入)" if i >= n else ""
        lines.append(f"{i+1:>3}  {name:<12} {auc_s:>8} {result['per_class_f1@0.5'][i]:>8.4f}{skipped}")
    lines.append("=" * 74)
    return "\n".join(lines)
