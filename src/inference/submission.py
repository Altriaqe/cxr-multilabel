"""submission.csv 生成与格式自检。

赛题要求（官方通知 §七.2）：
  * 文件名为 submission.csv，UTF-8 编码；
  * 列为 Study_id, Predict_class, Probability；
  * Predict_class 为 1-10 的类别 id（官方编号，从 1 开始）；
  * Probability ∈ [0, 1]；
  * 一次检查可输出多行。

**关键决策**：每次检查输出全部 10 个类别，而不是只输出概率 >= 0.5 的行。
理由：主指标是 macro-AUC，AUC 需要每个类别的完整概率排序信息。
若只交高分行，缺失位置会被判分脚本按 0 补齐，低分区间全部并列，
AUC 会被系统性低估；若判分脚本自行按 0.5 阈值算 F1，输出全量也不受影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.constants import CLASS_NAMES_ZH


def build_submission(
    study_probs: np.ndarray,
    study_order: Sequence[str],
    out_path,
    clip: float = 1e-6,
) -> pd.DataFrame:
    """把 (n_studies, 10) 概率矩阵展开成官方要求的长表。"""
    probs = np.clip(np.asarray(study_probs, dtype=np.float64), clip, 1.0 - clip)
    n_studies, n_cls = probs.shape
    assert n_cls == 10, f"期望 10 个类别，实际 {n_cls}"

    df = pd.DataFrame(
        {
            "Study_id": np.repeat(np.asarray(study_order, dtype=object), n_cls),
            "Predict_class": np.tile(np.arange(1, n_cls + 1, dtype=np.int64), n_studies),
            "Probability": probs.reshape(-1),
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 用 utf-8-sig 还是 utf-8？官方要求 UTF-8。纯 ASCII 内容下两者字节一致（无 BOM 更安全）
    df.to_csv(out_path, index=False, encoding="utf-8", float_format="%.6f")
    return df


def validate_submission(path, expected_studies: Optional[Sequence[str]] = None,
                        rows_per_study: int = 10) -> Dict[str, object]:
    """格式自检，返回报告字典；发现硬伤直接抛异常。"""
    path = Path(path)
    df = pd.read_csv(path, dtype={"Study_id": str})
    report: Dict[str, object] = {}
    report["rows"] = len(df)
    report["columns"] = list(df.columns)

    expected_cols = ["Study_id", "Predict_class", "Probability"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"列名不符：{list(df.columns)}，应为 {expected_cols}")

    if df.isna().any().any():
        raise ValueError("存在缺失值")

    if not df["Predict_class"].between(1, 10).all():
        raise ValueError(f"Predict_class 越界：{[int(v) for v in df['Predict_class'].unique()]}")

    if not df["Probability"].between(0.0, 1.0).all():
        raise ValueError(
            f"Probability 越界：[{df['Probability'].min()}, {df['Probability'].max()}]"
        )

    n_studies = df["Study_id"].nunique()
    report["n_studies"] = int(n_studies)
    report["rows_per_study"] = report["rows"] / max(n_studies, 1)

    if expected_studies is not None:
        exp = set(expected_studies)
        got = set(df["Study_id"].unique())
        if exp != got:
            raise ValueError(
                f"Study_id 不匹配：缺少 {len(exp - got)} 个，多出 {len(got - exp)} 个；"
                f"示例 missing={list(exp - got)[:3]} extra={list(got - exp)[:3]}"
            )
        report["study_ids_match"] = True

    dup = df.duplicated(subset=["Study_id", "Predict_class"]).sum()
    if dup:
        raise ValueError(f"存在 {dup} 条 (Study_id, Predict_class) 重复记录")

    raw = path.read_bytes()
    report["has_bom"] = raw[:3] == b"\xef\xbb\xbf"
    report["size_mb"] = round(len(raw) / 1024 ** 2, 3)

    # 概率分布概览（用于排查"全部输出同一个值"这类退化情况）
    stats = df.groupby("Predict_class")["Probability"].agg(["mean", "min", "max"]).round(4)
    report["per_class_prob_mean"] = stats["mean"].to_dict()
    return report


def format_report(report: Dict[str, object]) -> str:
    lines = [
        "=" * 66,
        "submission.csv 自检",
        "=" * 66,
        f"  行数            : {report['rows']}",
        f"  study 数        : {report['n_studies']}",
        f"  每 study 行数   : {report['rows_per_study']:.1f}",
        f"  列名            : {report['columns']}",
        f"  文件大小        : {report['size_mb']} MB",
        f"  带 BOM          : {report['has_bom']}",
    ]
    if report.get("study_ids_match"):
        lines.append("  Study_id 覆盖   : 与 test.csv 完全一致 ✓")
    lines.append("-" * 66)
    lines.append(f"  {'id':>3}  {'类别':<12} {'平均概率':>10}")
    for cid in range(1, 11):
        m = report["per_class_prob_mean"].get(cid, float("nan"))
        lines.append(f"  {cid:>3}  {CLASS_NAMES_ZH[cid-1]:<12} {m:>10.4f}")
    lines.append("=" * 66)
    return "\n".join(lines)
