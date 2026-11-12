"""在验证集上计算「多个模型集成 + 患者平滑」的最终分数（不需要 GPU）。

直接读 outputs/preds/valid_*.npz 里的 logits，按比赛口径：
  1) study 内多图 logit 均值 → 2) 成员 logit 均值 → 3) sigmoid
  4) （可选）患者平滑 λ=0.15

用法：
    python tools/eval_ensemble_valid.py                      # 默认 6 成员
    python tools/eval_ensemble_valid.py --member exp051_s896_best_896 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import load_split  # noqa: E402
from src.train.metrics import macro_auc  # noqa: E402

PRED = ROOT / "outputs" / "preds"

DEFAULT = [
    "exp051_s896_best_896",        # S@896 α=0.9   0.8120
    "exp035_distill_t_a09_best_768",  # T09 α=0.9     0.8086
    "exp036_s768_a05_aug_best_768",   # S@aug α=0.5   0.8075
    "exp032_distill_t_best_768",      # T05 α=0.5     0.8049
    "exp010_convnext_t_best_512",     # CVX@512
    "exp014_convnext_ssl_best_512",   # CVX_SSL@512
]


def load_logits(tag: str) -> np.ndarray:
    p = PRED / f"valid_{tag}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p)
    for k in ("logits", "arr_0", "data"):
        if k in z:
            return z[k].astype(np.float64)
    return z[list(z.keys())[0]].astype(np.float64)


def to_study(logits_img: np.ndarray, table) -> np.ndarray:
    return np.stack([logits_img[table.study_to_rows[s]].mean(0) for s in table.study_order])


def smooth(p: np.ndarray, subj: np.ndarray, lam: float) -> np.ndarray:
    out = p.copy()
    for u in np.unique(subj):
        m = subj == u
        if m.sum() < 2:
            continue
        out[m] = (1 - lam) * p[m] + lam * p[m].mean(0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", default=None)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.15)
    args = ap.parse_args()

    members = args.member or DEFAULT
    v = load_split("valid")
    y = v.study_labels()
    subj = np.array([str(v.subject_ids[v.study_to_rows[s][0]]) for s in v.study_order])

    total, ok = None, []
    for tag in members:
        try:
            lg = load_logits(tag)
        except FileNotFoundError:
            print(f"  跳过（无缓存）{tag}")
            continue
        s = to_study(lg, v)
        a = macro_auc(y, 1.0 / (1.0 + np.exp(-s)), 9)
        print(f"  {tag:<34s} val = {a:.4f}")
        total = s if total is None else total + s
        ok.append(tag)

    fused = total / len(ok)
    p = 1.0 / (1.0 + np.exp(-fused))
    a0 = macro_auc(y, p, 9)
    a1 = macro_auc(y, smooth(p, subj, args.lam), 9)
    print()
    print(f"集成（{len(ok)} 成员）         val = {a0:.4f}")
    print(f"集成 + 患者平滑 λ={args.lam}  val = {a1:.4f}  ({a1-a0:+.4f})")
    print(f"→ 线上预估 ≈ {127.8*a1 - 25.9:.2f} 分")

    # 与单模型对比
    best_single = max(
        (macro_auc(y, 1.0 / (1.0 + np.exp(-to_study(load_logits(t), v))), 9), t)
        for t in ok)
    print(f"（最强单成员 {best_single[1]} = {best_single[0]:.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
