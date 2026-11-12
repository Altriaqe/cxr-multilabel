"""穷举成员组合，找 val 最优的集成（含患者平滑）。

只用 outputs/preds/valid_*.npz 缓存，不需要 GPU。
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_ensemble_valid import load_logits, smooth, to_study  # noqa: E402

from src.data.manifest import load_split  # noqa: E402
from src.train.metrics import macro_auc  # noqa: E402

TAGS = [
    "exp066_s896_24ep_best_896",      # S@896 24 轮（当前最强单体，0.8130）
    "exp069_s896_36ep_best_896",      # S@896 36 轮（0.8121）
    # exp063（train+valid 合训）故意排除：它在 valid 上背过答案，local val 0.8312 是假的
    "exp051_s896_best_896",           # S@896 12 轮（0.8112）
    "exp054_t896_a05_best_896",       # T@896 12 轮 α=0.5（0.8111）
    "exp052_t896_best_896",           # T@896 12 轮 α=0.9（0.8107）
    "exp062_dinov3_best_896",         # S@896 DINOv3（0.8105）
    "exp038_s896_aug_best_896",
    "exp036_s768_a05_aug_best_768",
    "exp035_distill_t_a09_best_768",
    "exp032_distill_t_best_768",
    "exp010_convnext_t_best_512",
]

# 各成员权重的体积（MB），用于 500MB 上限判断（官方口径 500×1024×1024 = 524.3MB）
SIZES_MB = {
    "exp066_s896_24ep_best_896": 188.8,
    "exp069_s896_36ep_best_896": 188.8,
    "exp068_s896_v2_best_896": 188.8,
    "exp051_s896_best_896": 188.8,
    "exp038_s896_aug_best_896": 188.8,
    "exp062_dinov3_best_896": 188.8,
    "exp036_s768_a05_aug_best_768": 188.8,
    "exp054_t896_a05_best_896": 106.2,
    "exp052_t896_best_896": 106.2,
    "exp035_distill_t_a09_best_768": 106.2,
    "exp032_distill_t_best_768": 106.2,
    "exp010_convnext_t_best_512": 106.2,
}
LIMIT_MB = 500 * 1024 * 1024 / 1024 / 1024   # = 500 MiB → 500.0
EXTRA_MB = 1.0                                # cost.py + src + configs + 材料

LAM = 0.15


def main() -> int:
    v = load_split("valid")
    y = v.study_labels()
    subj = np.array([str(v.subject_ids[v.study_to_rows[s][0]]) for s in v.study_order])
    lgs = {t: to_study(load_logits(t), v) for t in TAGS}

    res = []
    for k in range(1, 6):
        for combo in itertools.combinations(TAGS, k):
            size = sum(SIZES_MB[t] for t in combo) + EXTRA_MB
            if size > LIMIT_MB:                  # 超过 500MB 的组合作废
                continue
            fused = np.mean([lgs[t] for t in combo], axis=0)
            p = smooth(1.0 / (1.0 + np.exp(-fused)), subj, LAM)
            res.append((macro_auc(y, p, 9), combo, size))
    res.sort(reverse=True, key=lambda x: x[0])

    print("Top 12 组合（含患者平滑 λ=0.15，已过滤 >500MB 的组合）：")
    for a, c, sz in res[:12]:
        names = ", ".join(t.replace("_best", "").replace("exp0", "")[:22] for t in c)
        print(f"  {a:.4f}  n={len(c)}  {sz:5.0f}MB  {names}")
    best_a, best_c, best_sz = res[0]
    print(f"\n最优 {best_a:.4f}（{best_sz:.0f}MB）→ 线上预估 {127.8 * best_a - 26.4:.2f}")
    print("成员:", list(best_c))

    # 只看 2-3 个成员的（交付耗时更安全，≤90ms）
    print("\n--- 2~3 成员的最优组合（耗时更安全）---")
    for a, c, sz in [r for r in res if len(r[1]) <= 3][:6]:
        names = ", ".join(t.replace("_best", "").replace("exp0", "")[:22] for t in c)
        print(f"  {a:.4f}  n={len(c)}  {sz:5.0f}MB  {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
