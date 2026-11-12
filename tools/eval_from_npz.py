"""从缓存 npz 直接算集成 val（含患者平滑），不用重跑推理。

用法：
    python tools/eval_from_npz.py \
        --npz outputs/preds/valid_exp066_s896_24ep_best_896_tta.npz \
              outputs/preds/valid_exp054_t896_a05_best_896_tta.npz \
        --lam 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import load_split  # noqa: E402
from src.train.metrics import SCORED_CLASSES, aggregate_study_probs, macro_auc  # noqa: E402


def smooth(sp: np.ndarray, groups: dict, lam: float) -> np.ndarray:
    out = sp.copy()
    for _k, rows in groups.items():
        if len(rows) < 2:
            continue
        p = sp[rows]
        total = p.sum(axis=0)
        for j, r in enumerate(rows):
            others = (total - p[j]) / (len(rows) - 1)
            out[r] = (1 - lam) * p[j] + lam * others
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    v = load_split("valid")
    y = v.study_labels()
    df = v.df
    sid = "study_id" if "study_id" in df.columns else "Study_id"
    sub = "subject_id" if "subject_id" in df.columns else "Subject_id"
    smap = dict(df.groupby(sid)[sub].first())
    subjects = [str(smap[s]) for s in v.study_order]
    g: dict[str, list[int]] = {}
    for i, s in enumerate(subjects):
        g.setdefault(s, []).append(i)
    groups = {k: np.asarray(vv) for k, vv in g.items()}

    lgs = [np.load(ROOT / p if not Path(p).is_absolute() else Path(p))["logits"] for p in args.npz]
    for p, lg in zip(args.npz, lgs):
        sp1 = aggregate_study_probs(lg, v.study_to_rows, v.study_order, method="logit_mean", logits=lg)
        print(f"  单个 {Path(p).stem[:44]:44s} 9类={macro_auc(y, sp1, SCORED_CLASSES):.4f}")

    fused = np.mean(lgs, axis=0)
    sp = aggregate_study_probs(fused, v.study_to_rows, v.study_order, method="logit_mean", logits=fused)
    a_raw = macro_auc(y, sp, SCORED_CLASSES)
    sp_sm = smooth(sp, groups, args.lam)
    a_sm = macro_auc(y, sp_sm, SCORED_CLASSES)
    print(f"=== 融合 {len(lgs)} 路 {args.tag} ===")
    print(f"  无平滑 9类 = {a_raw:.4f}")
    print(f"  含平滑 9类 = {a_sm:.4f}  (λ={args.lam})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
