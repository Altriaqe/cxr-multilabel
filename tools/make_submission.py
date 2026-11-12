"""生成 submission.csv（与包内 cost.py 完全同一条推理路径，**不含患者平滑**）。

## 为什么不含患者平滑
11-11 21:34 官方答疑：「每个 study 端到端计时，**每个目标 Study 重算其依赖的同患者
前向**、全链路计入该 Study 端到端耗时」。测试集平均 10.4 study/患者，
患者平滑会让目标 study 的耗时放大到患者规模倍（>500ms），必然超 100ms。
因此交付模型不含患者聚合，每个 study 的预测只依赖它自己的图像。

## 用法
    python tools/make_submission.py --data cxr_first/test --out outputs/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cost import XrayClassifier, infer_study, list_studies, load_members  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cxr_first/test")
    ap.add_argument("--members", default="members.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    members = load_members(ROOT / args.members, device)
    studies = list_studies(ROOT / args.data)
    print(f"members={len(members)}  studies={len(studies)}  "
          f"images={sum(len(p) for _, _, p in studies)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("Study_id,Predict_class,Probability\n")
        for i, s in enumerate(studies, 1):
            probs = infer_study(s, members, device)
            for c, p in enumerate(probs, start=1):
                f.write(f"{s[0]},{c},{p:.6f}\n")
            if i % 200 == 0:
                print(f"  {i}/{len(studies)}")
    print(f"写出 {out}（{len(studies)} study / {len(studies)*10} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
