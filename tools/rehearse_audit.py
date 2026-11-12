"""审核演练：构造"完整患者组"的测速目录，验证 cost.py 与 submission.csv 的 4 位小数一致性。

## 为什么需要
官方说明「同患者下提供全部 study」，而官方的 pictures_100 样例只给了 10 个 study，
患者组不完整 → cost.py 会按降级规则输出未平滑值，与 submission.csv 不同。
审核时如果给的是完整患者组，两者必须一致。

本脚本从 test split 里挑几个"多 study 患者"，把他们**全部** study 的图复制到临时目录，
然后跑 cost.py，与 submission.csv 逐条比对（按 4 位小数口径）。

用法：
    python tools/rehearse_audit.py --n-patients 4
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "cxr_first" / "test"
SUB_CSV = ROOT / "outputs" / "submission_2x896_fp16_2026-11-06.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=4)
    ap.add_argument("--out-dir", default="tmp/audit_probe")
    ap.add_argument("--sub", default=str(SUB_CSV))
    args = ap.parse_args()

    out = ROOT / args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 找一个多 study 的测试患者
    subs: dict[str, list[Path]] = {}
    for subj in sorted(TEST_DIR.iterdir()):
        if not subj.is_dir():
            continue
        studies = [s for s in subj.iterdir() if s.is_dir()]
        subs[subj.name] = studies

    lams = sorted(subs.items(), key=lambda kv: -len(kv[1]))
    chosen = [(k, v) for k, v in lams if len(v) >= 3][: args.n_patients]
    if not chosen:
        print("没有 ≥3 个 study 的患者")
        return 2

    print("选中的患者（按 study 数降序）:")
    n_studies = 0
    for subj, studies in chosen:
        print(f"  {subj}: {len(studies)} 个 study")
        for st in studies:
            dst = out / subj / st.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in st.iterdir():
                if img.suffix.lower() == ".jpg":
                    shutil.copy2(img, dst / img.name)
            n_studies += 1
    print(f"共 {len(chosen)} 个患者 / {n_studies} 个 study（患者组完整）")

    cmd = [sys.executable, "-X", "utf8", "cost.py", "--members", "members.json",
           "--pictures", str(out), "--out", "tmp/audit_probe.csv",
           "--rounds", "3", "--fast-decode", "--patient-lambda", "0.15"]
    print("\n$", " ".join(cmd))
    if subprocess.call([str(c) for c in cmd], cwd=ROOT) != 0:
        return 3

    # 比对（4 位小数口径）
    ref: dict[tuple[str, int], float] = {}
    with open(args.sub, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[(r["Study_id"], int(r["Predict_class"]))] = float(r["Probability"])
    got: dict[tuple[str, int], float] = {}
    with open(ROOT / "tmp" / "audit_probe.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            got[(r["Study_id"], int(r["Predict_class"]))] = float(r["Probability"])

    common = set(ref) & set(got)
    if not common:
        print("❌ 没有交集，Study_id 对不上")
        return 4

    max_raw, max_round = 0.0, 0
    for k in common:
        d = abs(ref[k] - got[k])
        max_raw = max(max_raw, d)
        if round(ref[k], 4) != round(got[k], 4):
            max_round += 1

    print(f"\n比对 {len(common)} 条（{len(common)//10} 个 study × 10 类）")
    print(f"  最大 |Δp|（未舍入）      = {max_raw:.6f}")
    print(f"  4 位小数后不一致的条数    = {max_round}")
    print(f"  判定：{'✅ 一致' if max_raw <= 1e-3 and max_round == 0 else '❌ 不一致'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
