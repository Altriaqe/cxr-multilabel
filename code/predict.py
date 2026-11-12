"""推理入口（官方附录 A 要求的 predict.py）。

## 职责
1. `load_models()`：从 `model/` + `members.json` 载入所有推理成员，**这是唯一的模型加载路径**
2. `infer_study()`：**这是唯一的推理路径**（单 study → 10 类概率）
3. `predict()`：对一个图片目录批量推理，返回 {study_id: 概率向量}
4. 命令行：`python code/predict.py <图片目录> [--out submission.csv]`

## 与 cost.py 的关系（第 7 条要求）
`cost.py` **必须** `from predict import load_models, infer_study`，
即耗时脚本与预测脚本共用同一套模型加载与推理函数——本文件就是那套函数。

## 关于患者级聚合
**本实现不含任何患者级聚合**。官方 11-11 21:34 答疑：
「每个 study 端到端计时，每个目标 Study 重算其依赖的同患者前向、
全链路计入该 Study 端到端耗时」。
测试集平均 10.4 study/患者，做患者平滑会让单个 study 的耗时放大到患者规模倍
（>500 ms），远超 100 ms 上限，因此交付模型对每个 study 独立推理。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # 材料包根目录
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import XrayClassifier                       # noqa: E402
from utils import (NUM_CLASSES, gpu_decode, list_studies,  # noqa: E402
                   read_raw, resize_norm_on_gpu, start_thread_pool)

FAST = True


# ------------------------------------------------------------------ 模型加载
def load_models(device=None, members_file: str | Path | None = None) -> list[dict]:
    """载入推理成员。`members.json` 里每个成员的 ckpt 路径相对于材料包根目录。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(members_file) if members_file else (ROOT / "members.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    members = []
    for m in cfg:
        ck_path = Path(m["ckpt"])
        if not ck_path.is_absolute():
            ck_path = ROOT / ck_path
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = XrayClassifier(arch=ck.get("arch") or m.get("arch"),
                               num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()                     # ★ eval 模式（第 6 条要求）
        for p in model.parameters():
            p.requires_grad_(False)
        members.append({
            "name": m.get("name", ck_path.stem),
            "model": model,
            "image_size": int(m.get("image_size", 896)),
            "cache_size": int(m.get("cache_size", m.get("image_size", 896))),
        })
    return members


# ------------------------------------------------------------------ 推理
def infer_study(study, members, device) -> np.ndarray:
    """单个 study → 10 类概率（只依赖该 study 自己的图像）。"""
    _sid, _subj, paths = study
    if FAST and device.type == "cuda":
        try:
            raws = gpu_decode(paths)
        except Exception:
            raws = [read_raw(p, FAST) for p in paths]
    else:
        raws = [read_raw(p, FAST) for p in paths]

    with torch.inference_mode():
        logits_sum = None
        for m in members:
            batch = resize_norm_on_gpu(raws, m["image_size"], device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = m["model"](batch)
            member_logit = out.float().mean(dim=0)          # study 内多图平均
            logits_sum = member_logit if logits_sum is None else logits_sum + member_logit
        fused = logits_sum / len(members)                   # 成员等权平均
        probs = torch.sigmoid(fused)
    return probs.detach().cpu().numpy().astype(np.float64)


def predict(data_dir: str | Path, members: list[dict] | None = None,
            device=None) -> tuple[list[tuple[str, np.ndarray]], list[float]]:
    """对一个目录做批量推理，返回 ([(study_id, probs)], [每 study 耗时 ms])。"""
    import time
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if members is None:
        members = load_models(device)
    start_thread_pool()
    studies = list_studies(Path(data_dir))
    rows, times = [], []
    for s in studies:
        t0 = time.perf_counter()
        p = infer_study(s, members, device)
        times.append((time.perf_counter() - t0) * 1000.0)
        rows.append((s[0], p))
    return rows, times


def write_submission(rows, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("Study_id,Predict_class,Probability\n")
        for sid, probs in rows:
            for c, p in enumerate(probs, start=1):
                f.write(f"{sid},{c},{p:.6f}\n")


# ------------------------------------------------------------------ CLI
def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python code/predict.py <图片目录> [--out submission.csv]",
              file=sys.stderr)
        return 2
    data_dir = sys.argv[1]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = load_models(device)
    print(f"[predict] members={len(members)}  device={device}")
    rows, times = predict(data_dir, members, device)
    print(f"[predict] {len(rows)} study，平均 {np.mean(times):.1f} ms/study")
    if out:
        write_submission(rows, out)
        print(f"[predict] 写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
