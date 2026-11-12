"""把指定的若干成员等权融合成一份提交，并报告融合后的 val（校准用）。

## 为什么单独写一个

`ensemble_search.py` 会穷举所有子集，适合探索；但今天已经明确**不再加同类成员**
（`docs/开始局_2026-10-24.md` §E3：3→4 路只值 +0.10 分），只需要把指定的几只
融合成一个文件。这个脚本更短、更快、输出更直接。

## 关键：缓存

每个成员的 TTA 预测会缓存到 `outputs/preds/{split}_{ckpt_stem}_{size}_tta.npz`。
已存在的直接复用，所以重复运行是秒级的。

## 用法

    python tools/fuse_members.py ^
        --member "r50=checkpoints/exp001_resnet50_512_best.pth@512@576" ^
        --member "dn121=checkpoints/exp002_densenet121_512_best.pth@512@576" ^
        --member "cvx=checkpoints/exp010_convnext_t_best.pth@512@576" ^
        --member "cvx_ap=checkpoints/exp012_convnext_ap/best.pth@512@576" ^
        --member "dn121_ap=checkpoints/exp013_dn121_ap_best.pth@512@576" ^
        --out outputs/submission_5way.csv

## 口径

val 一律按官方 9 类（第 10 类不计入）+ TTA + logit_mean，与提交口径一致。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CACHE_DIR, NUM_CLASSES, OUTPUT_DIR  # noqa: E402
from src.data.dataset import XrayImageDataset, worker_init_fn  # noqa: E402
from src.data.manifest import load_split  # noqa: E402
from src.data.sampler import study_dominant_view  # noqa: E402
from src.data.transforms import build_any_eval_transform, build_any_tta_transforms  # noqa: E402
from src.inference.predictor import predict_dataset_with_tta, predict_loader  # noqa: E402
from src.inference.submission import build_submission, validate_submission  # noqa: E402
from src.models.model import build_model  # noqa: E402
from src.train.metrics import SCORED_CLASSES, aggregate_study_probs, macro_auc  # noqa: E402

PRED_DIR = OUTPUT_DIR / "preds"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_logits(ckpt: Path, size: int, cache_size: int, split: str, device, tta: bool = True):
    tag = f"{split}_{ckpt.stem}_{size}" + ("_tta" if tta else "")
    cache = PRED_DIR / f"{tag}.npz"
    if cache.exists():
        log(f"  缓存命中 {tag}")
        return np.load(cache)["logits"]

    table = load_split(split)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "resnet50")
    model = build_model(arch=arch, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device).eval()

    ds = XrayImageDataset(table, transform=build_any_eval_transform(arch, size),
                          cache_path=CACHE_DIR / f"{split}_{cache_size}.npy",
                          with_labels=(split != "test"))
    log(f"  推理 {ckpt.stem} @{split} size={size} tta={tta}")
    t0 = time.time()
    if tta:
        tts = [tf for _, tf in build_any_tta_transforms(arch, size)]
        _, lg = predict_dataset_with_tta(model, ds, device, tts, use_hflip=True,
                                         batch_size=8, num_workers=2, return_logits=True)
    else:
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                            pin_memory=True, worker_init_fn=worker_init_fn)
        _, lg = predict_loader(model, loader, device, n_out=NUM_CLASSES, return_logits=True)
    log(f"    用时 {time.time()-t0:.0f}s")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=lg)
    del model
    torch.cuda.empty_cache()
    return lg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True,
                    help="名称=权重路径@输入尺寸@缓存尺寸")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "submission_fused.csv"))
    ap.add_argument("--no-val", action="store_true", help="跳过验证集评估（只出提交）")
    ap.add_argument("--dump-valid", default=None,
                    help="把 valid 的 study 级概率写到该路径（用于按患者结构分层评估）")
    ap.add_argument("--no-tta", action="store_true",
                    help="关闭 TTA（交付用：6 路 TTA 会把延迟放大 6 倍，超 100ms）")
    args = ap.parse_args()
    use_tta = not args.no_tta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    for spec in args.member:
        name, _, rest = spec.partition("=")
        parts = rest.split("@")
        ckpt = Path(parts[0])
        size = int(parts[1]) if len(parts) > 1 and parts[1] else 512
        csz = int(parts[2]) if len(parts) > 2 and parts[2] else 576
        members.append((name.strip(), ckpt, size, csz))

    missing = [n for n, p, _, _ in members if not p.exists()]
    if missing:
        log(f"权重缺失: {missing}")
        return 2

    # ---------------------------------------------------------------- 验证集
    if not args.no_val:
        v = load_split("valid")
        y = v.study_labels()
        dom = study_dominant_view(v)
        groups = np.array([dom.get(s, "UNK") for s in v.study_order])

        log("=== 各成员 val（9类+TTA）===")
        val_lgs, names = [], []
        for name, ckpt, size, csz in members:
            lg = get_logits(ckpt, size, csz, "valid", device, tta=use_tta)
            sp = aggregate_study_probs(lg, v.study_to_rows, v.study_order,
                                       method="logit_mean", logits=lg)
            a9 = macro_auc(y, sp, SCORED_CLASSES)
            ap_v = macro_auc(y[groups == "AP"], sp[groups == "AP"], SCORED_CLASSES)
            log(f"  {name:<12} 整体9类={a9:.4f}   AP={ap_v:.4f}")
            val_lgs.append(lg)
            names.append(name)

        fused = np.mean(val_lgs, axis=0)
        sp = aggregate_study_probs(fused, v.study_to_rows, v.study_order,
                                   method="logit_mean", logits=fused)

        # 可选：把 valid 的 study 级概率导出，供分层评估（按患者结构加权）使用
        if getattr(args, "dump_valid", None):
            from pathlib import Path as _P
            dp = _P(args.dump_valid)
            dp.parent.mkdir(parents=True, exist_ok=True)
            with open(dp, "w", encoding="utf-8", newline="") as f:
                f.write("Study_id,Predict_class,Probability\n")
                for sid, probs in zip(v.study_order, sp):
                    for c, p in enumerate(probs, start=1):
                        f.write(f"{sid},{c},{p:.6f}\n")
            log(f"  已导出 valid 概率 → {dp}")
        a9 = macro_auc(y, sp, SCORED_CLASSES)
        a10 = macro_auc(y, sp, None)
        ap_v = macro_auc(y[groups == "AP"], sp[groups == "AP"], SCORED_CLASSES)
        pa_v = macro_auc(y[groups == "PA"], sp[groups == "PA"], SCORED_CLASSES)
        lat_v = macro_auc(y[groups == "LAT"], sp[groups == "LAT"], SCORED_CLASSES)
        log(f"=== {len(names)} 路等权融合 ===")
        log(f"  整体 9类 = {a9:.4f}   （10类参考 {a10:.4f}）")
        log(f"  AP={ap_v:.4f}  PA={pa_v:.4f}  LAT={lat_v:.4f}")

    # ---------------------------------------------------------------- 测试集
    log("=== 生成测试集提交 ===")
    t = load_split("test")
    test_lgs = []
    for name, ckpt, size, csz in members:
        test_lgs.append(get_logits(ckpt, size, csz, "test", device, tta=use_tta))
    fused_t = np.mean(test_lgs, axis=0)
    sp_t = aggregate_study_probs(fused_t, t.study_to_rows, t.study_order,
                                 method="logit_mean", logits=fused_t)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_submission(sp_t, t.study_order, out)
    rep = validate_submission(out, expected_studies=t.study_order)
    ok = (rep["rows"] == 14140 and rep["n_studies"] == 1414
          and rep.get("study_ids_match") and not rep["has_bom"])
    log(f"  写出 {out}")
    log(f"  {rep['rows']} 行 / {rep['n_studies']} study / Study_id一致={rep.get('study_ids_match')} "
        f"/ BOM={rep['has_bom']}  -> {'校验通过' if ok else '校验失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
