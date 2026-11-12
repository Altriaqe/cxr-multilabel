"""把交付权重从 fp32 转存为 fp16，用于缩小交付包（上传失败时的退路）。

## 为什么需要
平台上传偶发失败（分片上传的某一片断连）。交付包 274MB 分成多片，
任何一片失败整体失败。权重 fp32→fp16 后包体减半（约 156MB），分片数减半。

## 影响
- `load_state_dict` 会把 fp16 值 `copy_` 进 fp32 参数（自动向上转型），模型仍是 fp32，
  推理时照样走 `autocast(fp16)`，**不影响运行方式和合规性**。
- 但权重被舍入到 fp16 精度（相对误差 ~1e-3），**推理输出会变**：
  必须用新权重重跑 `cost.py` 生成 submission.csv，并重新提交一枪。
- 申报清单里的权重哈希要同步更新。

## 用法
    python tools/shrink_checkpoints.py --src checkpoints/exp066_s896_24ep_best.pth \
        checkpoints/exp054_t896_a05_best.pth --out-dir dist/ckpt_fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out-dir", default="dist/ckpt_fp16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_before = total_after = 0
    for src in args.src:
        p = Path(src)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        # 只转浮点张量（BN 的 num_batches_tracked 是 int64，保持原样）
        new_sd = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                new_sd[k] = v.half()
            else:
                new_sd[k] = v
        ck["model_state_dict"] = new_sd
        ck["_fp16_storage"] = True
        dst = out_dir / p.name
        torch.save(ck, dst)
        b, a = p.stat().st_size, dst.stat().st_size
        total_before += b
        total_after += a
        print(f"{p.name}: {b/1024/1024:.1f} MB → {a/1024/1024:.1f} MB")

    print(f"\n合计 {total_before/1024/1024:.1f} MB → {total_after/1024/1024:.1f} MB "
          f"（节省 {(total_before-total_after)/1024/1024:.1f} MB）")
    print(f"输出目录: {out_dir}")
    print("\n下一步：用新权重跑 cost.py 生成 CSV（必须重跑，数值会变）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
