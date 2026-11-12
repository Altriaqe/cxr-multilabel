"""训练主入口。

用法:
    python train.py --config configs/resnet50_512.yaml --run-name exp001_resnet50
    python train.py --config configs/resnet50_512.yaml --smoke      # 冒烟测试

设计说明见 docs/review.md。与旧版 train.py 的关键差别：
  * 数据集按行索引，不再每样本做 O(N) 布尔索引；
  * 标签解析支持 "7;9" 形式的多标签；
  * 验证在 **study 级** 计算 macro-AUC；
  * ReduceLROnPlateau 的 verbose 参数已移除（PyTorch 2.10 不再接受）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CACHE_DIR, CHECKPOINT_DIR, CLASS_NAMES_ZH, LOG_DIR, NUM_CLASSES  # noqa: E402
from src.data.dataset import XrayImageDataset, worker_init_fn  # noqa: E402
from src.data.manifest import load_split  # noqa: E402
from src.data.transforms import (  # noqa: E402
    build_any_eval_transform,
    build_any_train_transform,
    is_med_arch,
)
from src.models.model import build_model, model_size_mb  # noqa: E402
from src.train.loss import build_loss  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Tee:
    """把 stdout 同时写到控制台和文件。

    PowerShell 的 `>` / `*>` / `Tee-Object` 对原生进程输出有块缓冲，
    训练脚本跑到一半都看不到日志；这里在 Python 侧直接落盘，逐行 flush。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def make_loader(ds, batch_size, num_workers, shuffle, pin=True, persistent=True, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="训练胸部 X 光片多标签分类模型")
    ap.add_argument("--config", default="configs/resnet50_512.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个 batch，检查管线是否通")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs")
    ap.add_argument("--max-steps", type=int, default=None, help="每个 epoch 最多跑多少个 batch（调试用）")
    ap.add_argument("--log-file", default=None, help="同时把日志写入该文件（Python 侧直接落盘）")
    ap.add_argument("--train-with-valid", action="store_true",
                    help="把 valid 并入训练（规则第3条：并入后禁止再用它选模，会自动启用 no_val）")
    ap.add_argument("--init-from", default=None,
                    help="用 SSL 预训练权重初始化骨干（只灌 backbone.*，分类头随机）。不传则行为不变")
    ap.add_argument("--subset", type=int, default=None,
                    help="只训练前 N 个 study（数据量扫描实验用）。不传则用全量")
    args = ap.parse_args()

    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        _fh = open(args.log_file, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.stdout, _fh)
        sys.stderr = _Tee(sys.stderr, _fh)
        print(f"[{time.strftime('%H:%M:%S')}] 训练脚本启动")

    cfg = load_config(ROOT / args.config)
    run_name = args.run_name or cfg.get("run_name", "run")
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # 提到高优先级：本机常驻一个占满 ~15 核的 Chrome 进程，普通优先级会被饿死。
    # 在 spawn DataLoader worker 之前设置，子进程会继承。
    from src.utils.system import set_high_priority

    if set_high_priority():
        print("进程优先级已提升为 HIGH")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu = {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
        # 输入尺寸固定，benchmark 让 cudnn 在首个 epoch 选定算法后一直复用
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dcfg = cfg["data"]
    image_size = dcfg.get("image_size", 512)
    cache_size = dcfg.get("cache_size", 576)

    # ------------------------------------------------------------ 数据
    print("加载标注 ...")
    if getattr(args, "train_with_valid", False):
        from src.data.merge import build_train_with_valid
        train_table = build_train_with_valid()
        print(f"合训模式: train+valid 共 {train_table.n_images} 图 / {train_table.n_studies} study")
        print("           valid 已并入训练 -> 自动禁用验证选模（规则第3条）")
    else:
        train_table = load_split("train")
    # ★ 数据量扫描实验：只保留前 N 个 study（不传则行为不变）
    if getattr(args, "subset", None):
        from src.data.subset import subset_studies

        before = train_table.n_studies
        train_table = subset_studies(train_table, args.subset)
        print(f"[subset] 训练集 {before} → {train_table.n_studies} study "
              f"({train_table.n_images} 图)")
    val_table = load_split("valid")
    print(f"train: {train_table.n_images} 图 / {train_table.n_studies} study")
    print(f"valid: {val_table.n_images} 图 / {val_table.n_studies} study")

    train_cache = CACHE_DIR / f"train_{cache_size}.npy"
    val_cache = CACHE_DIR / f"valid_{cache_size}.npy"
    if not train_cache.exists():
        print(f"警告: 缓存 {train_cache} 不存在，将实时解码 JPEG（会慢很多）")

    train_tf = build_any_train_transform(
        cfg['model'].get('arch', 'resnet50'),
        image_size,
        scale=tuple(dcfg.get("scale", (0.80, 1.0))),
        rotate_deg=dcfg.get("rotate", 7.0),
        jitter=dcfg.get("jitter", 0.2),
        erasing_p=dcfg.get("erasing", 0.25),
        ra_ops=int(dcfg.get("ra_ops", 0)),
        ra_mag=int(dcfg.get("ra_mag", 0)),
        crop_ratio=tuple(dcfg.get("crop_ratio", (0.90, 1.10))),
    )
    eval_tf = build_any_eval_transform(cfg['model'].get('arch', 'resnet50'), image_size)

    if getattr(args, "train_with_valid", False):
        # 顺序必须与合并表一致：先 train 全部，再 valid 全部
        _va = load_split("valid")
        train_ds = XrayImageDataset(
            train_table, transform=train_tf, with_labels=True,
            cache_segments=[(train_cache, load_split("train").n_images),
                            (CACHE_DIR / f"valid_{cache_size}.npy", _va.n_images)],
        )
    else:
        train_ds = XrayImageDataset(train_table, transform=train_tf, cache_path=train_cache,
                                    with_labels=True,
                                    row_map=getattr(train_table, "row_map", None))
    val_ds = XrayImageDataset(val_table, transform=eval_tf, cache_path=val_cache,
                              with_labels=True,
                              row_map=getattr(val_table, "row_map", None))

    nw = dcfg.get("num_workers", 8)
    bs = dcfg.get("batch_size", 16)

    # AP 视图过采样：把训练分布往 test 的 AP 占比（62.6%）靠。
    # 说明见 src/data/sampler.py —— AP 组 AUC 比 PA 低约 0.146，而 test 里 AP 更多。
    ap_boost = float(dcfg.get("ap_oversample", 1.0))
    if ap_boost > 1.0:
        from src.data.sampler import build_ap_sampler, estimate_ap_share

        sampler = build_ap_sampler(train_table, ap_boost=ap_boost, seed=seed)
        est = estimate_ap_share(train_table, ap_boost)
        print(f"AP 过采样 boost={ap_boost} -> 训练分布 "
              f"AP={est.get('AP', 0):.1%} PA={est.get('PA', 0):.1%} "
              f"LAT={est.get('LAT', 0):.1%}（test 为 AP 62.6%）")
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler, num_workers=nw,
            pin_memory=True, drop_last=True,
            persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
            worker_init_fn=worker_init_fn,
        )
    else:
        train_loader = make_loader(train_ds, bs, nw, shuffle=True, drop_last=True)

    val_loader = make_loader(val_ds, dcfg.get("eval_batch_size", bs), nw, shuffle=False)

    # 正样本率 -> pos_weight（可选）
    pos_weight = None
    if cfg["train"].get("pos_weight", False):
        rates = train_table.study_labels().mean(axis=0)
        pos_weight = ((1 - rates) / np.clip(rates, 1e-3, 1)).tolist()
        print(f"pos_weight = {[round(w, 2) for w in pos_weight]}")

    criterion = build_loss(cfg["train"].get("loss", "bce"), pos_weight=pos_weight)

    # ------------------------------------------------------------ 模型
    mcfg = cfg["model"]
    model = build_model(
        arch=mcfg.get("arch", "resnet50"),
        num_classes=NUM_CLASSES,
        pretrained=mcfg.get("pretrained", True),
        drop=mcfg.get("drop", 0.4),
        drop_path=mcfg.get("drop_path", 0.1),
        grad_checkpointing=mcfg.get("grad_checkpointing", False),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model = {mcfg.get('arch')}  params = {n_params/1e6:.2f}M  fp32 = {model_size_mb(model):.1f} MB")

    # ------------------------------------------------------- 可选：用 SSL 权重初始化骨干
    # 不传 --init-from 时行为完全不变（保持向后兼容）。
    if getattr(args, "init_from", None):
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        ssl_state = ck.get("model_state_dict", ck)
        backbone = {k: v for k, v in ssl_state.items() if k.startswith("backbone.")}
        if not backbone:
            raise SystemExit(f"{args.init_from} 里没有 backbone.* 参数，不是 SSL 权重？")
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        print(f"SSL 初始化: 载入 {len(backbone)} 个骨干张量  "
              f"未匹配={len(missing)}（应只有分类头）  多余={len(unexpected)}（应为 0）")
        print(f"            来源 {args.init_from}")

    # ------------------------------------------------------------ 训练
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg.get("epochs", 15)

    # ---- 蒸馏：加载教师软标签（仅在 distill_alpha > 0 时）
    _teacher_probs = None
    _dalpha = float(tcfg.get("distill_alpha", 0.0))
    _dpath = tcfg.get("distill_probs", "")
    if _dalpha > 0.0 and _dpath:
        _p = Path(_dpath)
        if not _p.is_absolute():
            _p = ROOT / _p
        _arr = np.load(_p).astype(np.float32)
        _teacher_probs = torch.from_numpy(_arr)
        print(f"蒸馏: alpha={_dalpha}  软标签 {_p}  shape={tuple(_arr.shape)}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        val_table=val_table,
        agg_method=cfg.get("eval", {}).get("agg", "mean"),
        lr=tcfg.get("lr", 3e-4),
        weight_decay=tcfg.get("weight_decay", 1e-4),
        epochs=2 if args.smoke else epochs,
        warmup_epochs=tcfg.get("warmup_epochs", 1.0),
        grad_accum=tcfg.get("grad_accum", 1),
        grad_clip=tcfg.get("grad_clip", 1.0),
        amp=tcfg.get("amp", True),
        amp_dtype=tcfg.get("amp_dtype", "fp16"),
        ema_decay=float(tcfg.get("ema_decay", 0.0)),
                    val_fp32=bool(tcfg.get("val_fp32", True)),
                    no_val=bool(getattr(args, "train_with_valid", False)),
        ckpt_dir=CHECKPOINT_DIR,
        log_csv=LOG_DIR / f"{run_name}_{stamp}.csv",
        run_name=run_name,
        freeze_backbone_epochs=tcfg.get("freeze_backbone_epochs", 0),
        mixup_alpha=float(tcfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(tcfg.get("cutmix_alpha", 0.0)),
        mixup_prob=float(tcfg.get("mixup_prob", 0.5)),
        distill_alpha=float(tcfg.get("distill_alpha", 0.0)),
        teacher_probs=_teacher_probs,
        class_names=CLASS_NAMES_ZH,
        max_train_steps=args.max_steps,
        # 写了日志文件时关掉进度条：tqdm 用 \r 重绘，会把日志撑成几 MB 的乱码
        show_progress=not bool(args.log_file),
    )

    if args.smoke:
        print("=== SMOKE TEST: 只跑 2 个 batch ===")
        imgs, labels, idx = next(iter(train_loader))
        print(f"batch shape={tuple(imgs.shape)} dtype={imgs.dtype} labels={tuple(labels.shape)}")
        print(f"pixel range [{imgs.min():.2f}, {imgs.max():.2f}]  label sum={labels.sum().item():.1f}")
        model = model.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs[:2].to(device))
        print(f"logits shape={tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
        print("=== SMOKE OK ===")
        return 0

    # 保存配置快照
    (LOG_DIR / f"{run_name}_{stamp}_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # 预热数据管线：Windows 上每个 worker 都要重新 import torch + 加载 CUDA DLL，
    # 首次 spawn 实测要几分钟（还会被杀软扫描放大）。先付掉这笔钱，
    # 后面 persistent_workers 会复用进程，epoch 计时才是真实的。
    print("\n预热训练数据管线（首次 spawn 较慢，请稍候）...")
    t_warm = time.time()
    try:
        _ = next(iter(train_loader))
    except StopIteration:
        pass
    print(f"训练数据管线就绪，用时 {time.time() - t_warm:.0f}s\n", flush=True)

    best = trainer.fit()
    (LOG_DIR / f"{run_name}_{stamp}_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n最佳结果已写入 {LOG_DIR / f'{run_name}_{stamp}_best.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
