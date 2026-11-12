"""训练循环。

要点：
  * AMP 混合精度 + 梯度累积（8GB 显存下把等效 batch 拉到大 batch 的水平）；
  * 验证时先把图像级预测聚合成 **study 级** 再做 macro-AUC —— 与线上口径一致；
  * 权重衰减只作用于权重矩阵，不作用于 norm/bias（对照 ConvNeXt / ViT 的常规配方）。
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.metrics import (SCORED_CLASSES, aggregate_study_probs, evaluate, format_report)


# ------------------------------------------------------------------ 工具
def build_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine(optimizer, warmup_epochs: float, total_epochs: int, min_ratio: float = 0.02):
    """线性 warmup + 余弦退火。"""
    def fn(step: int):
        # step 以 epoch 为单位
        if step < warmup_epochs:
            return max(1e-3, (step + 1) / max(warmup_epochs, 1e-6))
        progress = (step - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-6)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


# ------------------------------------------------------------------ Trainer
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        device: torch.device,
        val_table=None,
        agg_method: str = "mean",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        warmup_epochs: float = 1.0,
        grad_accum: int = 1,
        grad_clip: float = 1.0,
        amp: bool = True,
        amp_dtype: str = "fp16",
        ema_decay: float = 0.0,
        val_fp32: bool = True,
        no_val: bool = False,
        ckpt_dir: Path = Path("checkpoints"),
        log_csv: Optional[Path] = None,
        run_name: str = "run",
        freeze_backbone_epochs: int = 0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mixup_prob: float = 0.5,
        distill_alpha: float = 0.0,
        teacher_probs=None,
        class_names=None,
        max_train_steps: Optional[int] = None,
        show_progress: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.val_table = val_table
        self.agg_method = agg_method
        self.epochs = epochs
        self.grad_accum = max(1, grad_accum)
        self.grad_clip = grad_clip
        self.amp = amp and device.type == "cuda"
        # ---------------------------------------------------------- 混合精度类型
        # fp16 的指数范围窄（最大 65504），某些结构会在前向里溢出并产生 inf 梯度，
        # 于是 GradScaler 每一步都判定为溢出而**跳过参数更新**。表现极具欺骗性：
        #   train loss 正常（0.30~0.40）、不报错、不崩溃，但 val AUC 恒等于 0.5000、
        #   F1 恒等于 0.0000，因为权重从头到尾一步没动。
        # ConvNeXt 就是典型受害者：它的 GRN（Global Response Norm）里要做
        #   torch.norm(x, p=2) 再除以自身均值，fp16 下极易 inf/inf=nan。
        #
        # bf16 的指数范围与 fp32 相同，不会溢出，因此**不需要 GradScaler**，
        # 且 Ada/Ampere 及以上架构原生支持，吞吐与 fp16 相当。
        # 遇到"loss 不动、AUC 恒为 0.5"的训练，第一件事就该查这个。
        self.amp_dtype_name = str(amp_dtype).lower()
        if self.amp and self.amp_dtype_name in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False          # bf16 无需梯度缩放
        else:
            self.amp_dtype = torch.float16
            self.use_scaler = self.amp
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.mixup_prob = float(mixup_prob)
        self.distill_alpha = float(distill_alpha)
        self.teacher_probs = (teacher_probs.to(device) if teacher_probs is not None else None)
        # 必须与索引（GPU）同设备，否则 PyTorch 报 "indices should be either on cpu or on the same device"
        self.class_names = class_names
        self.max_train_steps = max_train_steps
        self.show_progress = show_progress

        self.optimizer = AdamW(build_param_groups(model, weight_decay), lr=lr)
        self.scheduler = warmup_cosine(self.optimizer, warmup_epochs, epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.best_auc = -1.0

        self.best_ema_auc = -1.0

        self.no_val = bool(no_val)   # True: 跳过验证（valid 已并入训练时用）
        self.val_fp32 = bool(val_fp32)      # 验证是否强制 fp32（见 predict_loader 说明）
        self.ema_decay = float(ema_decay)

        self.ema_state = None

        if self.ema_decay > 0:

            self._ema_init()
        self.history: List[Dict] = []
        self.log_csv = Path(log_csv) if log_csv else None
        if self.log_csv:
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_loss", "val_auc", "val_f1", "lr", "elapsed_s", "is_best"]
                )

    # -------------------------------------------------------------- 训练
    def _set_backbone_grad(self, requires: bool) -> None:
        for name, p in self.model.named_parameters():
            if name.startswith("backbone") and "head" not in name:
                p.requires_grad = requires

    def _mix_batch(self, imgs: torch.Tensor, labels: torch.Tensor):
        """对一个 batch 施加 Mixup 或 CutMix（多标签：软标签直接线性混合）。

        官方《补充说明》第六条明确允许 Mixup/CutMix，推理时不做任何混合。
        `mixup_alpha = cutmix_alpha = 0` 时本函数原样返回（默认关闭，行为不变）。
        """
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:
            return imgs, labels
        if random.random() >= self.mixup_prob:
            return imgs, labels

        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or random.random() < 0.5
        )
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(imgs.size(0), device=imgs.device)

        if use_cutmix:
            h, w = imgs.shape[-2:]
            cut_rat = math.sqrt(max(0.0, 1.0 - lam))
            cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
            cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
            y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
            x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
            if y2 > y1 and x2 > x1:
                imgs = imgs.clone()
                imgs[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
                lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
        else:
            imgs = lam * imgs + (1.0 - lam) * imgs[perm]

        labels = lam * labels + (1.0 - lam) * labels[perm]
        return imgs, labels

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.freeze_backbone_epochs and epoch < self.freeze_backbone_epochs:
            self._set_backbone_grad(False)
        else:
            self._set_backbone_grad(True)

        running, n_steps = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{self.epochs} [train]",
                    leave=False, disable=not self.show_progress, mininterval=30)
        self.optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels, _idx) in enumerate(pbar):
            if self.max_train_steps and step >= self.max_train_steps:
                break
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            imgs, labels = self._mix_batch(imgs, labels)

            with torch.amp.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
                logits = self.model(imgs)
                if self.distill_alpha > 0.0 and self.teacher_probs is not None:
                    # 软标签来自教师集成；用同一个 BCE（对软目标同样成立）
                    soft = self.teacher_probs[_idx.to(self.device)].to(logits.dtype)
                    loss = ((1.0 - self.distill_alpha) * self.criterion(logits, labels)
                            + self.distill_alpha * self.criterion(logits, soft))
                else:
                    loss = self.criterion(logits, labels)
                loss_scaled = loss / self.grad_accum

            self.scaler.scale(loss_scaled).backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema_state is not None and not self.no_val:
                    self._ema_update()
                self.optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            n_steps += 1
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{running / n_steps:.4f}")

        return running / max(n_steps, 1)

    # -------------------------------------------------------------- 验证
    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, n_out: int):
        """返回 (probs, logits)。logits 供 study 级 logit_mean 融合使用。

        **验证默认走 fp32**（`self.val_fp32`，配置项 `train.val_fp32`，默认 True）。

        为什么验证不用 AMP：bf16 累加精度低，在弱模型（AUC ~0.72、大量近似并列）上
        会把排序扰动出千分之几。实测同一个 checkpoint 反复重算：

            fp32  benchmark=False : 0.718713
            fp32  benchmark=True  : 0.718713   <- 完全一致
            bf16  benchmark=False : 0.717694
            bf16  benchmark=True  : 0.717795
            bf16  Trainer 路径    : 0.719252   <- 三次散布极差 0.0026

        合规要求「完整训练日志 + 每轮 checkpoint 哈希」并支持重训核验，
        所以 `hashes.csv` 的 val_auc 必须任何人拿到权重都能精确复现 —— bf16 做不到。
        代价是每轮验证多约 1 分钟（约 10 分钟/轮的 10%）。
        """
        self.model.eval()
        probs = np.zeros((len(loader.dataset), n_out), dtype=np.float32)
        logits_out = np.zeros((len(loader.dataset), n_out), dtype=np.float32)
        use_amp = self.amp and not self.val_fp32
        for imgs, _labels, idx in tqdm(loader, desc="  [val]", leave=False,
                                       disable=not self.show_progress):
            imgs = imgs.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
                logits = self.model(imgs)
            probs[idx.numpy()] = torch.sigmoid(logits).float().cpu().numpy()
            logits_out[idx.numpy()] = logits.float().cpu().numpy()
        return probs, logits_out

    # -------------------------------------------------------------- EMA
    def _ema_init(self) -> None:
        self.ema_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def _ema_update(self) -> None:
        """每次优化器更新后调用，把权重向当前权重靠一步。

        EMA（指数滑动平均）把训练轨迹上的权重做平均，能显著压掉"最后一轮恰好
        没训好"的风险。本项目实测过这个痛点：ConvNeXt 在 ep9 到 0.8060 之后，
        ep10-14 一路跌到 0.788~0.794；DN121 也是 ep12 见顶后连跌。
        靠 best.pth 只保住最好那一轮，而 EMA 是全轨迹的平滑，通常比单轮最好更稳。
        """
        d = self.ema_decay
        for k, v in self.model.state_dict().items():
            cur = v.detach()
            if cur.dtype.is_floating_point:
                self.ema_state[k].mul_(d).add_(cur, alpha=1.0 - d)
            else:                                   # num_batches_tracked 之类的整数
                self.ema_state[k].copy_(cur)

    def validate(self) -> Dict[str, object]:
        if self.val_loader is None or self.val_table is None:
            return {"macro_auc": float("nan")}
        img_probs, img_logits = self.predict_loader(self.val_loader, self.model.num_classes)
        study_probs = aggregate_study_probs(
            img_probs, self.val_table.study_to_rows, self.val_table.study_order,
            method=self.agg_method, logits=img_logits,
        )
        y = self.val_table.study_labels()
        # 官方口径：第 10 类「无异常发现」不计入 macro-AUC（与其他 9 类互斥）。
        # 选模必须按计分的 9 类来，否则 best.pth 可能是"第 10 类特别强、真正计分的
        # 9 类很差"的那一轮。实测第 10 类与 9 类的强弱排序几乎是反的：
        #   R50_s43：c10 0.8774（最高）/ 9类 0.7861（最低）
        #   ConvNeXt：c10 0.8604（最低）/ 9类 0.7994（最高）
        return evaluate(y, study_probs, class_names=self.class_names,
                        scored_classes=SCORED_CLASSES)

    def validate_ema(self) -> Dict[str, object]:
        """临时换入 EMA 权重做一次验证，结束后恢复原权重。"""
        if self.ema_state is None or self.val_loader is None:
            return {"macro_auc": float("nan")}
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_state, strict=True)
        try:
            return self.validate()
        finally:
            self.model.load_state_dict(live, strict=True)

    # -------------------------------------------------------------- 主循环
    def fit(self) -> Dict[str, object]:
        t0 = time.time()
        best: Dict[str, object] = {}
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            if self.no_val:
                # valid 已并入训练（规则第 3 条），不得再用它选模/早停/调参。
                # 这里只存每轮权重，跑满固定轮数后取最后一轮当 best。
                metrics = {"macro_auc": float("nan"), "macro_f1@0.5": float("nan")}
                auc = float("nan")
                f1 = float("nan")
            else:
                metrics = self.validate()
                auc = float(metrics.get("macro_auc", float("nan")))
                f1 = float(metrics.get("macro_f1@0.5", float("nan")))

            # EMA 权重单独评一次；更优就存成 ema.pth，并在日志里标出来
            ema_auc, ema_f1 = float("nan"), float("nan")
            if self.ema_state is not None and not self.no_val:
                em = self.validate_ema()
                ema_auc = float(em.get("macro_auc", float("nan")))
                ema_f1 = float(em.get("macro_f1@0.5", float("nan")))
                if ema_auc > self.best_ema_auc:
                    self.best_ema_auc = ema_auc
                    self.save_checkpoint(epoch, em, "ema.pth", state=self.ema_state)

            is_best = True if self.no_val else (auc > self.best_auc)
            if is_best:
                self.best_auc = auc if auc == auc else self.best_auc
                best = metrics
                self.save_checkpoint(epoch, metrics, "best.pth")

            elapsed = time.time() - t0
            mark = "  *best*" if is_best else ""
            ema_mark = f"  ema={ema_auc:.4f}" if ema_auc == ema_auc else ""
            print(
                f"[{self.run_name}] epoch {epoch+1:>2}/{self.epochs}  loss={train_loss:.4f}  "
                f"valAUC={auc:.4f}  valF1={f1:.4f}  lr={lr_now:.2e}  {elapsed/60:.1f}min"
                f"{mark}{ema_mark}",
                flush=True,
            )
            self.save_epoch_checkpoint(epoch, metrics)
            self.history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_auc": auc, "val_f1": f1,
                 "ema_auc": ema_auc, "lr": lr_now, "elapsed_s": round(elapsed, 1),
                 "is_best": int(is_best)}
            )
            if self.log_csv:
                with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [epoch + 1, f"{train_loss:.5f}", f"{auc:.5f}", f"{f1:.5f}",
                         f"{lr_now:.3e}", round(elapsed, 1), int(is_best)]
                    )

        # no_val 模式下没有完整 metrics（只有 nan 占位），不能走 format_report
        if self.no_val or not best or "n_studies" not in best:
            print(f"\n[{self.run_name}] no_val 模式：不评估验证集，最后一轮权重已存为 *_best.pth")
        else:
            print("\n" + format_report(best, self.class_names or [f"c{i+1}" for i in range(10)]))
        return best

    # -------------------------------------------------------------- 审计留存
    def save_epoch_checkpoint(self, epoch: int, metrics: Dict[str, object]) -> None:
        """每轮保存一份权重 + 记录 SHA-256。

        为什么每轮都留：赛事合规审查要求提供「每轮 checkpoint 的时间戳及文件哈希」
        与「完整训练日志」，用于一致性审计与重训复现。只留 best.pth 无法证明训练
        过程真实发生过，也无法排除事后替换权重的嫌疑。

        目录结构：
            checkpoints/<run_name>/epochNN.pth
            checkpoints/<run_name>/hashes.csv   # epoch,timestamp,sha256,val_auc

        磁盘成本：约 27~106 MB/轮，20 轮约 0.5~2 GB。
        """
        import hashlib

        d = self.ckpt_dir / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"epoch{epoch + 1:02d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "arch": getattr(self.model, "arch", None),
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        csv_path = d / "hashes.csv"
        new_file = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            if new_file:
                f.write("epoch,timestamp,sha256,val_auc\n")
            f.write(
                f"{epoch + 1},{time.strftime('%Y-%m-%d %H:%M:%S')},{sha},"
                f"{float(metrics.get('macro_auc', float('nan'))):.6f}\n"
            )

    def save_checkpoint(self, epoch: int, metrics: Dict[str, object], name: str,
                        state: Optional[dict] = None) -> None:
        path = self.ckpt_dir / f"{self.run_name}_{name}"
        torch.save(
            {
                "epoch": epoch,
                "arch": getattr(self.model, "arch", None),
                "model_state_dict": state if state is not None else self.model.state_dict(),
                "metrics": metrics,
                "best_auc": self.best_auc,
                "ema_auc": self.best_ema_auc,
            },
            path,
        )
