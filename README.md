# CXR Multi-Label Classification

> 胸部 X 光片 10 类多标签分类的完整实现：训练、蒸馏、集成、患者级后处理，
> 以及一个满足「**≤100 ms/study 端到端** + **≤500 MB 推理权重**」双硬约束的推理包。

本项目源自一场算法竞赛（赛题：胸部 X 光多标签分类，指标为 9 类宏平均 AUC），
赛后整理开源。重点在于**工程约束下的取舍**——不是为了刷到最高分，而是说明在
「小数据 + 严格延迟 + 严格体积」三条线同时收紧时，哪些手段有效、哪些被证伪。

---

## 亮点

* **端到端 60–70 ms/study**（T4 级硬件），其中约 25 ms 的节省来自把 JPEG 解码与缩放全部搬到 GPU
* **推理权重 295 MB**（上限 500 MB），双模型集成
* **受控实验记录**：17 个方向被实测证伪，全部保留数据（见 `docs/技术报告.md`）
* **可复现的计时脚本**：`cost.py` 按官方接口实现，输出 `[CONCLUSION]` 行

---

## 权重下载

微调后的权重**不随仓库分发**，单独托管在 Hugging Face：

```
https://huggingface.co/Altriaqe/cxr-convnext-896-ensemble
```

```bash
pip install -U huggingface_hub
hf download Altriaqe/cxr-convnext-896-ensemble --local-dir weights
```

下载后把 `members.json` 里的 `ckpt` 路径改成本地路径即可：

```json
[{"name": "S896", "ckpt": "weights/exp066_s896_24ep_best.pth",
  "image_size": 896, "cache_size": 896}, ...]
```

> ⚠️ **数据集不公开**。本仓库不附带任何影像数据；训练数据请自行获取，
> 并按 `train.py` 的目录约定组织。

---

## 目录结构

```
.
├── README.md
├── LICENSE                    代码许可证（MIT）
├── requirements.txt
├── cost.py                    计时/一致性脚本（含患者平滑）
├── train.py                   训练入口
├── code/                      推理三件套（被 cost.py 复用）
│   ├── model.py               ConvNeXt + 10 类头
│   ├── predict.py             唯一的模型加载 / 推理路径
│   └── utils.py               GPU 解码、GPU 缩放归一化、目录扫描
├── src/                       训练框架（数据、增强、缓存、指标、Trainer）
├── configs/                   65 个训练配方（含所有被证伪的实验）
├── tools/                     实用脚本（建缓存、生成提交、集成搜索、一致性演练）
├── docs/
│   ├── 技术报告.md              方案、实验、结论、反思
│   └── 训练复现说明.md           逐步复现指南
└── licenses/                  上游许可证与 model card
```

---

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
# 需要 CUDA GPU；训练建议 ≥24 GB 显存，推理 T4 16 GB 足够
```

### 2. 数据准备

本项目**不附带任何数据**。按以下结构放置你自己的数据：

```
<DATA_ROOT>/train/{subject_xxx}/{study_xxx}/*.jpg
<DATA_ROOT>/valid/{subject_xxx}/{study_xxx}/*.jpg
```

并把 `<DATA_ROOT>` 写进 `src/constants.py` 的 `DATA_DIR`（或设置环境变量 `CXR_DATA_DIR`）。

### 3. 建图像缓存（可选，但强烈建议）

```bash
python tools/build_cache.py --size 896 --splits train valid --workers 10
```

缓存是 `uint8` 的 memmap 数组。训练时随机索引，比每次解码 JPEG 快一个量级。

### 4. 训练

```bash
python train.py --config configs/s896_24ep.yaml --run-name my_exp
```

### 5. 生成提交

先写一个 `members.json` 指定参与推理的权重：

```json
[
  {"name": "S896", "ckpt": "checkpoints/exp066_s896_24ep_best.pth",
   "image_size": 896, "cache_size": 896},
  {"name": "T896", "ckpt": "checkpoints/exp054_t896_a05_best.pth",
   "image_size": 896, "cache_size": 896}
]
```

```bash
python tools/make_submission.py --data <DATA_ROOT>/test \
    --members members.json --out submission.csv
```

### 6. 计时与一致性自检

```bash
python cost.py submission.csv <pictures_100> <pictures_5>
```

输出：

```
[CONCLUSION] 结论：100 个样本、batch=1 串行、连续3轮；单样本端到端墙钟时间先按
每个样本3轮取均值，再对100个取平均，最终平均耗时 62.3ms，未超过上限；
预测结果与 submission.csv 比对 100 个样本，不一致0个样本；推理精度fp16。
```

> `code/predict.py` 也可以单独使用：`python code/predict.py <图片目录> --out pred.csv`

---

## 方法

### 模型

| 组件 | 取值                                                                                          |
| ---- | --------------------------------------------------------------------------------------------- |
| 骨干 | ConvNeXt-Small / ConvNeXt-Tiny（timm，ImageNet-12k 预训练）                                   |
| 输入 | 896×896，灰度复制 3 通道，ImageNet 归一化                                                    |
| 损失 | BCE（10 类独立）                                                                              |
| 优化 | AdamW，lr 4e-5，warmup + cosine，bf16 训练 / fp16 推理，batch 8 × 梯度累积 4                 |
| 增强 | RandAugment(2,7)、RandomResizedCrop(0.65–1.0)、旋转 ±10°、ColorJitter、RandomErasing(0.35) |
| 蒸馏 | 3 个 896 模型等权集成生成软标签，α=0.9 / 0.5                                                 |

### 集成与后处理

* **两名成员 logit 等权平均**（实测在 100 ms 预算内的最优组合）
* **患者平滑**：`p' = (1-λ)p + λ·mean(同一 Subject 其他 study 的 p)`，λ=0.15 冻结
  * 单 study 患者退化为 λ=0（不平滑）
  * 属于确定性后处理：不拟合参数、不用测试集统计量、不用标签

### 推理链路

```
JPEG 字节 → nvJPEG 在 GPU 解码（整 study 批量）
         → GPU 双线性 + antialias 缩放到 896 + ImageNet 归一化
         → 每个成员各一次 fp16 前向（study 内多图合批）
         → study 内 logit 平均 → 成员间 logit 平均 → sigmoid
         → 患者平滑 → 写 CSV
```

在 8 核 Xeon 上，CPU 解码 + `cv2.resize` 合计约 60 ms/图；
把解码和缩放全搬到 GPU 后降到约 35 ms/图，直接决定了方案能否落进 100 ms。

---

## 实测结果

### 各组件贡献（验证集 9 类宏平均 AUC）

| 改动                      |              val |    增量 |
| ------------------------- | ---------------: | ------: |
| T@768 单模型              |           0.8086 |      — |
| + 患者平滑 λ=0.15        |           0.8111 | +0.0025 |
| + 896 分辨率 + 双模型集成 |           0.8134 | +0.0023 |
| + 训练 12 → 24 轮        |           0.8139 | +0.0005 |
| + 平滑（最终）            | **0.8167** |      — |

### 延迟分解（T4 级硬件，官方 100-study 口径）

| 阶段               |      2×896 模型 |
| ------------------ | ---------------: |
| GPU 解码（nvJPEG） |           ~35 ms |
| GPU 缩放 + 归一化  |            ~2 ms |
| 两次 fp16 前向     |           ~25 ms |
| 患者平滑 + 写出    |         ~0.01 ms |
| **合计**     | **~62 ms** |

---

## 被证伪的方向

以下方向都做过受控实验（单变量），**均无增益或负增益**，数据见 `docs/技术报告.md`：

| 方向                                                      | 结果                                             |
| --------------------------------------------------------- | ------------------------------------------------ |
| 医学预训练权重（RAD-DINO / BiomedCLIP / torchxrayvision） | val 0.72–0.77，显著更差                         |
| 更大骨干 ConvNeXt-Base@896（88M）                         | 0.8020 < S@896 的 0.8120                         |
| 1024 分辨率                                               | 单模 0.8116 / 混合集成 0.8131，均低于 896 双模型 |
| DINOv3 预训练（16.89 亿张 SSL）                           | 0.8113，集成更差                                 |
| 更强的教师（4 代迭代蒸馏）                                | 教师变强，学生反而降 0.002                       |
| TTA（3 尺度 × 翻转 = 6 路）                              | 0.8152 < 无 TTA 的 0.8167                        |
| 每类独立 λ 的患者平滑                                    | 分半交叉验证仅 +0.0001（过拟合）                 |
| 学习式患者聚合（小 MLP 学 λ）                            | 上界 0.8129 < 固定 λ 的 0.8138                  |
| 3/4/5 模型同分辨率集成                                    | 均 ≤ 0.8133（2 模型已饱和）                     |
| ImageNet-22k 预训练                                       | 0.8008 < in12k 的 0.8095                         |
| 30 轮长训练 / Mixup / 过采样 / model soup / EMA           | 均无增益                                         |

**结论**：在 30k 图规模上，**分辨率 + 集成 + 患者级后处理**是三个真实有效的杠杆，
而「更强预训练 / 更大模型 / 更复杂集成」都会被数据量或延迟预算吃掉。

---

## 引用与许可

* 代码：**MIT**（见 `LICENSE`）
* 预训练权重：timm 的 `convnext_small.in12k_ft_in1k` / `convnext_tiny.in12k_ft_in1k`，
  **Apache-2.0**（许可证全文与 model card 见 `licenses/`）
* 上游架构：`facebookresearch/ConvNeXt`（Apache-2.0）
* 本项目**不包含**任何医学数据集或由其派生的权重

---

## 致谢

训练框架的 early 版本参考了多个开源项目的数据管线设计；
`timm`、`torchvision`、`OpenCV` 提供了本项目依赖的全部基础设施。
