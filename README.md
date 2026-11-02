# DCVC-UF 高光谱压缩迁移验证

本仓库记录将 **DCVC-UF** 的 8 帧 chunk coding 结构迁移到高光谱图像压缩的基础验证实验，包括：

- HySpecNet-11k 官方 hard split 数据准备；
- 高光谱 band 到视频帧输入轴的映射；
- 原始视频权重零样本真实码流测试；
- HySpecNet hard-train 三卡迁移训练；
- 实际熵编码后的 bpppc 与 PSNR 评测；
- 组会汇报和完整实验记录。

> 本仓库是验证性实验，不是 DCVC-UF 官方实现。官方代码与预训练权重请从
> [microsoft/DCVC](https://github.com/microsoft/DCVC) 获取。

## 核心结论

DCVC-UF 的 8-band chunk 迁移在工程上可行，真实编码和解码流程可以完整运行。

但在本实验条件下，20 epoch 迁移训练没有改善率失真性能：

| QP | 原始视频权重 bpppc | 原始权重 PSNR | 迁移模型 bpppc | 迁移模型 PSNR |
|---:|---:|---:|---:|---:|
| 10 | 0.001297 | 29.44 dB | 0.000806 | 27.10 dB |
| 31 | 0.004734 | 33.55 dB | 0.001602 | 29.47 dB |
| 52 | 0.016775 | 37.86 dB | 0.008100 | 33.82 dB |
| 63 | 0.030839 | 40.11 dB | 0.019741 | 35.84 dB |

迁移模型降低了 bpppc，但同时降低了 PSNR。当前结果支持以下判断：

1. 高光谱 band 可以替换视频时间帧，结构兼容；
2. 原始视频权重已经能对高光谱 band 序列进行有效有损编码；
3. 当前全参数微调方案造成明显的率失真性能下降；
4. 仅比较 bpppc 不足以评价压缩方法，需要同时比较 PSNR 或使用 BD-rate。

## 数据和指标

### HySpecNet-11k

- 官方 hard split；
- 训练：512 patches；
- 验证：64 patches；
- 测试：全部 91 patches；
- 原始 patch：`128 x 128 x 224`；
- 删除 22 个水汽吸收谱段后：`128 x 128 x 202`；
- 正式测试使用前 200 个有效谱段；
- `200 = 25 x 8`，对应 25 个完整 8-band chunk。

归一化方式：

```text
clip(X, 0, 10000) / 10000
```

### 真实 bpppc

高光谱中一个 band 对应一个 channel：

```text
bpppc = total bitstream bits / (H * W * C)
      = total bitstream bits / (128 * 128 * 200)
```

码流包含 I-frame、P-chunk、latent、熵模型参数和头信息，不包含共享模型权重。

注意：

```text
总 bpp = bpppc * 200
```

因此 `0.004734 bpppc` 表示整个 200-band 像元平均使用约 `0.9468 bit`。

## 输入映射

每个高光谱 band 保存为灰度 PNG，通过官方 YCbCr 输入路径转换为：

```text
Y = band - 0.5
U = 0
V = 0
```

8 个相邻 band 拼接为：

```text
[B, 24, 128, 128]
```

对应原视频模型的 8 帧、每帧三通道输入。

## 实验配置

| 项目 | 配置 |
|---|---|
| GPU | 4 x RTX 3090 24 GB |
| 训练 GPU | 3 张 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| 模型 | DCVC-UF HT-S |
| 参数量 | 81,157,632 |
| Patch | 128 x 128 |
| Chunk | 8 bands |
| 训练 epochs | 20 |
| 全局 batch | 12 |
| 初始学习率 | 1e-4 |
| 初始化 | 官方视频预训练权重 |
| 测试 QP | 10, 31, 52, 63 |
| skip threshold | 0.15 |

## 仓库结构

```text
.
├── README.md
├── configs/
│   └── hyspecnet_hts_experiment.json
├── docs/
│   ├── group-meeting-report.md
│   └── bpppc-experiment-report.md
├── results/
│   ├── bpppc_baseline_91.json
│   ├── bpppc_baseline_q63_91.json
│   └── bpppc_final_91.json
└── scripts/
    ├── cache_hyspecnet.py
    ├── hyspecnet_video_dataset.py
    ├── prepare_hyspecnet_bppc.py
    ├── prepare_legacy_hsi_dataset.py
    ├── smoke_hts.py
    ├── summarize_bppc.py
    ├── train_hyspecnet.py
    ├── train_hyspecnet_stage0.sh
    └── train_legacy_hsi_stage0.sh
```

`legacy` 脚本记录早期 256 x 256 California/Arizona 数据的工程验证流程。
正式指标以 HySpecNet hard split 与 `results/` 中的 JSON 为准。

## 复现步骤

### 1. 获取官方代码和权重

```bash
git clone https://github.com/microsoft/DCVC.git
```

按照官方文档安装 Python、PyTorch、CUTLASS、rANS 熵编码扩展和 CUDA
推理扩展，并下载官方 checkpoint：

```text
cvpr2026_image.pth.tar
cvpr2026_video_hts.pth.tar
```

在部分服务器环境中，导入扩展前还需要加入 Conda 的 `lib` 目录：

```bash
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH
```

### 2. 下载 HySpecNet-11k

从官方项目页面获取数据：

<https://hyspecnet.rsim.berlin>

将本仓库 `scripts/` 加入 `PYTHONPATH`，并修改脚本中的服务器数据路径。

### 3. 缓存 hard split

```bash
python scripts/cache_hyspecnet.py \
  --output-root /path/to/DCVC-UF-HSI/data/hyspecnet
```

默认缓存：

- 512 个 train patches；
- 64 个 val patches；
- 91 个 test patches。

### 4. 三卡训练

修改 `scripts/train_hyspecnet_stage0.sh` 中的路径，然后执行：

```bash
EPOCHS=20 bash scripts/train_hyspecnet_stage0.sh
```

实际训练通过官方 `train_video.py` 的多进程 DDP 启动。

### 5. 生成真实码流测试序列

```bash
python scripts/prepare_hyspecnet_bppc.py \
  --dataset-root /path/to/DCVC-UF-HSI/data/hyspecnet \
  --output-root /path/to/DCVC-UF-HSI/data/hyspecnet_bppc \
  --sequences 91 \
  --bands 200
```

### 6. 运行真实 bpppc 测试

参考 `docs/group-meeting-report.md` 中的完整 `test_video.py` 命令。

结果汇总：

```bash
python scripts/summarize_bppc.py \
  /path/to/bpppc_final_91.json
```

## 为什么 bpppc 看起来很低

该指标按光谱通道归一化。一个 200-band patch 的总码率应乘以 200 后再与常规
bpp 比较。

例如：

```text
0.004734 bpppc x 200 = 0.9468 bit/pixel
```

这不是无损压缩，也不能与 LineRWKV 约 `5.6 bpppc` 的无损结果直接比较。

## 当前方案的问题

- 只使用 512 个训练 patch；
- 只训练 20 epoch，而官方 stage0 为 45 epoch；
- 直接使用从零训练设计的 `1e-4` 学习率；
- 全参数训练可能破坏原视频模型的空间先验和熵模型；
- 训练使用连续浮点缓存，测试使用 8-bit PNG；
- 训练改变了 64 个 QP 对应的量化参数；
- 尚未计算 SAM、ERGAS、MRAE 和 BD-rate。

## 下一步

- 冻结大部分主干；
- 学习率降至 `1e-5` 到 `2e-5`；
- 仅训练输入输出适配器、熵模型末端和最后编码阶段；
- 训练与测试统一使用 8-bit 输入；
- 固定目标 QP 或重新校准 QP 映射；
- 增加光谱保真度指标；
- 增加随机初始化对照实验。

## 参考

- [DCVC-UF 官方代码](https://github.com/microsoft/DCVC)
- [Ultra-Fast Neural Video Compression, CVPR 2026](https://arxiv.org/abs/2606.04410)
- [HySpecNet-11k 数据集](https://hyspecnet.rsim.berlin)
- [HySpecNet-11k 论文](https://arxiv.org/abs/2306.00385)

