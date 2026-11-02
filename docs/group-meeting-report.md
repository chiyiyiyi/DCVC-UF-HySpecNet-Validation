# DCVC-UF 高光谱压缩迁移验证实验

**汇报日期：** 2026-11-02  
**实验代码：** <https://github.com/microsoft/DCVC>  
**参考论文：** *Ultra-Fast Neural Video Compression*, CVPR 2026  
**数据集：** HySpecNet-11k 官方 hard split

---

## 1. 实验原因

DCVC-UF 将一个视频 chunk 中的 8 个连续帧联合编码为单个 latent，再使用位置专用解码器并行重建。该结构不依赖运动估计，因此理论上可以把视频帧的时间轴替换为高光谱图像的光谱轴：

\[
X_v=\{x_0,\ldots,x_7\}
\quad\longrightarrow\quad
X_s=\{x_b,\ldots,x_{b+7}\}
\]

本实验需要回答以下问题：

1. 原始视频权重能否直接处理高光谱 band 序列；
2. 能否生成真实 bitstream，而不是只使用训练阶段的 bpp 估计；
3. 在 HySpecNet-11k 上训练后，模型能否在相同 bpppc 下获得更高 PSNR；
4. 当前存储空间和 3 张 RTX 3090 是否足以完成基础验证。

---

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | 4 × NVIDIA RTX 3090 24 GB |
| 训练 GPU | GPU 0/1/2，最多同时使用 3 张 |
| Python | 3.12 |
| PyTorch | 2.11.0 + cu128 |
| CUDA 驱动支持 | CUDA 12.6 |
| 模型 | DCVC-UF HT-S |
| 参数量 | 81,157,632 |
| 熵编码 | MLCodec rANS 扩展 |
| 推理扩展 | CUTLASS CUDA inference extension |

官方 checkpoint 与模型结构严格匹配：

```text
missing keys:    0
unexpected keys: 0
```

使用的官方权重：

```text
CVPR2026/cvpr2026_image.pth.tar
CVPR2026/cvpr2026_video_hts.pth.tar
```

---

## 3. 前置工程验证

### 3.1 单 chunk 前向和反向

使用合成高光谱数据验证网络接口，输入为：

```text
1 个参考 band + 8 个相邻 HSI band
```

模型在单张 RTX 3090 上完成前向和反向传播。

| 分辨率 | Batch | 峰值显存 | 结果 |
|---|---:|---:|---|
| 128 × 128 | 1 | 0.60 GB | 通过 |
| 256 × 256 | 1 | 1.35 GB | 通过 |
| 256 × 256 | 8 | 8.70 GB | 通过 |

因此 3 张 3090 使用全局 batch 12 不存在显存风险。

### 3.2 服务器已有高光谱数据

使用服务器现有的 California1/Arizona HSI patch，每个样本为：

```text
256 × 256 × 172
```

构建了：

- 16 个训练场景；
- 4 个验证场景；
- 每个场景 172 个灰度 band；
- 8 band 为一个 chunk。

使用官方 stage0 计划训练 HT-S：

- 3 张 GPU；
- 全局 batch 12；
- 45 epoch；
- 使用官方图像和视频权重初始化；
- 训练成功，耗时约 4–5 分钟。

该实验仅用于验证数据加载、DDP、checkpoint 保存和 stage0 训练链路，没有完成真实 bpppc 评测。

---

## 4. HySpecNet-11k 正式实验

### 4.1 数据配置

使用 HySpecNet-11k 官方 hard split：

| 数据划分 | Patch 数 |
|---|---:|
| 训练 | 512 |
| 验证 | 64 |
| 测试 | 91 |

每个 patch 原始尺寸：

```text
128 × 128 × 224
```

按照官方规则删除 22 个水汽吸收谱段后：

```text
128 × 128 × 202
```

反射率处理：

```text
clip(X, 0, 10000) / 10000
```

为保证 8-band chunk 完整，正式测试使用前 200 个有效谱段：

```text
200 = 25 × 8
```

### 4.2 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | DCVC-UF HT-S |
| Patch | 128 × 128 |
| Chunk size | 8 |
| 全局 batch | 12 |
| 每卡 batch | 4 |
| DataLoader workers | 每卡 3 |
| Epoch | 20 |
| 初始学习率 | `1e-4` |
| 序列长度 | 9 → 17 → 33 |
| 初始化 | 官方视频预训练权重 |
| 并行方式 | 3 卡 DDP |
| 训练耗时 | 约 33 分钟 |

训练结束后保存：

```text
runs/hyspecnet_hts_stage0/ckpt.pth.tar
```

---

## 5. 实验方法

### 5.1 高光谱 band 到伪 YUV

每个 HSI band 保存为灰度 PNG。DCVC-UF 读取时转换为：

```text
Y = band
U = 0.5
V = 0.5
```

网络内部减去 0.5 后得到：

```text
Y = band - 0.5
U = 0
V = 0
```

8 个相邻 band 被拼接为：

```text
[1, 24, 128, 128]
```

对应原视频模型的 8 帧、每帧三通道输入。

### 5.2 零样本推理

“零样本”表示：

- 图像码流使用官方 `cvpr2026_image.pth.tar`；
- 视频 chunk 使用官方 `cvpr2026_video_hts.pth.tar`；
- 不在 HySpecNet 上更新任何参数；
- 只做确定性的输入格式转换；
- 在 91 个 hard-test patch 上进行真实编码和解码。

零样本实验包含：

```text
1 个 I-frame + 25 个 P-chunk
```

### 5.3 迁移训练

迁移实验从同一组官方权重初始化，在 HySpecNet hard-train 的 512 个 patch 上训练 20 epoch，再在全部 91 个 hard-test patch 上评测。

### 5.4 真实 bpppc

使用 rANS 熵编码生成实际 `.bin` 码流：

\[
\mathrm{bpppc}
=
\frac{\text{实际码流总比特数}}
{H \times W \times C}
=
\frac{\text{总比特数}}
{128 \times 128 \times 200}
\]

码流包含：

- I-frame 的编码比特；
- P-chunk 的编码比特；
- latent 和熵模型参数流；
- SPS/NAL 头信息。

计费不包含共享模型权重。

高光谱中一个 band 对应一个 channel，因此该口径也等价于 bpppb。

---

## 6. 实验结果

### 6.1 原始视频权重零样本结果

测试集为全部 91 个 HySpecNet hard-test patch。

| QP | bpppc | PSNR |
|---:|---:|---:|
| 10 | 0.001297 | 29.44 dB |
| 31 | 0.004734 | 33.55 dB |
| 52 | 0.016775 | 37.86 dB |
| 63 | 0.030839 | 40.11 dB |

需要注意：

\[
0.004734 \times 200
= 0.9468\ \text{总 bit/pixel}
\]

因此 bpppc 数值很小，是因为分母包含了 200 个光谱通道。

### 6.2 迁移模型结果

| QP | 原始权重 bpppc | 原始权重 PSNR | 迁移模型 bpppc | 迁移模型 PSNR |
|---:|---:|---:|---:|---:|
| 10 | 0.001297 | 29.44 dB | 0.000806 | 27.10 dB |
| 31 | 0.004734 | 33.55 dB | 0.001602 | 29.47 dB |
| 52 | 0.016775 | 37.86 dB | 0.008100 | 33.82 dB |
| 63 | 0.030839 | 40.11 dB | 0.019741 | 35.84 dB |

迁移训练使各 QP 下的 bpppc 降低了约 36%–66%，但 PSNR 同时下降约 2.3–4.3 dB。

例如：

- 迁移模型达到约 33.82 dB 时需要 0.0081 bpppc；
- 原始视频权重在 0.0047 bpppc 时已经达到 33.55 dB。

因此，在本次实验范围内，原始视频权重的率失真性能优于迁移模型。

### 6.3 与 HySpecNet 基线的关系

HySpecNet-11k 原始学习式基线：

| 方法 | bpppc | PSNR |
|---|---:|---:|
| SSCNet | 2.53 | 43.64 dB |
| 3D-CAE | 2.02 | 39.54 dB |
| 1D-CAE | 8.08 | 54.85 dB |

近期有损方法在约 `0.03/0.08 bpppb` 的参考结果：

| 方法 | 约 0.03 bpppb | 约 0.08 bpppb |
|---|---:|---:|
| PCA + JPEG2000 | 33.33 dB | 35.28 dB |
| 3D-SPIHT | 30.20 dB | 33.74 dB |
| MBT2018 | 37.23 dB | 39.02 dB |
| 3D-CAE | 33.72 dB | 34.72 dB |
| HyCoT | 34.36 dB | 38.37 dB |
| QRes-VAE | 37.73 dB | 40.60 dB |
| HyCASS | 42.28 dB | 44.61 dB |

注意：这些文献主要使用 HySpecNet easy split，本实验使用 hard split，因此不能直接排名。

另外，LineRWKV 的约 5.6 bpppc 属于无损率结果，目标为近似精确重建，不能与当前有损结果直接比较。

---

## 7. 结论

1. **结构迁移可行。** 高光谱 band 可以按 8 个一组替换视频时间帧，完整完成前向、反向、真实熵编码和解码。
2. **零样本视频权重已经具有极低码率压缩能力。** 但 PSNR 仅处于中等水平，对高光谱谱形的保真度仍需进一步验证。
3. **当前全参数迁移训练没有改善率失真。** 虽然降低了 bpppc，但同时显著降低了 PSNR。
4. **bpppc 不能单独作为优劣判据。** 必须在相同 PSNR 下比较 bpppc，或在相同 bpppc 下比较 PSNR。
5. **NVC 与无损预测编码不能直接比较。** LineRWKV 的 5.6 bpppc 与 DCVC-UF 的 0.001–0.03 bpppc 对应完全不同的质量目标。

---

## 8. 当前问题分析

当前迁移训练效果较差，可能原因包括：

- 只使用了 512 个训练 patch；
- 仅训练 20 epoch，未达到官方 45 epoch stage0；
- 直接使用从零训练设计的 `1e-4` 学习率；
- 全参数训练造成空间先验和熵模型遗忘；
- 训练使用连续浮点数据，测试使用 8-bit PNG；
- 训练改变了 64 个 QP 对应的码率位置；
- 尚未加入 SAM、ERGAS、MRAE 等光谱约束。

---

## 9. 下一步计划

建议优先执行以下实验：

1. 冻结大部分视频主干；
2. 将学习率降至 `1e-5～2e-5`；
3. 只训练输入输出适配层、熵模型末端和最后一个编码阶段；
4. 训练与测试统一量化为 8-bit；
5. 固定目标 QP 或重新校准 QP 映射；
6. 增加 SAM、ERGAS、MRAE；
7. 构建真正的率失真曲线并计算 BD-rate/BD-PSNR；
8. 增加随机初始化对照实验，证明视频预训练是否真正带来收益。

---

## 10. 实验文件

| 内容 | 服务器路径 |
|---|---|
| 最终迁移 checkpoint | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/runs/hyspecnet_hts_stage0/ckpt.pth.tar` |
| 零样本测试结果 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/runs/bpppc_baseline_91.json` |
| 零样本 QP63 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/runs/bpppc_baseline_q63_91.json` |
| 迁移模型测试结果 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/runs/bpppc_final_91.json` |
| 训练日志 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/logs/hyspecnet_stage0_20.log` |
| 最终测试日志 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/logs/bpppc_final_91.log` |
| 完整报告 | `/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/results/bpppc_experiment_report.md` |

