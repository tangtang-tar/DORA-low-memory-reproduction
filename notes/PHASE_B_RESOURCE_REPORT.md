# 阶段 B：本机资源测量报告

测试设备：NVIDIA GeForce RTX 4070 Laptop GPU，物理显存 8 GB，PyTorch 可用容量约 7.62 GiB；系统 RAM 31 GiB。

所有模型均在独立进程中依次测量。推理时间只用于比较本机路线，不代表论文吞吐量。

| 模型与模式 | 结果 | 加载时间 | 单次推理 | 峰值显存 | 峰值 RAM |
|---|---:|---:|---:|---:|---:|
| Policy FP16 | 成功 | 0.93 s | 1.25 s / 64 tokens | 2.89 GiB | 3.64 GiB |
| BGE-M3 FP16 | 成功 | 0.70 s | 0.41 s / 2 texts | 1.07 GiB | 4.05 GiB |
| PRM CPU BF16 | 成功 | 0.44 s | 205.84 s / 2 steps | 0 GiB | 12.92 GiB |
| PRM CPU/GPU BF16 | 成功 | 1.19 s | 1.48 s / 2 steps | 6.01 GiB | 14.46 GiB |
| PRM 4-bit NF4 | 成功 | 2.83 s | 0.32 s / 2 steps | 4.43 GiB | 4.57 GiB |
| PRM GPU BF16 | 失败 | 第 3/4 分片前 OOM | 未运行 | 失败时已分配 7.32 GiB | — |
| PRM 8-bit mixed | 失败 | 可完成加载 | 前向动态量化时 OOM | 失败时已分配 7.24 GiB | — |

## 路线结论

阶段 B4 首选 PRM 4-bit NF4：它能完整留在 GPU，速度最快，并且仍留下约 3 GiB 物理显存余量。由于阶段 B3 会让 Policy、PRM、BGE 分时运行，不要求这些模型同时驻留显存。

CPU/GPU BF16 是保真度对照路线：它不量化权重，实际推理可行，但占用约 14.46 GiB RAM 和 6.01 GiB 显存。后续可以用它检查 4-bit 步骤分数排序是否发生明显变化。

纯 CPU 只作为保底路线。两步就需要约 206 秒，不适合 DORA 多轮搜索。

8-bit 在当前自定义 PRM 与 bitsandbytes 组合下不采用：即使限制静态 GPU 权重预算，前向时动态搬运/量化层仍会占满显存。继续为它写复杂 offload 代码的收益低于已经成功的 4-bit 路线。
