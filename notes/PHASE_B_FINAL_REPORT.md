# 阶段 B 最终报告

## 已完成的目标

阶段 B 已在 8 GB RTX 4070 Laptop GPU 上完成：本地配置、资源测量、三段分时流水线、4-bit PRM、单题端到端验证，以及 5 题小规模对照实验。

三个模型从不同时驻留显存。Policy、PRM、BGE 通过 JSONL 文件交接，因此任一阶段可以独立重跑。

## 低显存路线

| 模块 | 运行方式 | 实测峰值显存 |
|---|---|---:|
| Qwen2.5-1.5B Policy | FP16 GPU | 2.89 GiB |
| Qwen2.5-Math-PRM-7B | 4-bit NF4 GPU | 4.43 GiB |
| BGE-M3 | FP16 GPU | 1.07 GiB |

PRM 混合 BF16 也可运行，但占约 6.01 GiB 显存和 14.46 GiB RAM。纯 CPU 能运行但两步评分约需 206 秒。全 GPU BF16 和当前 8-bit offload 路线均实测 OOM。

## 单题端到端结果

- 题目：把 `(0,3)` 转成极坐标。
- 四条 rollout 的 PRM 最后一步分数：`[0.7539, 1.0, 1.0, 1.0]`。
- DORA 分配：`[0,1,1,2]`。
- 最终答案：`(3,π/2)`，判定正确。
- 实际流水线峰值显存：4.43 GiB。

## B6 五题初步实验

固定同一组 5 题 × 4 rollout，再切换选择与分配方法，避免不同采样造成不公平比较。

| 方法 | 正确/总数 | 准确率 |
|---|---:|---:|
| 普通 Self-Consistency 多数票 | 2/5 | 40% |
| 均匀 rollout + PRM 加权票 | 2/5 | 40% |
| DORA，仅质量分配，不使用 BGE | 2/5 | 40% |
| DORA + BGE，alpha=0.01 | 2/5 | 40% |
| DORA + BGE，alpha=0.1 | 2/5 | 40% |
| DORA + BGE，alpha=1.0 | 2/5 | 40% |

模型推理时间：Policy 约 45.15 秒，4-bit PRM 约 7.40 秒，BGE 约 0.86 秒。20 条候选中有 6 条达到 512-token 上限。

三种 BGE alpha 在这 5 题上的整数分配完全相同。原因不是 BGE 没有产生连续差异，而是：rollout 总预算只有 4，PRM 质量 softmax 已经较尖锐，BGE 的变化经过整数舍入后消失。

## 当前不能声称的结论

5 题样本太小，不能用 40% 与论文数字比较，也不能据此认定 DORA 无效。当前实验的价值是证明低显存实现可运行，并暴露工程瓶颈。

当前流水线只在初始完整候选上模拟下一轮预算分配。单题中所有候选已完成，因此按早停规则没有继续扩展；B6 也主要比较选择/分配结果，不等价于论文的大预算多轮指标复现。

## 最值得做成 DORA 2.0 的方向

1. **可靠答案协议与解析**：小 Policy 经常不输出 boxed 答案，官方提取器会把 `π/2` 错取成 `2`。先修复这一点，比继续调 BGE 更可能提高真实准确率。
2. **自适应分配温度**：当 PRM 分数接近 0/1 饱和时，固定 `quality_temperature=0.1` 会压倒多样性项。可根据候选分数熵自动调温度。
3. **避免小预算舍入损失**：预算只有 4 时，连续 BGE 权重常被整数化抹平。可以使用随机分配、最大剩余法的多轮累计版本，或延迟到跨轮统一结算。
4. **对部分推理而非完整答案做聚类**：DORA 的资源分配价值应发生在推理尚未结束时。下一阶段应保存每个 step 的状态，并在 step 边界执行 PRM+BGE 分配。
5. **量化保真检查**：4-bit 很快，但仍应在更多候选上与混合 BF16 的排序相关性做对照，确认量化没有改变资源分配顺序。

## 主要产物

- 单题配置：`configs/phase_b_dora.yaml`
- B6 配置：`configs/phase_b_stability.yaml`
- 资源报告：`notes/PHASE_B_RESOURCE_REPORT.md`
- 单题结果：`results/phase_b/final_result.json`
- 五题逐题结果：`results/phase_b/stability/evaluation_by_problem.jsonl`
- 五题汇总：`results/phase_b/stability/summary.csv`
