# DORA 低显存复现：阶段性成果

本项目复现 NeurIPS 2025 工作 **Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling** 的核心搜索机制，并探索在单张 8 GB 消费级 GPU 上运行 DORA 的工程路线。

> **项目状态：阶段性结项，进入维护状态。** 已完成低显存工程复现、真实逐步搜索、量化保真、小预算模拟和20题探索实验；不再继续小样本调参。尚未完成论文规模准确率复现，不能据此声称 DORA 或整数分配改动优于基线。

> **非官方声明：**这是独立的低显存复现项目，不代表原论文作者或官方 DORA 项目。

## 上游项目与论文

- 官方代码：[WangXinglin/DORA](https://github.com/WangXinglin/DORA)
- 论文：[Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling](https://arxiv.org/abs/2506.15707)
- 本地实验基于官方仓库 commit `150a0f73fa50ddc484f81cc8a23aebc40f546aa0`。

本仓库不复制官方 `src/`。请将官方仓库克隆到本项目的 `src/` 目录，再应用文末所述的兼容性修复。使用本项目或上游实现时，请引用原论文。

## 准备项目

```bash
git clone https://github.com/tangtang-tar/DORA-low-memory-reproduction.git
cd DORA-low-memory-reproduction
git clone https://github.com/WangXinglin/DORA.git src
git -C src checkout 150a0f73fa50ddc484f81cc8a23aebc40f546aa0
git -C src apply ../patches/modeling_qwen2_rm.patch
```

本地资源默认放置为：

```text
models/Qwen2.5-1.5B-Instruct
models/Qwen2.5-Math-PRM-7B
models/bge-m3
datasets/MATH-500
envs/dora
```

这些大体积目录均已被 `.gitignore` 排除。所有配置路径相对于仓库根解析，也可以在运行前设置 `DORA_ROOT` 指向其他项目根目录。

## 已完成

- 在 RTX 4070 Laptop 8 GB 上分别运行 Qwen2.5-1.5B Policy、Qwen2.5-Math-PRM-7B 和 BGE-M3。
- 通过 JSONL 中间文件让三个模型分时执行，避免同时驻留显存。
- 使用 4-bit NF4 将 7B PRM 峰值显存降至 4.43 GiB。
- 实现三轮“生成一步 → PRM 打分 → BGE 多样性 → 资源重分配”。
- 在 40 条冻结候选上比较 4-bit PRM 与混合 BF16 PRM。
- 完成12万轮CPU模拟，量化小预算整数化造成的多样性信号损失。
- 完成真实 PRM/BGE 轨迹重放与三种分配方法的 20 题端到端对照。
- 修复不可靠的最终答案抽取，区分推理错误、截断和解析失败。
- 建立统一的命令、环境、stdout/stderr 和退出状态记录。

## 核心结果

| 指标 | 结果 |
|---|---:|
| Policy FP16 峰值显存 | 2.89 GiB |
| BGE-M3 FP16 峰值显存 | 1.07 GiB |
| PRM 4-bit NF4 峰值显存 | 4.43 GiB |
| PRM 混合 BF16 峰值显存 | 6.01 GiB |
| 4-bit / BF16 Pearson | 0.9958 |
| 4-bit / BF16 Spearman | 0.9883 |
| PRM Top-1 一致率 | 100% |
| 整数资源分配一致率 | 100% |
| 答案抽取样本内一致率 | 40/40 |
| D2 官方独立取整原始预算违规 | 13/60 |
| D2 pass@4（官方 / 最大余数 / 累计） | 15/20 / 16/20 / 16/20 |
| D2 多数答案正确率（三种方法） | 13/20 |

公开的脱敏汇总见 [`results_summary/`](results_summary/)，其中不包含题目文本、模型输出、用户名、主机名或本机路径。

逐步 DORA 的三轮整数分配为：

```text
[1, 2, 0, 1]
[1, 1, 1, 1]
[2, 0, 1, 1]
```

其中第 0、2 轮为非均匀分配；验证脚本确认父路径复制、文本继承和后续续写真正发生。

## 重要局限

- 方法对照只有 5 道题，所有方法均为 2/5，不能形成方法优越性结论。
- 答案抽取规则是在当前 40 条候选上迭代形成，100% 是样本内验收结果。
- 1.5B Policy 的 40 条候选中有 12 条在 token 上限处截断。
- 当前没有复现论文的 MATH-500、AIME2024、AIME2025 完整结果。
- 小预算下连续多样性权重可能在整数舍入时消失。
- D2 的20题实验显示覆盖差异，但没有证明最终正确率提升；最终候选仍有约21%–24%截断。

## 文档入口

- [项目结项报告](docs/FINAL_REPORT.md)
- [阶段技术报告](docs/STAGE_REPORT.md)
- [一页展示摘要](docs/PRESENTATION_BRIEF.md)
- [结果证据索引](docs/RESULTS_INDEX.md)
- [阶段 A 环境报告](notes/PHASE_A.md)
- [阶段 B 最终报告](notes/PHASE_B_FINAL_REPORT.md)
- [阶段 C 验收报告](notes/PHASE_C_FINAL_REPORT.md)
- [阶段 D1 CPU 模拟报告](notes/PHASE_D1_CPU_SIMULATION_REPORT.md)
- [阶段 D2 真实重放与20题端到端报告](notes/PHASE_D2_REAL_REPLAY_AND_20_PROBLEM_REPORT.md)

## 复现与验收

激活环境：

```bash
source $DORA_ROOT/activate_dora.sh
```

运行现有验收：

```bash
python $DORA_ROOT/scripts_local/verify_phase_b.py
python $DORA_ROOT/scripts_local/verify_phase_c2.py
```

正式任务应通过日志包装器启动：

```bash
$DORA_ROOT/scripts_local/run_logged.sh phase_c2 \
  python $DORA_ROOT/scripts_local/phase_c2_run.py
```

每次运行在 `logs/<时间>_<名称>/` 保存 `output.log`、`command.txt`、`environment.txt` 和 `status.txt`。模型、数据、环境、结果与日志不进入本地实验 Git 仓库。

运行阶段 D1 CPU 模拟：

```bash
python $DORA_ROOT/scripts_local/phase_d1_cpu_simulation.py
python $DORA_ROOT/scripts_local/phase_d1_analyze.py
python $DORA_ROOT/scripts_local/phase_d1_plot.py
python $DORA_ROOT/scripts_local/verify_phase_d1.py
```

运行阶段 D2（约 50 分钟本地 GPU 时间）：

```bash
python $DORA_ROOT/scripts_local/phase_d2_replay.py
python $DORA_ROOT/scripts_local/verify_phase_d2_replay.py
python $DORA_ROOT/scripts_local/phase_d2_run.py
python $DORA_ROOT/scripts_local/phase_d2_evaluate.py
python $DORA_ROOT/scripts_local/phase_d2_analyze.py
python $DORA_ROOT/scripts_local/verify_phase_d2.py
```

## 仓库边界

- DORA 根仓库：管理 `configs/`、`scripts_local/`、`notes/` 和文档。
- `src/`：官方上游源码仓库，保留独立 Git 历史。
- 官方源码仅有一个必要修复：缺失的本地 `Qwen2RMConfig` 导入改为当前 Transformers 中等价的 `Qwen2Config`。

## 引用

```bibtex
@article{wang2025every,
  title={Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling},
  author={Wang, Xinglin and Li, Yiwei and Feng, Shaoxiong and Yuan, Peiwen and Zhang, Yueqi and Shi, Jiayi and Tan, Chuyi and Pan, Boyuan and Hu, Yao and Li, Kan},
  journal={arXiv preprint arXiv:2506.15707},
  year={2025}
}
```
