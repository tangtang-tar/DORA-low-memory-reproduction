# DORA 低显存复现：阶段性成果

本项目复现 NeurIPS 2025 工作 **Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling** 的核心搜索机制，并探索在单张 8 GB 消费级 GPU 上运行 DORA 的工程路线。

> 当前定位：已完成低显存工程复现、真实逐步搜索验证和量化保真检查；尚未完成 MATH-500 全量准确率复现，不能据此声称 DORA 优于基线。

## 已完成

- 在 RTX 4070 Laptop 8 GB 上分别运行 Qwen2.5-1.5B Policy、Qwen2.5-Math-PRM-7B 和 BGE-M3。
- 通过 JSONL 中间文件让三个模型分时执行，避免同时驻留显存。
- 使用 4-bit NF4 将 7B PRM 峰值显存降至 4.43 GiB。
- 实现三轮“生成一步 → PRM 打分 → BGE 多样性 → 资源重分配”。
- 在 40 条冻结候选上比较 4-bit PRM 与混合 BF16 PRM。
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

## 文档入口

- [阶段技术报告](docs/STAGE_REPORT.md)
- [一页展示摘要](docs/PRESENTATION_BRIEF.md)
- [结果证据索引](docs/RESULTS_INDEX.md)
- [阶段 A 环境报告](notes/PHASE_A.md)
- [阶段 B 最终报告](notes/PHASE_B_FINAL_REPORT.md)
- [阶段 C 验收报告](notes/PHASE_C_FINAL_REPORT.md)

## 复现与验收

激活环境：

```bash
source /media/tangtang/Data/DORA/activate_dora.sh
```

运行现有验收：

```bash
python /media/tangtang/Data/DORA/scripts_local/verify_phase_b.py
python /media/tangtang/Data/DORA/scripts_local/verify_phase_c2.py
```

正式任务应通过日志包装器启动：

```bash
/media/tangtang/Data/DORA/scripts_local/run_logged.sh phase_c2 \
  python /media/tangtang/Data/DORA/scripts_local/phase_c2_run.py
```

每次运行在 `logs/<时间>_<名称>/` 保存 `output.log`、`command.txt`、`environment.txt` 和 `status.txt`。模型、数据、环境、结果与日志不进入本地实验 Git 仓库。

## 仓库边界

- DORA 根仓库：管理 `configs/`、`scripts_local/`、`notes/` 和文档。
- `src/`：官方上游源码仓库，保留独立 Git 历史。
- 官方源码仅有一个必要修复：缺失的本地 `Qwen2RMConfig` 导入改为当前 Transformers 中等价的 `Qwen2Config`。

