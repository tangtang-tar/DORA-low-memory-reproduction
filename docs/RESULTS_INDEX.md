# 结果证据索引

本文档把阶段报告中的结论映射到原始产物，便于复查。

公开仓库只提交不含题目和生成文本的 `results_summary/`。下表中的 `results/` 是本地完整实验运行后生成的原始证据，已通过 `.gitignore` 排除，避免公开模型输出、运行环境信息和大体积文件。

| 公开汇总 | 文件 |
|---|---|
| 资源、答案抽取、量化一致率和分配 | `results_summary/metrics.json` |
| 5题方法对照 | `results_summary/small_scale_methods.csv` |
| 小预算整数分配CPU模拟 | `results_summary/phase_d1/` |

| 结论 | 原始文件 |
|---|---|
| 本地环境、模型和数据准备完成 | `notes/PHASE_A.md` |
| 各模型显存、内存和速度 | `results/phase_b/resource_measurements.jsonl` |
| 单题最终答案、分配与峰值显存 | `results/phase_b/final_result.json` |
| 5题各方法汇总 | `results/phase_b/stability/summary.csv` |
| 5题逐题结果 | `results/phase_b/stability/evaluation_by_problem.jsonl` |
| 40条答案抽取汇总 | `results/phase_c/c1/summary.json` |
| 40条人工标注 | `results/phase_c/c1/manual_annotations.jsonl` |
| 40条自动抽取明细 | `results/phase_c/c1/extraction_results.jsonl` |
| C2 每轮部分路径 | `results/phase_c/c2/round_*_policy.jsonl` |
| C2 每轮 PRM 分数 | `results/phase_c/c2/round_*_prm.jsonl` |
| C2 每轮分配 | `results/phase_c/c2/round_*_allocation.jsonl` |
| C3 4-bit 分数 | `results/phase_c/c3/scores_4bit.jsonl` |
| C3 混合 BF16 分数 | `results/phase_c/c3/scores_mixed_bf16.jsonl` |
| C3 相关性和一致率 | `results/phase_c/c3/comparison.json` |
| C3 逐题分配比较 | `results/phase_c/c3/per_problem.jsonl` |

## 自动验收

```bash
source $DORA_ROOT/activate_dora.sh
python $DORA_ROOT/scripts_local/verify_phase_b.py
python $DORA_ROOT/scripts_local/verify_phase_c2.py
```

阶段 B 验收检查资源测量、单题完整链、峰值显存、5题/20 rollout 和6种汇总方法。阶段 C2 验收检查三轮预算守恒、非均匀分配、父路径复制和真实续写。
