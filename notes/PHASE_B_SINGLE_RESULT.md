# 阶段 B：单题端到端结果

## 结果

- 题目：将直角坐标 `(0,3)` 转成极坐标。
- 标准答案：`(3, π/2)`。
- 四条 Policy 候选的官方提取结果：`0`、`2`、`(3,π/2)`、`(3,π/2)`。
- 4-bit PRM 最后一步分数：`0.7539`、`1.0`、`1.0`、`1.0`。
- DORA 下一轮预算分配：`[0,1,1,2]`。
- 最终加权答案：`(3,π/2)`，与标准答案等价。
- 三个模型分时运行的实测峰值显存：约 `4.43 GiB`。
- Policy、PRM、BGE 的模型推理时间合计：约 `6.86 s`，不含反复加载模型和脚本启动时间。

四条候选都已经形成完整回答，所以本题在初始 rollout 后早停。分配事件仍被记录，但没有人为生成无意义的第二轮。

## 从头复现文件链

先激活 DORA 环境：

```bash
source /media/tangtang/Data/DORA/activate_dora.sh
```

然后依次运行，不能并行：

```bash
python /media/tangtang/Data/DORA/scripts_local/phase_b_policy.py
python /media/tangtang/Data/DORA/scripts_local/phase_b_prm.py
python /media/tangtang/Data/DORA/scripts_local/phase_b_embedding.py
python /media/tangtang/Data/DORA/scripts_local/phase_b_finalize.py
```

依次运行的原因是 8 GB 显存不能让三个模型长期同时驻留。每个脚本结束后，操作系统会回收该进程占用的显存；中间 JSONL 让下一阶段接着工作。

## 中间产物

- `results/phase_b/policy_rollouts.jsonl`
- `results/phase_b/prm_scores.jsonl`
- `results/phase_b/embedding_scores.jsonl`
- `results/phase_b/final_result.json`

## 暴露出的工程问题

Policy 即使收到 boxed 格式要求，也不保证每次严格遵守。官方答案提取器会把没有 boxed 的 `π/2` 错提取为最后一个数字 `2`。这说明输出格式约束与答案解析是后续 DORA 2.0 可以改进的一处，而不是单纯增加 rollout 就能解决的问题。
