"""B6-4：在同一组候选上比较 SC、均匀 PRM 与多种 DORA 分配。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sal.utils.math import find_answer_with_largest_sum, find_majority_answer
from sal.utils.qwen_math_parser import extract_answer, math_equal, strip_string
from phase_b_stability_common import load_config, read_jsonl, write_jsonl


def softmax(values, temperature):
    values = np.asarray(values, dtype=np.float64) / temperature
    exponentials = np.exp(values - values.max())
    return exponentials / exponentials.sum()


def exact_allocation(probabilities, budget):
    raw = np.asarray(probabilities) * budget
    allocation = np.floor(raw).astype(int)
    remainder = budget - int(allocation.sum())
    order = np.argsort(-(raw - allocation))
    allocation[order[:remainder]] += 1
    return allocation


def equivalent(prediction, reference):
    return bool(math_equal(strip_string(prediction), strip_string(reference)))


config = load_config()
output_dir = Path(config["output_dir"])
records = read_jsonl(output_dir / "embedding_scores.jsonl")
groups = defaultdict(list)
for record in records:
    groups[record["problem_index"]].append(record)

methods = [
    "self_consistency",
    "uniform_rollout_prm_vote",
    "dora_quality_only",
    *[f"dora_bge_alpha_{alpha}" for alpha in config["balance_alphas"]],
]
correct_counts = {method: 0 for method in methods}
evaluation_rows = []

for problem_index, group in sorted(groups.items()):
    predictions = [extract_answer(record["text"], "math") for record in group]
    quality_scores = np.asarray([record["prm_last_score"] for record in group])
    reference = group[0]["reference_answer"]
    cosine = np.asarray([record["cosine_similarity_row"] for record in group])

    answers = {}
    allocations = {}
    answers["self_consistency"] = find_majority_answer(predictions)
    answers["uniform_rollout_prm_vote"] = find_answer_with_largest_sum(
        predictions, quality_scores.tolist()
    )

    quality_weights = softmax(quality_scores, config["quality_temperature"])
    quality_allocation = exact_allocation(quality_weights, config["n"])
    allocations["dora_quality_only"] = quality_allocation.tolist()
    answers["dora_quality_only"] = find_answer_with_largest_sum(
        predictions, quality_allocation.tolist()
    )

    for alpha in config["balance_alphas"]:
        row_scaled = cosine / alpha
        row_exp = np.exp(row_scaled - row_scaled.max(axis=1, keepdims=True))
        diversity = np.diag(row_exp / row_exp.sum(axis=1, keepdims=True))
        combined = quality_weights * diversity
        combined = combined / combined.sum()
        allocation = exact_allocation(combined, config["n"])
        method = f"dora_bge_alpha_{alpha}"
        allocations[method] = allocation.tolist()
        answers[method] = find_answer_with_largest_sum(predictions, allocation.tolist())

    correctness = {method: equivalent(answer, reference) for method, answer in answers.items()}
    for method, is_correct in correctness.items():
        correct_counts[method] += int(is_correct)

    evaluation_rows.append(
        {
            "problem_index": problem_index,
            "problem_id": group[0]["problem_id"],
            "reference_answer": reference,
            "rollout_answers": predictions,
            "prm_last_scores": quality_scores.tolist(),
            "answers": answers,
            "correct": correctness,
            "allocations": allocations,
        }
    )

write_jsonl(output_dir / "evaluation_by_problem.jsonl", evaluation_rows)

summary_rows = []
for method in methods:
    summary_rows.append(
        {
            "method": method,
            "correct": correct_counts[method],
            "total": len(groups),
            "accuracy": correct_counts[method] / len(groups),
        }
    )

with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["method", "correct", "total", "accuracy"])
    writer.writeheader()
    writer.writerows(summary_rows)

parse_failures = sum(not extract_answer(record["text"], "math") for record in records)
truncated = sum(record["generation_tokens"] >= config["max_tokens"] for record in records)
with (output_dir / "summary.md").open("w", encoding="utf-8") as file:
    file.write("# B6 五题初步实验\n\n")
    file.write("| 方法 | 正确/总数 | 准确率 |\n|---|---:|---:|\n")
    for row in summary_rows:
        file.write(
            f"| {row['method']} | {row['correct']}/{row['total']} | {row['accuracy']:.1%} |\n"
        )
    file.write(f"\n候选总数：{len(records)}；空答案提取：{parse_failures}；达到 token 上限：{truncated}。\n")
    file.write("\n这是 5 题工程冒烟实验，样本太小，不能当作论文效果结论。\n")

print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
print(f"逐题结果：{output_dir / 'evaluation_by_problem.jsonl'}")
print(f"汇总：{output_dir / 'summary.md'}")
