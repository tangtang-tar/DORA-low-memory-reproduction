"""B5：汇总单题 DORA 文件链，提取最终答案并与标准答案比较。"""

import json

from sal.utils.math import find_answer_with_largest_sum, find_majority_answer
from sal.utils.qwen_math_parser import extract_answer, math_equal, strip_string

from phase_b_common import RESULT_DIR, read_jsonl


records = read_jsonl(RESULT_DIR / "embedding_scores.jsonl")
predictions = [extract_answer(record["text"], "math") for record in records]
weights = [record["combined_weight"] for record in records]

majority_answer = find_majority_answer(predictions)
weighted_answer = find_answer_with_largest_sum(predictions, weights)
reference_answer = records[0]["reference_answer"]

resource_rows = read_jsonl(RESULT_DIR / "resource_measurements.jsonl")
successful_gpu_peaks = [
    row["peak_gpu_allocated_gib"]
    for row in resource_rows
    if row.get("success")
    and row.get("mode") in {"policy_fp16", "bge_fp16", "prm_4bit"}
    and "peak_gpu_allocated_gib" in row
]

result = {
    "problem_id": records[0]["problem_id"],
    "problem": records[0]["problem"],
    "reference_answer": reference_answer,
    "rollout_answers": predictions,
    "prm_last_scores": [record["prm_last_score"] for record in records],
    "allocation": [record["allocated_rollouts"] for record in records],
    "allocation_budget": sum(record["allocated_rollouts"] for record in records),
    "early_stop": True,
    "early_stop_reason": "all initial rollouts produced complete answers",
    "majority_answer": majority_answer,
    "weighted_answer": weighted_answer,
    "final_answer": weighted_answer,
    "correct": bool(
        math_equal(strip_string(weighted_answer), strip_string(reference_answer))
    ),
    "total_model_seconds": (
        records[0]["policy_seconds_total"]
        + records[0]["prm_seconds_total"]
        + records[0]["embedding_seconds_total"]
    ),
    "measured_peak_gpu_gib": max(successful_gpu_peaks),
}

output_path = RESULT_DIR / "final_result.json"
with output_path.open("w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"结果：{output_path}")
