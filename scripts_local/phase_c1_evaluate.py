"""C1：将自动提取结果与 40 条人工标签比较，并分类错误来源。"""

import json
from pathlib import Path

from reliable_answer import answers_equivalent, extract_reliable_answer
from project_paths import ROOT


RESULT_DIR = ROOT / "results/phase_c/c1"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


candidates = read_jsonl(RESULT_DIR / "candidates.jsonl")
annotations = {
    row["candidate_id"]: row["manual_answer"]
    for row in read_jsonl(RESULT_DIR / "manual_annotations.jsonl")
}
assert len(candidates) == len(annotations) == 40

results = []
for candidate in candidates:
    extraction = extract_reliable_answer(candidate["text"], candidate["is_truncated"])
    manual_answer = annotations[candidate["candidate_id"]]
    extraction_match = answers_equivalent(extraction["answer"], manual_answer)
    manual_correct = (
        manual_answer is not None
        and answers_equivalent(manual_answer, candidate["reference_answer"])
    )

    if not extraction_match:
        failure_type = "parse_error"
    elif manual_answer is None:
        failure_type = "truncated_no_final" if candidate["is_truncated"] else "no_final_answer"
    elif manual_correct:
        failure_type = "correct"
    else:
        failure_type = "reasoning_error"

    results.append(
        {
            "candidate_id": candidate["candidate_id"],
            "is_truncated": candidate["is_truncated"],
            "manual_answer": manual_answer,
            "automatic_answer": extraction["answer"],
            "source": extraction["source"],
            "extraction_match": extraction_match,
            "manual_answer_correct": manual_correct,
            "failure_type": failure_type,
        }
    )

with (RESULT_DIR / "extraction_results.jsonl").open("w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

matched = sum(result["extraction_match"] for result in results)
counts = {}
for result in results:
    counts[result["failure_type"]] = counts.get(result["failure_type"], 0) + 1

summary = {
    "matched": matched,
    "total": len(results),
    "agreement": matched / len(results),
    "acceptance_threshold": 0.95,
    "passed": matched / len(results) >= 0.95,
    "failure_type_counts": counts,
}
with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("不一致项：")
for result in results:
    if not result["extraction_match"]:
        print(result)
