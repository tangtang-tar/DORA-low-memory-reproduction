"""Evaluate D2 direction coverage, truncation, pass@4 and majority accuracy."""

import csv
import json
from collections import Counter

import numpy as np

from phase_d2_common import CONFIG, OUTPUT_DIR, method_dir, read_jsonl
from project_paths import ROOT
from reliable_answer import answers_equivalent, extract_reliable_answer, normalized_text


PUBLIC_DIR = ROOT / CONFIG["public_output_dir"]


def complete_link_direction_count(allocation, cosine, threshold):
    selected = [index for index, count in enumerate(allocation) if count > 0]
    clusters = []
    for index in selected:
        for cluster in clusters:
            if all(cosine[index, member] >= threshold for member in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    return len(clusters)


def effective_direction_count(allocation, cosine):
    allocation = np.asarray(allocation, dtype=np.float64)
    probability = allocation / allocation.sum()
    concentration = float(probability @ cosine @ probability)
    return 1.0 / concentration


def majority_answer(answers):
    usable = [(normalized_text(answer), answer) for answer in answers if answer is not None]
    if not usable:
        return None
    counts = Counter(key for key, _ in usable)
    winner = sorted(counts, key=lambda key: (-counts[key], key))[0]
    return next(answer for key, answer in usable if key == winner)


def main():
    problem_rows = []
    method_rows = []
    for method in CONFIG["methods"]:
        allocation_records = []
        for round_index in range(CONFIG["allocation_rounds"]):
            allocation_records.extend(
                read_jsonl(method_dir(method) / f"round_{round_index}_allocation.jsonl")
            )
        finals = read_jsonl(
            method_dir(method) / f"round_{CONFIG['allocation_rounds']}_policy.jsonl"
        )
        allocations_by_problem = {}
        for record in allocation_records:
            allocations_by_problem.setdefault(record["problem_index"], []).append(record)
        finals_by_problem = {}
        for record in finals:
            finals_by_problem.setdefault(record["problem_index"], []).append(record)

        for problem_index in CONFIG["problem_indices"]:
            candidate_rows = finals_by_problem[problem_index]
            extracted = []
            correct = []
            truncated = []
            for candidate in candidate_rows:
                is_truncated = (
                    not candidate["reached_eos"]
                    and candidate["added_tokens"] >= CONFIG["final_max_tokens"]
                )
                answer = extract_reliable_answer(candidate["current_text"], is_truncated)["answer"]
                extracted.append(answer)
                correct.append(answers_equivalent(answer, candidate["reference_answer"]))
                truncated.append(is_truncated)

            direction_counts = []
            effective_counts = []
            candidate_coverages = []
            raw_violations = 0
            grouped_rounds = {}
            for record in allocations_by_problem[problem_index]:
                grouped_rounds.setdefault(record["round"], []).append(record)
            for records in grouped_rounds.values():
                allocation = [row["allocated_rollouts"] for row in records]
                cosine = np.asarray([row["cosine_similarity_row"] for row in records])
                candidate_coverages.append(sum(value > 0 for value in allocation))
                direction_counts.append(complete_link_direction_count(
                    allocation, cosine, CONFIG["direction_similarity_threshold"]
                ))
                effective_counts.append(effective_direction_count(allocation, cosine))
                if records[0].get("official_raw_budget_sum") not in (None, CONFIG["budget"]):
                    raw_violations += 1

            majority = majority_answer(extracted)
            problem_rows.append({
                "method": method,
                "problem_index": problem_index,
                "problem_id": candidate_rows[0]["problem_id"],
                "pass_at_4": any(correct),
                "correct_candidates": sum(correct),
                "majority_correct": answers_equivalent(majority, candidate_rows[0]["reference_answer"]),
                "parsed_candidates": sum(answer is not None for answer in extracted),
                "truncated_candidates": sum(truncated),
                "mean_candidate_coverage": float(np.mean(candidate_coverages)),
                "mean_direction_coverage": float(np.mean(direction_counts)),
                "mean_effective_directions": float(np.mean(effective_counts)),
                "official_raw_budget_violation_rounds": raw_violations,
            })

        own = [row for row in problem_rows if row["method"] == method]
        method_rows.append({
            "method": method,
            "problems": len(own),
            "pass_at_4_correct": sum(row["pass_at_4"] for row in own),
            "pass_at_4_accuracy": float(np.mean([row["pass_at_4"] for row in own])),
            "majority_correct": sum(row["majority_correct"] for row in own),
            "majority_accuracy": float(np.mean([row["majority_correct"] for row in own])),
            "candidate_accuracy": sum(row["correct_candidates"] for row in own) / (len(own) * CONFIG["budget"]),
            "truncation_rate": sum(row["truncated_candidates"] for row in own) / (len(own) * CONFIG["budget"]),
            "mean_candidate_coverage": float(np.mean([row["mean_candidate_coverage"] for row in own])),
            "mean_direction_coverage": float(np.mean([row["mean_direction_coverage"] for row in own])),
            "mean_effective_directions": float(np.mean([row["mean_effective_directions"] for row in own])),
            "official_raw_budget_violation_rounds": sum(
                row["official_raw_budget_violation_rounds"] for row in own
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (OUTPUT_DIR, PUBLIC_DIR):
        for filename, rows in (("end_to_end_by_problem.csv", problem_rows), ("end_to_end_summary.csv", method_rows)):
            with (directory / filename).open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        metadata = {
            "problem_indices": CONFIG["problem_indices"],
            "budget": CONFIG["budget"],
            "allocation_rounds": CONFIG["allocation_rounds"],
            "direction_metric": (
                "complete-link greedy clusters at cosine >= "
                f"{CONFIG['direction_similarity_threshold']}; effective count = 1/(p^T cosine p)"
            ),
            "accuracy_metrics": ["pass_at_4", "normalized-answer majority", "candidate accuracy"],
            "cumulative_limitation": "Fractional debt is aligned by deterministic beam slot across rounds.",
        }
        (directory / "end_to_end_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(method_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
