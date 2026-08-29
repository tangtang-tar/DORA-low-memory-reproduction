"""Replay integer allocators on saved *real* PRM/BGE allocation signals.

This is deliberately a trace replay: all methods see the same candidates produced by
the historical largest-remainder run. It measures immediate allocation differences,
not counterfactual downstream accuracy. The latter belongs to the D2 end-to-end run.
"""

import csv
import json
from pathlib import Path

import numpy as np

from allocation_simulation import CumulativeDeficitAllocator, largest_remainder
from project_paths import ROOT, load_config


CONFIG = load_config("configs/phase_d2.yaml")
OUTPUT_DIR = ROOT / CONFIG["replay_output_dir"]
PUBLIC_DIR = ROOT / CONFIG["public_output_dir"]


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def official_round(probabilities, budget):
    """Exact behavior in upstream dora.py: independent NumPy rounding."""
    return np.rint(np.asarray(probabilities, dtype=np.float64) * budget).astype(np.int64)


def upstream_count_repair(allocation, budget):
    """Reproduce upstream repeat-then-truncate repair and return effective counts."""
    expanded = [index for index, count in enumerate(allocation) for _ in range(int(count))]
    if not expanded:
        return np.zeros(len(allocation), dtype=np.int64)
    if len(expanded) != budget:
        expanded = (expanded * (budget // len(expanded) + 1))[:budget]
    return np.bincount(expanded, minlength=len(allocation)).astype(np.int64)


def direction_coverage(allocation, cosine, threshold):
    """Count connected components represented after thresholding real BGE cosine."""
    count = len(allocation)
    parent = list(range(count))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(count):
        for right in range(left + 1, count):
            if cosine[left, right] >= threshold:
                union(left, right)
    return len({find(index) for index, value in enumerate(allocation) if value > 0})


def main():
    budget = int(CONFIG["budget"])
    threshold = float(CONFIG["direction_similarity_threshold"])
    files = [ROOT / item for item in CONFIG["replay_inputs"]]
    cumulative = None
    detail = []

    for round_index, path in enumerate(files):
        records = read_jsonl(path)
        weights = np.asarray([row["combined_weight"] for row in records], dtype=np.float64)
        weights /= weights.sum()
        cosine = np.asarray([row["cosine_similarity_row"] for row in records], dtype=np.float64)
        if cumulative is None or cumulative.candidate_count != len(records):
            cumulative = CumulativeDeficitAllocator(len(records))

        raw_official = official_round(weights, budget)
        allocations = {
            "official_round_raw": raw_official,
            "official_round": upstream_count_repair(raw_official, budget),
            "largest_remainder": largest_remainder(weights, budget),
            # Slot-aligned diagnostic. End-to-end runs maintain method-specific histories.
            "cumulative_deficit": cumulative.allocate(weights, budget),
        }
        baseline = allocations["largest_remainder"]
        for method, allocation in allocations.items():
            detail.append(
                {
                    "trace": str(path.relative_to(ROOT)),
                    "round": round_index,
                    "method": method,
                    "candidate_count": len(records),
                    "budget": budget,
                    "raw_sum": int(allocation.sum()),
                    "budget_deviation": int(allocation.sum()) - budget,
                    "allocation": json.dumps(allocation.tolist()),
                    "differs_from_largest_remainder": bool(np.any(allocation != baseline)),
                    "l1_from_largest_remainder": int(np.abs(allocation - baseline).sum()),
                    "candidate_coverage": int(np.count_nonzero(allocation)),
                    "direction_coverage": direction_coverage(allocation, cosine, threshold),
                    "continuous_weights": json.dumps(weights.tolist()),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(detail[0])
    for directory in (OUTPUT_DIR, PUBLIC_DIR):
        with (directory / "replay_detail.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detail)

    compared = [row for row in detail if row["method"] != "largest_remainder"]
    summary = {
        "source": "real PRM scores and real BGE cosine similarities from Phase C2",
        "trace_rounds": len(files),
        "budget": budget,
        "official_raw_budget_violation_rounds": sum(
            row["budget_deviation"] != 0 for row in detail if row["method"] == "official_round_raw"
        ),
        "method_difference_rows": sum(row["differs_from_largest_remainder"] for row in compared),
        "comparison_rows": len(compared),
        "limitations": [
            "Only one problem and three real rounds existed before D2.",
            "Replay is counterfactual only at each allocation decision; downstream candidates remain from the historical run.",
            "Cumulative replay aligns debt by beam slot because candidate identities change; causal effects require method-specific end-to-end runs.",
        ],
    }
    for directory in (OUTPUT_DIR, PUBLIC_DIR):
        with (directory / "replay_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
