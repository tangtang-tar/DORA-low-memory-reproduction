"""Run Phase D1 CPU simulation and write trial-level and grouped statistics."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from allocation_simulation import (
    CumulativeDeficitAllocator,
    largest_remainder,
    systematic_stochastic,
    synthetic_dora_weights,
    total_variation,
    validate_allocation,
)
from project_paths import ROOT, load_config


CONFIG = load_config("configs/phase_d1_cpu_simulation.yaml")
OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METHODS = ("largest_remainder", "systematic_stochastic", "cumulative_deficit")


def allocate_pair(method, quality, combined, budget, rng, states):
    if method == "largest_remainder":
        return largest_remainder(quality, budget), largest_remainder(combined, budget)
    if method == "systematic_stochastic":
        # Common random numbers isolate the effect of changing the probabilities.
        offset = float(rng.random())
        return (
            systematic_stochastic(quality, budget, offset),
            systematic_stochastic(combined, budget, offset),
        )
    return (
        states[method]["quality"].allocate(quality, budget),
        states[method]["combined"].allocate(combined, budget),
    )


def mean_confidence_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))


rng = np.random.default_rng(CONFIG["seed"])
trial_rows = []
for candidate_count in CONFIG["candidate_counts"]:
    for budget in CONFIG["budgets"]:
        for trial in range(CONFIG["trials"]):
            states = {
                "cumulative_deficit": {
                    "quality": CumulativeDeficitAllocator(candidate_count),
                    "combined": CumulativeDeficitAllocator(candidate_count),
                }
            }
            accumulators = {
                method: {
                    "signal_rounds": 0,
                    "erased_rounds": 0,
                    "continuous_tv": 0.0,
                    "instantaneous_tv_error": 0.0,
                    "coverage": 0.0,
                    "combined_actual": np.zeros(candidate_count, dtype=np.int64),
                    "combined_target": np.zeros(candidate_count, dtype=np.float64),
                }
                for method in METHODS
            }

            for _ in range(CONFIG["rounds"]):
                quality, combined = synthetic_dora_weights(
                    rng,
                    candidate_count,
                    CONFIG["quality_temperature"],
                    CONFIG["similarity_temperature"],
                )
                continuous_tv = total_variation(quality, combined)
                for method in METHODS:
                    quality_allocation, combined_allocation = allocate_pair(
                        method, quality, combined, budget, rng, states
                    )
                    validate_allocation(quality_allocation, budget, candidate_count)
                    validate_allocation(combined_allocation, budget, candidate_count)
                    item = accumulators[method]
                    if continuous_tv > CONFIG["continuous_signal_threshold"]:
                        item["signal_rounds"] += 1
                        item["erased_rounds"] += int(
                            np.array_equal(quality_allocation, combined_allocation)
                        )
                    item["continuous_tv"] += continuous_tv
                    item["instantaneous_tv_error"] += total_variation(
                        combined_allocation / budget, combined
                    )
                    item["coverage"] += float(np.count_nonzero(combined_allocation))
                    item["combined_actual"] += combined_allocation
                    item["combined_target"] += combined * budget

            for method, item in accumulators.items():
                rounds = CONFIG["rounds"]
                cumulative_total = budget * rounds
                trial_rows.append(
                    {
                        "candidate_count": candidate_count,
                        "budget": budget,
                        "trial": trial,
                        "method": method,
                        "signal_erasure_rate": item["erased_rounds"] / item["signal_rounds"],
                        "signal_survival_rate": 1.0 - item["erased_rounds"] / item["signal_rounds"],
                        "mean_continuous_signal_tv": item["continuous_tv"] / rounds,
                        "mean_instantaneous_allocation_tv_error": item["instantaneous_tv_error"] / rounds,
                        "mean_direction_coverage": item["coverage"] / rounds,
                        "final_cumulative_allocation_tv_error": total_variation(
                            item["combined_actual"] / cumulative_total,
                            item["combined_target"] / cumulative_total,
                        ),
                    }
                )

trial_path = OUTPUT_DIR / "trial_metrics.csv"
with trial_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=trial_rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(trial_rows)

grouped = defaultdict(lambda: defaultdict(list))
metric_names = [
    "signal_erasure_rate",
    "signal_survival_rate",
    "mean_continuous_signal_tv",
    "mean_instantaneous_allocation_tv_error",
    "mean_direction_coverage",
    "final_cumulative_allocation_tv_error",
]
for row in trial_rows:
    key = (row["candidate_count"], row["budget"], row["method"])
    for metric in metric_names:
        grouped[key][metric].append(row[metric])

summary_rows = []
for (candidate_count, budget, method), metrics in sorted(grouped.items()):
    row = {
        "candidate_count": candidate_count,
        "budget": budget,
        "method": method,
        "trials": CONFIG["trials"],
        "rounds_per_trial": CONFIG["rounds"],
    }
    for metric in metric_names:
        mean, ci95 = mean_confidence_interval(metrics[metric])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_ci95"] = ci95
    summary_rows.append(row)

summary_path = OUTPUT_DIR / "summary.csv"
with summary_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows)

public_config = dict(CONFIG)
public_config["output_dir"] = str(Path(CONFIG["output_dir"]).relative_to(ROOT))
metadata = {
    "config": public_config,
    "methods": list(METHODS),
    "trial_row_count": len(trial_rows),
    "summary_row_count": len(summary_rows),
    "definitions": {
        "signal_erasure_rate": "P(integer allocation unchanged | continuous DORA weights changed)",
        "instantaneous_allocation_tv_error": "TV(actual allocation / budget, continuous target)",
        "cumulative_allocation_tv_error": "TV(cumulative actual, cumulative continuous target)",
    },
}
with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

print(f"trial rows: {len(trial_rows)} -> {trial_path}")
print(f"summary rows: {len(summary_rows)} -> {summary_path}")
