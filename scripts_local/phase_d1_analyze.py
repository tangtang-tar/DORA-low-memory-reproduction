"""Create paired method comparisons and machine-readable Phase D1 findings."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from project_paths import load_config


CONFIG = load_config("configs/phase_d1_cpu_simulation.yaml")
OUTPUT_DIR = Path(CONFIG["output_dir"])
with (OUTPUT_DIR / "trial_metrics.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

indexed = {
    (int(row["candidate_count"]), int(row["budget"]), int(row["trial"]), row["method"]): row
    for row in rows
}


def paired_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    ci95 = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, ci95


comparison_rows = []
for candidate_count in CONFIG["candidate_counts"]:
    for budget in CONFIG["budgets"]:
        for baseline in ("largest_remainder", "systematic_stochastic"):
            for metric in (
                "signal_erasure_rate",
                "final_cumulative_allocation_tv_error",
                "mean_instantaneous_allocation_tv_error",
            ):
                differences = []
                for trial in range(CONFIG["trials"]):
                    base = indexed[(candidate_count, budget, trial, baseline)]
                    cumulative = indexed[(candidate_count, budget, trial, "cumulative_deficit")]
                    differences.append(float(base[metric]) - float(cumulative[metric]))
                mean, ci95 = paired_interval(differences)
                comparison_rows.append(
                    {
                        "candidate_count": candidate_count,
                        "budget": budget,
                        "baseline": baseline,
                        "metric": metric,
                        "baseline_minus_cumulative_mean": mean,
                        "ci95": ci95,
                        "ci95_low": mean - ci95,
                        "ci95_high": mean + ci95,
                    }
                )

with (OUTPUT_DIR / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=comparison_rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(comparison_rows)

summary = defaultdict(dict)
with (OUTPUT_DIR / "summary.csv").open(encoding="utf-8") as file:
    for row in csv.DictReader(file):
        summary[(int(row["candidate_count"]), int(row["budget"]))][row["method"]] = row

minimum_budget = min(CONFIG["budgets"])
small_rows = [
    indexed[(candidate_count, minimum_budget, trial, method)]
    for candidate_count in CONFIG["candidate_counts"]
    for trial in range(CONFIG["trials"])
    for method in ("largest_remainder", "cumulative_deficit")
]
by_method = defaultdict(list)
for row in small_rows:
    by_method[row["method"]].append(row)


def average(method, metric):
    return float(np.mean([float(row[metric]) for row in by_method[method]]))


largest_erasure = average("largest_remainder", "signal_erasure_rate")
cumulative_erasure = average("cumulative_deficit", "signal_erasure_rate")
largest_long_error = average("largest_remainder", "final_cumulative_allocation_tv_error")
cumulative_long_error = average("cumulative_deficit", "final_cumulative_allocation_tv_error")
largest_instant_error = average("largest_remainder", "mean_instantaneous_allocation_tv_error")
cumulative_instant_error = average("cumulative_deficit", "mean_instantaneous_allocation_tv_error")

monotonic_checks = []
for candidate_count in CONFIG["candidate_counts"]:
    for method in ("largest_remainder", "systematic_stochastic", "cumulative_deficit"):
        values = [
            float(summary[(candidate_count, budget)][method]["signal_erasure_rate_mean"])
            for budget in CONFIG["budgets"]
        ]
        monotonic_checks.append(
            {
                "candidate_count": candidate_count,
                "method": method,
                "values_by_increasing_budget": values,
                "strictly_decreasing": all(left > right for left, right in zip(values, values[1:])),
            }
        )

findings = {
    "simulation_scope": {
        "candidate_counts": CONFIG["candidate_counts"],
        "budgets": CONFIG["budgets"],
        "trials": CONFIG["trials"],
        "rounds_per_trial": CONFIG["rounds"],
        "total_weight_scenarios": len(CONFIG["candidate_counts"]) * len(CONFIG["budgets"]) * CONFIG["trials"] * CONFIG["rounds"],
    },
    "minimum_budget_aggregate": {
        "budget": minimum_budget,
        "largest_remainder_erasure_rate": largest_erasure,
        "cumulative_deficit_erasure_rate": cumulative_erasure,
        "relative_erasure_reduction": 1.0 - cumulative_erasure / largest_erasure,
        "largest_remainder_cumulative_tv_error": largest_long_error,
        "cumulative_deficit_cumulative_tv_error": cumulative_long_error,
        "relative_cumulative_error_reduction": 1.0 - cumulative_long_error / largest_long_error,
        "largest_remainder_instantaneous_tv_error": largest_instant_error,
        "cumulative_deficit_instantaneous_tv_error": cumulative_instant_error,
        "relative_instantaneous_error_increase": cumulative_instant_error / largest_instant_error - 1.0,
    },
    "budget_trend_checks": monotonic_checks,
    "all_erasure_curves_strictly_decrease_with_budget": all(
        item["strictly_decreasing"] for item in monotonic_checks
    ),
    "interpretation_boundary": (
        "The simulation establishes a systematic discretization effect under the declared "
        "synthetic score/similarity generator. It does not by itself establish an accuracy gain "
        "on MATH-500 or universal behavior for every PRM and embedding distribution."
    ),
}
with (OUTPUT_DIR / "key_findings.json").open("w", encoding="utf-8") as file:
    json.dump(findings, file, ensure_ascii=False, indent=2)

print(json.dumps(findings["minimum_budget_aggregate"], ensure_ascii=False, indent=2))
print("all erasure curves strictly decrease with budget:", findings["all_erasure_curves_strictly_decrease_with_budget"])
