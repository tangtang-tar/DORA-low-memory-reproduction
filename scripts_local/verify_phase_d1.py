"""Acceptance checks for the Phase D1 CPU allocation simulation."""

import csv
import filecmp
import json
from pathlib import Path

import numpy as np

from allocation_simulation import (
    CumulativeDeficitAllocator,
    largest_remainder,
    systematic_stochastic,
    validate_allocation,
)
from project_paths import load_config


config = load_config("configs/phase_d1_cpu_simulation.yaml")
output_dir = Path(config["output_dir"])

probabilities = np.array([0.26, 0.24, 0.26, 0.24])
changed = np.array([0.24, 0.26, 0.24, 0.26])
assert not np.allclose(probabilities, changed)
assert np.array_equal(largest_remainder(probabilities, 4), largest_remainder(changed, 4))

for budget in config["budgets"]:
    validate_allocation(largest_remainder(probabilities, budget), budget, 4)
    validate_allocation(systematic_stochastic(probabilities, budget, 0.37), budget, 4)
allocator = CumulativeDeficitAllocator(4)
for _ in range(20):
    validate_allocation(allocator.allocate(probabilities, 4), 4, 4)
assert np.abs(allocator.actual - probabilities * 80).sum() <= 4

# Systematic sampling is empirically unbiased over a deterministic offset grid.
draws = np.zeros(4, dtype=np.float64)
offsets = (np.arange(10_000) + 0.5) / 10_000
for offset in offsets:
    draws += systematic_stochastic(probabilities, 4, float(offset))
assert np.allclose(draws / (len(offsets) * 4), probabilities, atol=1e-4)

with (output_dir / "metadata.json").open(encoding="utf-8") as file:
    metadata = json.load(file)
expected_summary_rows = len(config["candidate_counts"]) * len(config["budgets"]) * 3
expected_trial_rows = expected_summary_rows * config["trials"]
assert metadata["summary_row_count"] == expected_summary_rows
assert metadata["trial_row_count"] == expected_trial_rows

with (output_dir / "summary.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
assert len(rows) == expected_summary_rows
assert {row["method"] for row in rows} == {
    "largest_remainder", "systematic_stochastic", "cumulative_deficit"
}
for row in rows:
    for metric in (
        "signal_erasure_rate_mean",
        "signal_survival_rate_mean",
        "mean_instantaneous_allocation_tv_error_mean",
        "final_cumulative_allocation_tv_error_mean",
    ):
        assert 0.0 <= float(row[metric]) <= 1.0

small_budget_rows = [row for row in rows if int(row["budget"]) == min(config["budgets"])]
assert max(float(row["signal_erasure_rate_mean"]) for row in small_budget_rows) > 0.05

with (output_dir / "key_findings.json").open(encoding="utf-8") as file:
    findings = json.load(file)
assert findings["simulation_scope"]["total_weight_scenarios"] == 120_000
assert findings["all_erasure_curves_strictly_decrease_with_budget"] is True
minimum = findings["minimum_budget_aggregate"]
assert minimum["relative_erasure_reduction"] > 0.5
assert minimum["relative_cumulative_error_reduction"] > 0.5
assert minimum["relative_instantaneous_error_increase"] > 0.0

with (output_dir / "paired_comparisons.csv").open(encoding="utf-8") as file:
    comparisons = list(csv.DictReader(file))
budget_four_erasure = [
    row for row in comparisons
    if int(row["budget"]) == 4
    and row["baseline"] == "largest_remainder"
    and row["metric"] == "signal_erasure_rate"
]
assert len(budget_four_erasure) == len(config["candidate_counts"])
assert all(float(row["ci95_low"]) > 0.0 for row in budget_four_erasure)

for plot in (
    "signal_erasure_by_budget.png",
    "cumulative_error_by_budget.png",
    "instantaneous_error_by_budget.png",
):
    path = output_dir / plot
    assert path.is_file() and path.stat().st_size > 10_000

public_dir = Path(config["output_dir"]).parents[0].parents[0] / "results_summary/phase_d1"
for artifact in (
    "summary.csv",
    "paired_comparisons.csv",
    "key_findings.json",
    "metadata.json",
    "signal_erasure_by_budget.png",
    "cumulative_error_by_budget.png",
    "instantaneous_error_by_budget.png",
):
    assert filecmp.cmp(output_dir / artifact, public_dir / artifact, shallow=False)

print("Phase D1 acceptance passed")
print(f"trial rows: {expected_trial_rows}; summary rows: {expected_summary_rows}")
print("three methods, monotonic budget trend, paired confidence intervals, trade-off, and plots verified")
