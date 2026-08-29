"""Paired uncertainty analysis for the 20-problem D2 comparison."""

import csv
import itertools

import numpy as np
from scipy.stats import binomtest

from phase_d2_common import CONFIG, OUTPUT_DIR
from project_paths import ROOT


PUBLIC_DIR = ROOT / CONFIG["public_output_dir"]
with (OUTPUT_DIR / "end_to_end_by_problem.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
lookup = {(row["method"], int(row["problem_index"])): row for row in rows}
rng = np.random.default_rng(CONFIG["seed"])


def boolean(row, key):
    return row[key].lower() == "true"


def paired_interval(differences, samples=20000):
    differences = np.asarray(differences, dtype=np.float64)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


comparisons = []
for left, right in itertools.combinations(CONFIG["methods"], 2):
    for metric in (
        "pass_at_4", "majority_correct", "mean_candidate_coverage",
        "mean_direction_coverage", "mean_effective_directions",
    ):
        if metric in ("pass_at_4", "majority_correct"):
            left_values = np.asarray([boolean(lookup[(left, q)], metric) for q in CONFIG["problem_indices"]])
            right_values = np.asarray([boolean(lookup[(right, q)], metric) for q in CONFIG["problem_indices"]])
            left_only = int(np.sum(left_values & ~right_values))
            right_only = int(np.sum(~left_values & right_values))
            discordant = left_only + right_only
            p_value = float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0
            differences = right_values.astype(float) - left_values.astype(float)
        else:
            left_values = np.asarray([float(lookup[(left, q)][metric]) for q in CONFIG["problem_indices"]])
            right_values = np.asarray([float(lookup[(right, q)][metric]) for q in CONFIG["problem_indices"]])
            left_only = right_only = ""
            p_value = ""
            differences = right_values - left_values
        lower, upper = paired_interval(differences)
        comparisons.append({
            "left": left, "right": right, "metric": metric,
            "right_minus_left": float(np.mean(differences)),
            "paired_bootstrap_ci_low": lower, "paired_bootstrap_ci_high": upper,
            "left_only_successes": left_only, "right_only_successes": right_only,
            "mcnemar_exact_p": p_value,
        })

for directory in (OUTPUT_DIR, PUBLIC_DIR):
    with (directory / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
print("wrote", len(comparisons), "paired comparisons")
