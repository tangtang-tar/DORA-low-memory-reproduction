"""Full D2 acceptance checks after all three 20-problem runs complete."""

import csv
import json

from phase_d2_common import CONFIG, OUTPUT_DIR, method_dir, read_jsonl
from project_paths import ROOT


for method in CONFIG["methods"]:
    for round_index in range(CONFIG["allocation_rounds"]):
        policy = read_jsonl(method_dir(method) / f"round_{round_index}_policy.jsonl")
        prm = read_jsonl(method_dir(method) / f"round_{round_index}_prm.jsonl")
        allocation = read_jsonl(method_dir(method) / f"round_{round_index}_allocation.jsonl")
        assert len(policy) == len(prm) == len(allocation) == 20 * CONFIG["budget"]
        for problem_index in CONFIG["problem_indices"]:
            group = [row for row in allocation if row["problem_index"] == problem_index]
            assert len(group) == CONFIG["budget"]
            assert sum(row["allocated_rollouts"] for row in group) == CONFIG["budget"]
    finals = read_jsonl(method_dir(method) / f"round_{CONFIG['allocation_rounds']}_policy.jsonl")
    assert len(finals) == 20 * CONFIG["budget"]

public = ROOT / CONFIG["public_output_dir"]
with (OUTPUT_DIR / "end_to_end_summary.csv").open(encoding="utf-8") as file:
    summary = list(csv.DictReader(file))
with (OUTPUT_DIR / "end_to_end_by_problem.csv").open(encoding="utf-8") as file:
    problems = list(csv.DictReader(file))
assert len(summary) == 3
assert len(problems) == 60
assert {row["method"] for row in summary} == set(CONFIG["methods"])
for filename in (
    "end_to_end_summary.csv", "end_to_end_by_problem.csv", "end_to_end_metadata.json",
    "paired_comparisons.csv",
):
    assert (OUTPUT_DIR / filename).read_bytes() == (public / filename).read_bytes()
json.loads((OUTPUT_DIR / "end_to_end_metadata.json").read_text(encoding="utf-8"))
print("D2 full 20-problem acceptance passed")
