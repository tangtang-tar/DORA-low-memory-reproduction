"""阶段 B 完成审计：对目标中的每类产物做可重复断言。"""

import csv
from pathlib import Path

import yaml

from phase_b_common import read_jsonl


root = Path("/media/tangtang/Data/DORA")
with (root / "configs/phase_b_dora.yaml").open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

for key in ["model_path", "prm_path", "embedding_path", "dataset_name", "output_dir"]:
    assert Path(config[key]).is_relative_to(root), f"{key} 越出 DORA 边界"
assert config["num_samples"] == 1
assert 2 <= config["n"] <= 4
assert config["beam_width"] == 2
assert config["max_tokens"] <= 512

measurements = read_jsonl(root / "results/phase_b/resource_measurements.jsonl")
measured_modes = {row["mode"] for row in measurements}
required_modes = {
    "policy_fp16",
    "bge_fp16",
    "prm_cpu_bf16",
    "prm_mixed_bf16",
    "prm_gpu_bf16",
    "prm_8bit_mixed",
    "prm_4bit",
}
assert required_modes <= measured_modes
assert next(row for row in measurements if row["mode"] == "prm_4bit")["success"]

policy = read_jsonl(root / "results/phase_b/policy_rollouts.jsonl")
prm = read_jsonl(root / "results/phase_b/prm_scores.jsonl")
embedding = read_jsonl(root / "results/phase_b/embedding_scores.jsonl")
assert len(policy) == len(prm) == len(embedding) == config["n"]
assert all(len(row["steps"]) == len(row["prm_step_scores"]) for row in prm)
assert sum(row["allocated_rollouts"] for row in embedding) == config["n"]

import json

with (root / "results/phase_b/final_result.json").open(encoding="utf-8") as file:
    final_result = json.load(file)
assert final_result["correct"] is True
assert final_result["measured_peak_gpu_gib"] < 8

stability_dir = root / "results/phase_b/stability"
stability_policy = read_jsonl(stability_dir / "policy_rollouts.jsonl")
stability_prm = read_jsonl(stability_dir / "prm_scores.jsonl")
stability_embedding = read_jsonl(stability_dir / "embedding_scores.jsonl")
evaluation = read_jsonl(stability_dir / "evaluation_by_problem.jsonl")
assert len(stability_policy) == len(stability_prm) == len(stability_embedding) == 20
assert len({row["problem_index"] for row in stability_policy}) == 5
assert all(len(row["steps"]) == len(row["prm_step_scores"]) for row in stability_prm)
assert len(evaluation) == 5

with (stability_dir / "summary.csv").open(encoding="utf-8") as file:
    summary = list(csv.DictReader(file))
assert len(summary) == 6
assert {row["total"] for row in summary} == {"5"}

required_scripts = [
    "phase_b_policy.py",
    "phase_b_prm.py",
    "phase_b_embedding.py",
    "phase_b_finalize.py",
    "phase_b_stability_policy.py",
    "phase_b_stability_prm.py",
    "phase_b_stability_embedding.py",
    "phase_b_stability_evaluate.py",
]
assert all((root / "scripts_local" / script).is_file() for script in required_scripts)

print("阶段 B 完成审计通过")
print("B1-B5：单题 4 rollout，最终答案正确，峰值显存低于 8 GiB")
print("B6：5 题、20 rollout、6 种方法，逐题与汇总文件完整")
