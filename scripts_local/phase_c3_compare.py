"""比较 4-bit 与混合 BF16 PRM，并测量它们是否产生相同资源分配。"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from project_paths import load_config


config = load_config("configs/phase_c3.yaml")
output_dir = Path(config["output_dir"])


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def integer_allocation(values, budget):
    """最大余数法：先取整，再把余量给小数部分最大的路径。"""
    scaled = np.asarray(values, dtype=float)
    scaled = scaled / scaled.sum() * budget
    result = np.floor(scaled).astype(int)
    order = np.argsort(-(scaled - result), kind="stable")
    result[order[: budget - result.sum()]] += 1
    return scaled.tolist(), result.tolist()


scores = {
    mode: read_jsonl(output_dir / f"scores_{mode}.jsonl")
    for mode in ("4bit", "mixed_bf16")
}
assert [r["candidate_id"] for r in scores["4bit"]] == [r["candidate_id"] for r in scores["mixed_bf16"]]
with Path(config["candidate_path"]).open(encoding="utf-8") as file:
    candidates = [json.loads(line) for line in file if line.strip()]

four = np.array([r["prm_last_score"] for r in scores["4bit"]])
mixed = np.array([r["prm_last_score"] for r in scores["mixed_bf16"]])
embedder = SentenceTransformer(
    config["embedding_path"], device="cuda", local_files_only=True,
    model_kwargs={"torch_dtype": torch.float16},
)

groups = defaultdict(list)
for index, candidate in enumerate(candidates):
    groups[candidate["problem_index"]].append(index)

details = []
strict_top1_matches = 0
top_set_overlaps = 0
allocation_matches = 0
pair_agree = 0
pair_total = 0
for problem_index, indices in sorted(groups.items()):
    texts = [candidates[index]["text"] for index in indices]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    similarities = embeddings @ embeddings.T
    diversity = 1.0 - (similarities.sum(axis=1) - 1.0) / (len(indices) - 1)

    mode_data = {}
    for mode, all_scores in (("4bit", four), ("mixed_bf16", mixed)):
        values = all_scores[indices]
        quality = np.exp((values - values.max()) / config["quality_temperature"])
        quality /= quality.sum()
        utility = quality + config["diversity_weight"] * np.maximum(diversity, 0)
        continuous, allocation = integer_allocation(utility, config["budget"])
        mode_data[mode] = {
            "scores": values.tolist(),
            "top1": int(np.argmax(values)),
            "top_set": np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-8)).tolist(),
            "continuous_allocation": continuous,
            "integer_allocation": allocation,
        }

    strict_top1_matches += mode_data["4bit"]["top1"] == mode_data["mixed_bf16"]["top1"]
    top_set_overlaps += bool(set(mode_data["4bit"]["top_set"]) & set(mode_data["mixed_bf16"]["top_set"]))
    allocation_matches += mode_data["4bit"]["integer_allocation"] == mode_data["mixed_bf16"]["integer_allocation"]
    for left, right in combinations(range(len(indices)), 2):
        sign_four = np.sign(mode_data["4bit"]["scores"][left] - mode_data["4bit"]["scores"][right])
        sign_mixed = np.sign(mode_data["mixed_bf16"]["scores"][left] - mode_data["mixed_bf16"]["scores"][right])
        pair_agree += sign_four == sign_mixed
        pair_total += 1
    details.append({"problem_index": problem_index, **mode_data})

group_count = len(groups)
summary = {
    "candidate_count": len(candidates),
    "problem_count": group_count,
    "pearson": float(pearsonr(four, mixed).statistic),
    "spearman": float(spearmanr(four, mixed).statistic),
    "within_problem_pairwise_ranking_agreement": pair_agree / pair_total,
    "strict_top1_consistency": strict_top1_matches / group_count,
    "tie_aware_top_set_overlap": top_set_overlaps / group_count,
    "exact_allocation_consistency": allocation_matches / group_count,
    "thresholds": {"spearman_gt_0.9": False, "top1_gt_0.9": False, "allocation_gt_0.8": False},
}
summary["thresholds"]["spearman_gt_0.9"] = summary["spearman"] > 0.9
summary["thresholds"]["top1_gt_0.9"] = summary["strict_top1_consistency"] > 0.9
summary["thresholds"]["allocation_gt_0.8"] = summary["exact_allocation_consistency"] > 0.8
summary["passed"] = all(summary["thresholds"].values())

with (output_dir / "per_problem.jsonl").open("w", encoding="utf-8") as file:
    for detail in details:
        file.write(json.dumps(detail, ensure_ascii=False) + "\n")
with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
