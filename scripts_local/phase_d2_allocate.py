"""Compute real BGE signals and apply one of three D2 integer allocators."""

import argparse
import json

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from allocation_simulation import CumulativeDeficitAllocator, largest_remainder, softmax
from phase_d2_common import CONFIG, method_dir, read_jsonl, write_jsonl
from phase_d2_replay import official_round, upstream_count_repair


parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=CONFIG["methods"])
parser.add_argument("--round", type=int, required=True)
args = parser.parse_args()
destination = method_dir(args.method) / f"round_{args.round}_allocation.jsonl"
if destination.exists():
    print(f"resume: {destination} already exists")
    raise SystemExit(0)
records = read_jsonl(method_dir(args.method) / f"round_{args.round}_prm.jsonl")

model = BGEM3FlagModel(CONFIG["embedding_path"], use_fp16=True)
vectors = model.encode(
    [row["current_text"] for row in records], batch_size=4, max_length=512, return_dense=True
)["dense_vecs"]
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

state_path = method_dir(args.method) / "cumulative_state.json"
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
grouped = {}
for index, record in enumerate(records):
    grouped.setdefault(record["problem_index"], []).append(index)

for problem_index, indices in grouped.items():
    scores = [records[index]["prm_last_score"] for index in indices]
    cosine = vectors[indices] @ vectors[indices].T
    quality = softmax(scores, CONFIG["quality_temperature"])
    diversity = np.diag(softmax(cosine, CONFIG["balance_alpha"], axis=1))
    weights = quality * diversity
    weights /= weights.sum()
    raw_official = None
    if args.method == "official_round":
        raw_official = official_round(weights, CONFIG["budget"])
        allocation = upstream_count_repair(raw_official, CONFIG["budget"])
    elif args.method == "largest_remainder":
        allocation = largest_remainder(weights, CONFIG["budget"])
    else:
        saved = state.get(str(problem_index), {})
        allocator = CumulativeDeficitAllocator(len(indices))
        if saved:
            allocator.target = np.asarray(saved["target"], dtype=np.float64)
            allocator.actual = np.asarray(saved["actual"], dtype=np.int64)
        allocation = allocator.allocate(weights, CONFIG["budget"])
        state[str(problem_index)] = {
            "target": allocator.target.tolist(), "actual": allocator.actual.tolist()
        }
    assert int(allocation.sum()) == CONFIG["budget"]
    for local_index, record_index in enumerate(indices):
        record = records[record_index]
        record.update({
            "cosine_similarity_row": cosine[local_index].tolist(),
            "quality_weight": float(quality[local_index]),
            "diversity_weight": float(diversity[local_index]),
            "combined_weight": float(weights[local_index]),
            "continuous_allocation": float(weights[local_index] * CONFIG["budget"]),
            "allocated_rollouts": int(allocation[local_index]),
            "official_raw_allocation": (
                int(raw_official[local_index]) if raw_official is not None else None
            ),
            "official_raw_budget_sum": int(raw_official.sum()) if raw_official is not None else None,
        })

write_jsonl(destination, records)
if args.method == "cumulative_deficit":
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(records)} allocations to {destination}")
