"""B3-3：BGE 计算语义关系，并输出下一轮 rollout 分配。"""

import time

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from phase_b_common import RESULT_DIR, load_config, read_jsonl, write_jsonl


def softmax(values, temperature):
    scaled = np.asarray(values, dtype=np.float64) / temperature
    exponentials = np.exp(scaled - scaled.max())
    return exponentials / exponentials.sum()


def exact_integer_allocation(probabilities, budget):
    raw = np.asarray(probabilities) * budget
    allocation = np.floor(raw).astype(int)
    remainder = budget - int(allocation.sum())
    order = np.argsort(-(raw - allocation))
    allocation[order[:remainder]] += 1
    return allocation


config = load_config()
records = read_jsonl(RESULT_DIR / "prm_scores.jsonl")
model = BGEM3FlagModel(config["embedding_path"], use_fp16=True)
started = time.perf_counter()
vectors = model.encode(
    [record["text"] for record in records],
    batch_size=config["beam_width"],
    max_length=config["max_tokens"],
    return_dense=True,
)["dense_vecs"]
elapsed = time.perf_counter() - started
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cosine_matrix = vectors @ vectors.T

# 与官方 dora.py 一致：质量先做温度 softmax；多样性使用行 softmax 的对角值。
quality_weights = softmax([record["prm_last_score"] for record in records], 0.1)
row_scaled = cosine_matrix / config["balance_alpha"]
row_exp = np.exp(row_scaled - row_scaled.max(axis=1, keepdims=True))
diversity_weights = np.diag(row_exp / row_exp.sum(axis=1, keepdims=True))
combined = quality_weights * diversity_weights
combined = combined / combined.sum()
allocation = exact_integer_allocation(combined, config["n"])

for index, record in enumerate(records):
    record["cosine_similarity_row"] = cosine_matrix[index].tolist()
    record["quality_weight"] = float(quality_weights[index])
    record["diversity_weight"] = float(diversity_weights[index])
    record["combined_weight"] = float(combined[index])
    record["allocated_rollouts"] = int(allocation[index])
    record["children_per_expansion"] = config["beam_width"]
    record["embedding_seconds_total"] = elapsed

output_path = RESULT_DIR / "embedding_scores.jsonl"
write_jsonl(output_path, records)
print(f"rollout 总预算：{int(allocation.sum())}，分配：{allocation.tolist()}")
print(f"结果：{output_path}")
