"""C2 BGE 子进程：计算连续权重和整数预算，记录淘汰与复制。"""

import argparse

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from phase_c2_common import CONFIG, OUTPUT_DIR, read_jsonl, write_jsonl


def softmax(values, temperature, axis=-1):
    values = np.asarray(values, dtype=np.float64) / temperature
    exponentials = np.exp(values - values.max(axis=axis, keepdims=True))
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def exact_allocation(probabilities, budget):
    raw = probabilities * budget
    allocation = np.floor(raw).astype(int)
    remainder = budget - int(allocation.sum())
    order = np.argsort(-(raw - allocation))
    allocation[order[:remainder]] += 1
    return raw, allocation


parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True)
args = parser.parse_args()
records = read_jsonl(OUTPUT_DIR / f"round_{args.round}_prm.jsonl")

model = BGEM3FlagModel(CONFIG["embedding_path"], use_fp16=True)
vectors = model.encode(
    [record["current_text"] for record in records],
    batch_size=2,
    max_length=512,
    return_dense=True,
)["dense_vecs"]
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cosine = vectors @ vectors.T

quality = softmax(
    [record["prm_last_score"] for record in records],
    CONFIG["quality_temperature"],
)
row_probability = softmax(cosine, CONFIG["balance_alpha"], axis=1)
diversity = np.diag(row_probability)
combined = quality * diversity
combined = combined / combined.sum()
continuous, allocation = exact_allocation(combined, CONFIG["budget"])

for index, record in enumerate(records):
    record["cosine_similarity_row"] = cosine[index].tolist()
    record["quality_weight"] = float(quality[index])
    record["diversity_weight"] = float(diversity[index])
    record["combined_weight"] = float(combined[index])
    record["continuous_allocation"] = float(continuous[index])
    record["allocated_rollouts"] = int(allocation[index])
    record["was_eliminated"] = int(allocation[index]) == 0
    record["was_duplicated"] = int(allocation[index]) > 1

output_path = OUTPUT_DIR / f"round_{args.round}_allocation.jsonl"
write_jsonl(output_path, records)
print(f"round {args.round} allocation：{allocation.tolist()}，continuous={continuous.round(3).tolist()}")
