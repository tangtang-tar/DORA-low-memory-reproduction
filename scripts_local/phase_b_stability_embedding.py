"""B6-3：按题计算 4 条候选之间的 BGE 余弦相似度。"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from phase_b_stability_common import load_config, read_jsonl, write_jsonl


config = load_config()
output_dir = Path(config["output_dir"])
records = read_jsonl(output_dir / "prm_scores.jsonl")
groups = defaultdict(list)
for record in records:
    groups[record["problem_index"]].append(record)

model = BGEM3FlagModel(config["embedding_path"], use_fp16=True)
started = time.perf_counter()
for group in groups.values():
    vectors = model.encode(
        [record["text"] for record in group],
        batch_size=config["beam_width"],
        max_length=config["max_tokens"],
        return_dense=True,
    )["dense_vecs"]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = vectors @ vectors.T
    for index, record in enumerate(group):
        record["cosine_similarity_row"] = cosine[index].tolist()

elapsed = time.perf_counter() - started
for record in records:
    record["embedding_seconds_all"] = elapsed

output_path = output_dir / "embedding_scores.jsonl"
write_jsonl(output_path, records)
print(f"编码 {len(groups)} 题：{output_path}")
print(f"BGE 总编码时间：{elapsed:.2f} 秒")
