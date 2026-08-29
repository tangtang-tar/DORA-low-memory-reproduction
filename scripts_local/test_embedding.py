"""阶段 A：验证 BGE-M3 能把文本编码为向量并计算余弦相似度。"""

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from project_paths import ROOT


MODEL_PATH = ROOT / "models/bge-m3"

# use_fp16=True 会降低显存占用；此测试与 Policy 测试分开运行。
model = BGEM3FlagModel(str(MODEL_PATH), use_fp16=True)
texts = [
    "The answer is five.",
    "The result equals 5.",
    "A triangle has three sides.",
]
vectors = model.encode(texts, return_dense=True)["dense_vecs"]

# BGE 输出已归一化，但这里显式归一化，让余弦相似度的含义更直观。
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
similarities = vectors @ vectors.T

np.set_printoptions(precision=3, suppress=True)
print("文本顺序：")
for index, text in enumerate(texts):
    print(f"{index}: {text}")
print("\n余弦相似度矩阵：")
print(similarities)
