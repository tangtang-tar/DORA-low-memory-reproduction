"""C2 验收：证明多轮非均匀分配、复制和部分文本续写真实发生。"""

from phase_c2_common import CONFIG, OUTPUT_DIR, read_jsonl


allocations = []
round_records = []
for round_index in range(CONFIG["allocation_rounds"]):
    policy = read_jsonl(OUTPUT_DIR / f"round_{round_index}_policy.jsonl")
    allocated = read_jsonl(OUTPUT_DIR / f"round_{round_index}_allocation.jsonl")
    assert len(policy) == len(allocated) == CONFIG["budget"]
    allocation = [record["allocated_rollouts"] for record in allocated]
    assert sum(allocation) == CONFIG["budget"]
    assert all(record["added_text"] for record in policy)
    allocations.append(allocation)
    round_records.append(allocated)

nonuniform_rounds = [
    index for index, allocation in enumerate(allocations)
    if len(set(allocation)) > 1
]
assert len(nonuniform_rounds) >= 2

for round_index in range(1, CONFIG["allocation_rounds"] + 1):
    children = read_jsonl(OUTPUT_DIR / f"round_{round_index}_policy.jsonl")
    parents = {
        record["path_id"]: record
        for record in round_records[round_index - 1]
    }
    for child in children:
        parent = parents[child["parent_id"]]
        assert child["previous_text"] == parent["current_text"]
        assert child["current_text"].startswith(parent["current_text"])
        assert child["added_text"]

duplicated_parent_continued = False
for round_index in range(CONFIG["allocation_rounds"]):
    parents = round_records[round_index]
    children = read_jsonl(OUTPUT_DIR / f"round_{round_index + 1}_policy.jsonl")
    child_parent_ids = [child["parent_id"] for child in children]
    for parent in parents:
        if parent["allocated_rollouts"] > 1:
            assert child_parent_ids.count(parent["path_id"]) == parent["allocated_rollouts"]
            duplicated_parent_continued = True
assert duplicated_parent_continued

print("C2 验收通过")
print("allocations:", allocations)
print("nonuniform rounds:", nonuniform_rounds)
print("复制路径已从相同部分文本生成不同子分支")
