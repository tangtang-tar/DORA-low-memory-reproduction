"""B3-2/B4：用 4-bit PRM 逐步评分，保存为 prm_scores.jsonl。"""

import time

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from modeling_qwen2_rm import Qwen2ForProcessRewardModel
from phase_b_common import RESULT_DIR, load_config, read_jsonl, write_jsonl


config = load_config()
input_path = RESULT_DIR / "policy_rollouts.jsonl"
records = read_jsonl(input_path)

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["score"],
)
tokenizer = AutoTokenizer.from_pretrained(config["prm_path"], local_files_only=True)
model = Qwen2ForProcessRewardModel.from_pretrained(
    config["prm_path"],
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
    quantization_config=quantization,
    low_cpu_mem_usage=True,
    local_files_only=True,
).eval()
step_separator_id = tokenizer.encode("<extra_0>")[0]

started = time.perf_counter()
for record in records:
    steps = [step for step in record["text"].split("\n\n") if step.strip()]
    marked_answer = "<extra_0>".join(steps) + "<extra_0>"
    conversation = [[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": record["problem"]},
        {"role": "assistant", "content": marked_answer},
    ]]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        logits = model(input_ids=input_ids)[0]
    mask = input_ids == step_separator_id
    scores = logits.softmax(dim=-1)[mask][:, 1].float().cpu().tolist()
    record["steps"] = steps
    record["prm_step_scores"] = scores
    record["prm_last_score"] = scores[-1]

elapsed = time.perf_counter() - started
for record in records:
    record["prm_seconds_total"] = elapsed

output_path = RESULT_DIR / "prm_scores.jsonl"
write_jsonl(output_path, records)
print(f"已评分 {len(records)} 条 rollout：{output_path}")
