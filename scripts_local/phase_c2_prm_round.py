"""C2 PRM 子进程：用 4-bit PRM 给当前部分路径逐步评分。"""

import argparse

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from modeling_qwen2_rm import Qwen2ForProcessRewardModel
from phase_c2_common import CONFIG, OUTPUT_DIR, read_jsonl, write_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True)
args = parser.parse_args()
records = read_jsonl(OUTPUT_DIR / f"round_{args.round}_policy.jsonl")

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["score"],
)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["prm_path"], local_files_only=True)
model = Qwen2ForProcessRewardModel.from_pretrained(
    CONFIG["prm_path"],
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
    quantization_config=quantization,
    low_cpu_mem_usage=True,
    local_files_only=True,
).eval()
separator_id = tokenizer.encode("<extra_0>")[0]

for record in records:
    steps = [step for step in record["current_text"].split("\n\n") if step.strip()]
    marked_answer = "<extra_0>".join(steps) + "<extra_0>"
    conversation = [[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": record["problem"]},
        {"role": "assistant", "content": marked_answer},
    ]]
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(input_ids=input_ids)[0]
    scores = logits.softmax(dim=-1)[input_ids == separator_id][:, 1].float().cpu().tolist()
    record["steps"] = steps
    record["prm_step_scores"] = scores
    record["prm_last_score"] = scores[-1]

output_path = OUTPUT_DIR / f"round_{args.round}_prm.jsonl"
write_jsonl(output_path, records)
print(f"round {args.round} PRM：{[round(r['prm_last_score'], 4) for r in records]}")
