"""在完全相同的冻结候选上运行一种 PRM 精度，输出逐步分数。"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from modeling_qwen2_rm import Qwen2ForProcessRewardModel
from project_paths import load_config


config = load_config("configs/phase_c3.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["4bit", "mixed_bf16"], required=True)
args = parser.parse_args()

quantization = None
if args.mode == "4bit":
    device_map = {"": "cuda"}
    max_memory = None
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["score"],
    )
else:
    # 8 GiB 显卡无法完整容纳 7B BF16；Accelerate 把超出的层放在内存。
    device_map = "auto"
    max_memory = {0: "6500MiB", "cpu": "20GiB"}

tokenizer = AutoTokenizer.from_pretrained(config["prm_path"], local_files_only=True)
model = Qwen2ForProcessRewardModel.from_pretrained(
    config["prm_path"],
    device_map=device_map,
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization,
    low_cpu_mem_usage=True,
    local_files_only=True,
).eval()
separator_id = tokenizer.encode("<extra_0>")[0]

with Path(config["candidate_path"]).open(encoding="utf-8") as file:
    candidates = [json.loads(line) for line in file if line.strip()]

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"scores_{args.mode}.jsonl"
with output_path.open("w", encoding="utf-8") as output:
    for index, candidate in enumerate(candidates, start=1):
        steps = [step.strip() for step in candidate["text"].split("\n\n") if step.strip()]
        marked_answer = "<extra_0>".join(steps) + "<extra_0>"
        conversation = [[
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": candidate["problem"]},
            {"role": "assistant", "content": marked_answer},
        ]]
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids)[0]
        scores = logits.softmax(dim=-1)[input_ids == separator_id][:, 1].float().cpu().tolist()
        record = {
            "candidate_id": candidate["candidate_id"],
            "problem_index": candidate["problem_index"],
            "mode": args.mode,
            "step_count": len(steps),
            "prm_step_scores": scores,
            "prm_last_score": scores[-1],
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{args.mode} {index:02d}/{len(candidates)} {candidate['candidate_id']}: {scores[-1]:.6f}", flush=True)
