"""C1：生成并冻结 40 条用于人工标注和提取器验证的候选。"""

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


from project_paths import load_config


config = load_config("configs/phase_c1.yaml")
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

dataset = Dataset.load_from_disk(config["dataset_name"]).select(range(config["num_samples"]))
tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    config["model_path"],
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
).eval()

records = []
started_all = time.perf_counter()
for problem_index, sample in enumerate(dataset):
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the problem step by step, using at most 160 words. "
                "End with exactly one final line: Therefore, the final answer is: $\\boxed{answer}$."
            ),
        },
        {"role": "user", "content": sample["problem"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt] * config["n"], return_tensors="pt", padding=True).to("cuda")
    torch.manual_seed(config["seed"] + problem_index)
    torch.cuda.manual_seed_all(config["seed"] + problem_index)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_tokens"],
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs.input_ids.shape[1]
    for rollout_id, output in enumerate(outputs):
        generated = output[prompt_length:]
        eos_positions = (generated == tokenizer.eos_token_id).nonzero(as_tuple=False)
        reached_eos = len(eos_positions) > 0
        content_length = int(eos_positions[0].item()) if reached_eos else int(generated.shape[0])
        content_ids = generated[:content_length]
        text = tokenizer.decode(content_ids, skip_special_tokens=True)
        records.append(
            {
                "candidate_id": f"p{problem_index:02d}_r{rollout_id}",
                "problem_index": problem_index,
                "rollout_id": rollout_id,
                "problem_id": sample["unique_id"],
                "problem": sample["problem"],
                "reference_answer": sample["answer"],
                "text": text,
                "generation_tokens": content_length,
                "reached_eos": reached_eos,
                "is_truncated": not reached_eos,
            }
        )

output_path = output_dir / "candidates.jsonl"
with output_path.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"冻结评测集：{len(records)} 条，{output_path}")
print(f"明确截断：{sum(record['is_truncated'] for record in records)} 条")
print(f"生成耗时：{time.perf_counter() - started_all:.2f} 秒")
