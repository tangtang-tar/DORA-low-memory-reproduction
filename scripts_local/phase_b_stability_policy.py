"""B6-1：为 5 道题各生成 4 条固定 rollout。"""

import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_b_stability_common import load_config, write_jsonl


config = load_config()
output_dir = Path(config["output_dir"])
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
                "Solve the problem in at most 3 concise steps and 160 words. "
                "Separate steps with a blank line. The last line must be exactly: "
                "Therefore, the final answer is: $\\boxed{answer}$."
            ),
        },
        {"role": "user", "content": sample["problem"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt] * config["n"], return_tensors="pt", padding=True).to("cuda")
    torch.manual_seed(config["seed"] + problem_index)
    torch.cuda.manual_seed_all(config["seed"] + problem_index)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_tokens"],
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )
    problem_seconds = time.perf_counter() - started
    prompt_length = inputs.input_ids.shape[1]
    for rollout_id, output in enumerate(outputs):
        text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        records.append(
            {
                "problem_index": problem_index,
                "problem_id": sample["unique_id"],
                "problem": sample["problem"],
                "reference_answer": sample["answer"],
                "rollout_id": rollout_id,
                "text": text,
                "generation_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "policy_seconds_problem": problem_seconds,
            }
        )

elapsed = time.perf_counter() - started_all
for record in records:
    record["policy_seconds_all"] = elapsed

output_path = output_dir / "policy_rollouts.jsonl"
write_jsonl(output_path, records)
print(f"生成 {len(dataset)} 题 × {config['n']} rollout = {len(records)} 条：{output_path}")
print(f"Policy 总生成时间：{elapsed:.2f} 秒")
