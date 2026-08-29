"""B3-1：Policy 生成初始 rollout，并保存为 policy_rollouts.jsonl。"""

import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_b_common import RESULT_DIR, load_config, write_jsonl


config = load_config()
dataset = Dataset.load_from_disk(config["dataset_name"])
sample = dataset.select(range(config["dataset_start"], config["dataset_end"]))[0]

tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    config["model_path"],
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
).eval()

messages = [
    {
        "role": "system",
        "content": (
            "Solve the problem in at most 3 short steps and 120 words. "
            "Separate steps with a blank line. The last line must be exactly in this form: "
            "Therefore, the final answer is: $\\boxed{answer}$."
        ),
    },
    {"role": "user", "content": sample["problem"]},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompts = [prompt] * config["n"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
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
elapsed = time.perf_counter() - started

records = []
prompt_length = inputs.input_ids.shape[1]
for rollout_id, output in enumerate(outputs):
    generated = output[prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    records.append(
        {
            "problem_id": sample["unique_id"],
            "problem": sample["problem"],
            "reference_answer": sample["answer"],
            "rollout_id": rollout_id,
            "text": text,
            "generation_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "policy_seconds_total": elapsed,
        }
    )

output_path = RESULT_DIR / "policy_rollouts.jsonl"
write_jsonl(output_path, records)
print(f"已写入 {len(records)} 条 rollout：{output_path}")
