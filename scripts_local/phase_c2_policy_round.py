"""C2 Policy 子进程：按上一轮配额复制路径，再生成一个步骤或最终答案。"""

import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_c2_common import CONFIG, OUTPUT_DIR, read_jsonl, write_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--final", action="store_true")
args = parser.parse_args()

dataset = Dataset.load_from_disk(CONFIG["dataset_name"])
sample = dataset[CONFIG["problem_index"]]

if args.round == 0:
    parents = [
        {
            "path_id": None,
            "current_text": "",
            "ancestry": [],
        }
        for _ in range(CONFIG["budget"])
    ]
else:
    previous = read_jsonl(OUTPUT_DIR / f"round_{args.round - 1}_allocation.jsonl")
    parents = []
    for record in previous:
        for _ in range(record["allocated_rollouts"]):
            parents.append(record)
    assert len(parents) == CONFIG["budget"]

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_path"],
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
).eval()

records = []
for path_index, parent in enumerate(parents):
    current_text = parent["current_text"]
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the problem step by step. Separate major reasoning steps with a blank line. "
                "When finished, end with: Therefore, the final answer is: $\\boxed{answer}$."
            ),
        },
        {"role": "user", "content": sample["problem"]},
    ]
    if current_text:
        messages.append({"role": "assistant", "content": current_text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.manual_seed(CONFIG["seed"] + args.round * 100 + path_index)
    torch.cuda.manual_seed_all(CONFIG["seed"] + args.round * 100 + path_index)
    generation_kwargs = {
        "max_new_tokens": CONFIG["final_max_tokens"] if args.final else CONFIG["step_max_tokens"],
        "do_sample": True,
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "pad_token_id": tokenizer.eos_token_id,
    }
    if not args.final:
        generation_kwargs["stop_strings"] = ["\n\n"]
        generation_kwargs["tokenizer"] = tokenizer
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)[0]
    added_ids = output[inputs.input_ids.shape[1] :]
    reached_eos = bool((added_ids == tokenizer.eos_token_id).any())
    added_text = tokenizer.decode(added_ids, skip_special_tokens=True)
    path_id = f"r{args.round}_p{path_index}"
    records.append(
        {
            "round": args.round,
            "path_id": path_id,
            "parent_id": parent["path_id"],
            "ancestry": parent["ancestry"] + ([parent["path_id"]] if parent["path_id"] else []),
            "problem_id": sample["unique_id"],
            "problem": sample["problem"],
            "reference_answer": sample["answer"],
            "previous_text": current_text,
            "added_text": added_text,
            "current_text": current_text + added_text,
            "added_tokens": int(added_ids.shape[0]),
            "reached_eos": reached_eos,
            "is_final_generation": args.final,
        }
    )

output_path = OUTPUT_DIR / f"round_{args.round}_policy.jsonl"
write_jsonl(output_path, records)
print(f"round {args.round} Policy：{len(records)} 条，final={args.final}，{output_path}")
