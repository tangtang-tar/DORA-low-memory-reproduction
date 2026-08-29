"""Generate one D2 search step for all 20 problems with a single Policy load."""

import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_d2_common import CONFIG, method_dir, read_jsonl, write_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=CONFIG["methods"])
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--final", action="store_true")
args = parser.parse_args()

destination = method_dir(args.method) / f"round_{args.round}_policy.jsonl"
if destination.exists():
    print(f"resume: {destination} already exists")
    raise SystemExit(0)

dataset = Dataset.load_from_disk(CONFIG["dataset_name"])
samples = [(index, dataset[index]) for index in CONFIG["problem_indices"]]
previous_by_problem = {}
if args.round > 0:
    previous = read_jsonl(method_dir(args.method) / f"round_{args.round - 1}_allocation.jsonl")
    for record in previous:
        previous_by_problem.setdefault(record["problem_index"], []).append(record)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_path"], torch_dtype=torch.float16, device_map="cuda", local_files_only=True
).eval()

records = []
for problem_position, (problem_index, sample) in enumerate(samples):
    if args.round == 0:
        parents = [{"path_id": None, "current_text": "", "ancestry": []}] * int(CONFIG["budget"])
    else:
        parents = []
        for record in previous_by_problem[problem_index]:
            parents.extend([record] * int(record["allocated_rollouts"]))
        assert len(parents) == CONFIG["budget"]

    for slot, parent in enumerate(parents):
        current_text = parent["current_text"]
        messages = [
            {"role": "system", "content": (
                "Solve the problem step by step. Separate major reasoning steps with a blank line. "
                "When finished, end with: Therefore, the final answer is: $\\boxed{answer}$."
            )},
            {"role": "user", "content": sample["problem"]},
        ]
        if current_text:
            messages.append({"role": "assistant", "content": current_text})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
            )
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        run_seed = int(CONFIG["seed"]) + problem_position * 10000 + args.round * 100 + slot
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        kwargs = {
            "max_new_tokens": CONFIG["final_max_tokens"] if args.final else CONFIG["step_max_tokens"],
            "do_sample": True,
            "temperature": CONFIG["temperature"],
            "top_p": CONFIG["top_p"],
            "pad_token_id": tokenizer.eos_token_id,
        }
        if not args.final:
            kwargs.update(stop_strings=["\n\n"], tokenizer=tokenizer)
        with torch.inference_mode():
            output = model.generate(**inputs, **kwargs)[0]
        added_ids = output[inputs.input_ids.shape[1]:]
        added_text = tokenizer.decode(added_ids, skip_special_tokens=True)
        path_id = f"q{problem_index}_r{args.round}_p{slot}"
        records.append({
            "method": args.method, "round": args.round, "slot": slot,
            "path_id": path_id, "parent_id": parent["path_id"],
            "ancestry": parent["ancestry"] + ([parent["path_id"]] if parent["path_id"] else []),
            "problem_index": problem_index, "problem_id": sample["unique_id"],
            "problem": sample["problem"], "reference_answer": sample["answer"],
            "previous_text": current_text, "added_text": added_text,
            "current_text": current_text + added_text, "added_tokens": int(added_ids.shape[0]),
            "reached_eos": bool((added_ids == tokenizer.eos_token_id).any()),
            "is_final_generation": args.final, "seed": run_seed,
        })

write_jsonl(destination, records)
print(f"wrote {len(records)} generations to {destination}")
