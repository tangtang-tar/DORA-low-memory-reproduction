"""阶段 B：在独立进程中测量单个模型的加载与一次推理资源。"""

import argparse
import json
import resource
import time
from pathlib import Path

import psutil
import torch

from project_paths import ROOT


POLICY_PATH = ROOT / "models/Qwen2.5-1.5B-Instruct"
BGE_PATH = ROOT / "models/bge-m3"
PRM_PATH = ROOT / "models/Qwen2.5-Math-PRM-7B"
RESULT_PATH = ROOT / "results/phase_b/resource_measurements.jsonl"


def ram_metrics():
    process = psutil.Process()
    return {
        "rss_gib": process.memory_info().rss / 1024**3,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def gpu_metrics():
    if not torch.cuda.is_available():
        return {"peak_gpu_allocated_gib": 0.0, "peak_gpu_reserved_gib": 0.0}
    return {
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def measure_policy():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(POLICY_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    ).eval()
    load_seconds = time.perf_counter() - started

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve briefly: 3x + 5 = 20."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    inference_seconds = time.perf_counter() - started
    return load_seconds, inference_seconds, int(output.shape[1] - inputs.input_ids.shape[1])


def measure_bge():
    from FlagEmbedding import BGEM3FlagModel

    started = time.perf_counter()
    model = BGEM3FlagModel(str(BGE_PATH), use_fp16=True)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    output = model.encode(
        ["The answer is five.", "The result equals 5."],
        batch_size=2,
        max_length=128,
        return_dense=True,
    )
    inference_seconds = time.perf_counter() - started
    return load_seconds, inference_seconds, int(output["dense_vecs"].shape[-1])


def prm_input(tokenizer):
    steps = [
        "Subtract 5 from both sides, so 3x = 15.",
        "Divide both sides by 3, so x = 5.",
    ]
    answer = "<extra_0>".join(steps) + "<extra_0>"
    conversation = [[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "Solve 3x + 5 = 20."},
        {"role": "assistant", "content": answer},
    ]]
    return tokenizer.apply_chat_template(conversation, padding=True, return_tensors="pt")


def measure_prm(mode):
    from modeling_qwen2_rm import Qwen2ForProcessRewardModel
    from transformers import AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if mode == "prm_cpu_bf16":
        device_map = {"": "cpu"}
        max_memory = None
    elif mode == "prm_mixed_bf16":
        device_map = "auto"
        max_memory = {0: "6500MiB", "cpu": "20GiB"}
    elif mode == "prm_gpu_bf16":
        device_map = {"": "cuda"}
        max_memory = None
    elif mode == "prm_8bit_mixed":
        device_map = "auto"
        # 8-bit offload 在前向时还要把当前层搬回 GPU，因此预留约 2.4 GiB。
        max_memory = {0: "5200MiB", "cpu": "20GiB"}
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["score"],
        )
    elif mode == "prm_4bit":
        device_map = {"": "cuda"}
        max_memory = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["score"],
        )
    else:
        raise ValueError(mode)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(PRM_PATH, local_files_only=True)
    model = Qwen2ForProcessRewardModel.from_pretrained(
        PRM_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=True,
        quantization_config=quantization_config,
    ).eval()
    load_seconds = time.perf_counter() - started

    input_ids = prm_input(tokenizer).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids)[0]
    inference_seconds = time.perf_counter() - started
    step_count = int((input_ids == tokenizer.encode("<extra_0>")[0]).sum())
    return load_seconds, inference_seconds, step_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "policy_fp16",
            "bge_fp16",
            "prm_cpu_bf16",
            "prm_mixed_bf16",
            "prm_gpu_bf16",
            "prm_8bit_mixed",
            "prm_4bit",
        ],
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if args.mode == "policy_fp16":
        load_seconds, inference_seconds, output_size = measure_policy()
    elif args.mode == "bge_fp16":
        load_seconds, inference_seconds, output_size = measure_bge()
    else:
        load_seconds, inference_seconds, output_size = measure_prm(args.mode)

    result = {
        "mode": args.mode,
        "success": True,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "output_size": output_size,
        **ram_metrics(),
        **gpu_metrics(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
