"""阶段 A：验证本地 Policy 模型能在单张 8 GB 显卡上完成一次生成。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/media/tangtang/Data/DORA/models/Qwen2.5-1.5B-Instruct"

# 这里只加载 1.5B Policy；不要同时加载 7B PRM 和 BGE-M3。
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
)

messages = [
    {"role": "system", "content": "You are a careful mathematical reasoner."},
    {"role": "user", "content": "If 3x + 5 = 20, find x and explain briefly."},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

# 只解码新生成的 token，避免把输入提示词重复打印出来。
answer_ids = output_ids[0, inputs["input_ids"].shape[1] :]
print(tokenizer.decode(answer_ids, skip_special_tokens=True))
