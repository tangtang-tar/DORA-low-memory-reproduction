"""C2 多轮搜索共享配置与 JSONL 文件函数。"""

import json
from pathlib import Path

import yaml


ROOT = Path("/media/tangtang/Data/DORA")
with (ROOT / "configs/phase_c2.yaml").open(encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)
OUTPUT_DIR = Path(CONFIG["output_dir"])


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
