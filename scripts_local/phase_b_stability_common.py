"""B6 稳定性实验共享路径和 JSONL 函数。"""

import json
from pathlib import Path

import yaml


ROOT = Path("/media/tangtang/Data/DORA")
CONFIG_PATH = ROOT / "configs/phase_b_stability.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
