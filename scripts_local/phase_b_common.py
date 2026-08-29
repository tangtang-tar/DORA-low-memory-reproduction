"""阶段 B 三段流水线共享的少量文件与配置函数。"""

import json
from pathlib import Path

from project_paths import ROOT, load_config as load_project_config
CONFIG_PATH = ROOT / "configs/phase_b_dora.yaml"
RESULT_DIR = ROOT / "results/phase_b"


def load_config():
    return load_project_config("configs/phase_b_dora.yaml")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
