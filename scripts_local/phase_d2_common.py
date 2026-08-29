"""Shared paths and JSONL helpers for resumable D2 end-to-end runs."""

import json
from pathlib import Path

from project_paths import ROOT, load_config


CONFIG = load_config("configs/phase_d2.yaml")
OUTPUT_DIR = Path(CONFIG["end_to_end_output_dir"])


def method_dir(method):
    if method not in CONFIG["methods"]:
        raise ValueError(f"unknown method: {method}")
    return OUTPUT_DIR / method


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
