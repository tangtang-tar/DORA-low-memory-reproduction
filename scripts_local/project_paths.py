"""Portable project-root and YAML path handling for experiment scripts."""

import os
from pathlib import Path

import yaml


ROOT = Path(
    os.environ.get("DORA_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()

PATH_KEYS = {
    "candidate_path",
    "dataset_name",
    "embedding_path",
    "model_path",
    "output_dir",
    "prm_path",
}


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(relative_path):
    """Load YAML and resolve known filesystem fields against the project root."""
    with (ROOT / relative_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in PATH_KEYS & config.keys():
        config[key] = str(resolve_project_path(config[key]))
    return config
