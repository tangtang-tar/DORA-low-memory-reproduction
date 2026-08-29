#!/usr/bin/env bash

# 默认以本脚本所在目录为项目根；也可在 source 前显式设置 DORA_ROOT。
_dora_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DORA_ROOT=${DORA_ROOT:-$_dora_script_dir}

# 将各工具默认写入用户主目录的缓存改到 DORA 目录。
export CONDA_PKGS_DIRS="$DORA_ROOT/cache/conda_pkgs"
export CONDA_ENVS_PATH="$DORA_ROOT/envs"
export PIP_CACHE_DIR="$DORA_ROOT/cache/pip"
export HF_HOME="$DORA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$DORA_ROOT/cache/huggingface/hub"
export HF_DATASETS_CACHE="$DORA_ROOT/cache/huggingface/datasets"
export TORCH_HOME="$DORA_ROOT/cache/torch"
export TRITON_CACHE_DIR="$DORA_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$DORA_ROOT/cache/vllm"
export CUDA_CACHE_PATH="$DORA_ROOT/cache/cuda"
export XDG_CACHE_HOME="$DORA_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$DORA_ROOT/cache/xdg-config"
export XDG_DATA_HOME="$DORA_ROOT/cache/xdg-data"
export CONDARC="$DORA_ROOT/configs/condarc"
export TMPDIR="$DORA_ROOT/tmp"

# 让 Python 能找到官方仓库 scripts/sal 包。
export PYTHONPATH="$DORA_ROOT/src/scripts${PYTHONPATH:+:$PYTHONPATH}"

# 初始化 Conda shell hook，再激活项目内环境。
if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda is unavailable; initialize Conda or set CONDA_EXE first." >&2
    return 1
fi
conda activate "$DORA_ROOT/envs/dora" || return 1

cd "$DORA_ROOT/src"
unset _dora_script_dir
